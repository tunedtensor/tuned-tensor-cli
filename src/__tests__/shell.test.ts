import { describe, expect, it, vi } from "vitest";
import { EventEmitter } from "node:events";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { stripVTControlCharacters } from "node:util";
import { tmpdir } from "node:os";
import { join } from "node:path";
import chalk from "chalk";
import {
  ShellParseError,
  createShellSession,
  isCatalogCommand,
  parseSlashCommand,
  renderShellBanner,
  renderShellPrompt,
  renderSubmittedShellInput,
  resetShellPromptStyle,
  routeShellCommand,
  shouldStartInteractiveShell,
  tokenizeShellInput,
  withForegroundSignalHandoff,
  type ShellCommandRequest,
  type ShellSessionIO,
} from "../shell.js";
import { createCommandCompleter } from "../command-catalog.js";
import type { ShellContext } from "../shell-context.js";
import type { LiveUsage } from "../local-runtime/live-usage.js";
import { terminalWidth } from "../terminal-markdown.js";

describe("tokenizeShellInput", () => {
  it("parses whitespace, quotes, escapes, empty values, and joined fragments", () => {
    expect(
      tokenizeShellInput(
        String.raw`runs report "run one" --label='hello world' escaped\ value "" pre"mid"post`,
      ),
    ).toEqual([
      "runs",
      "report",
      "run one",
      "--label=hello world",
      "escaped value",
      "",
      "premidpost",
    ]);
  });

  it("treats quoted and escaped punctuation as literal argv", () => {
    expect(
      tokenizeShellInput(String.raw`label edit "a | b; c" escaped\>value '$HOME $(literal)'`),
    ).toEqual([
      "label",
      "edit",
      "a | b; c",
      "escaped>value",
      "$HOME $(literal)",
    ]);
  });

  it.each([
    "runs list | cat",
    "runs list > out",
    "runs list < in",
    "runs list && models list",
    "runs list || models list",
    "runs list &",
    "runs list ; models list",
    "runs get $(whoami)",
    "runs get `whoami`",
    "!nvidia-smi",
  ])("rejects shell syntax: %s", (input) => {
    expect(() => tokenizeShellInput(input)).toThrow(ShellParseError);
  });

  it.each([
    ["unterminated 'quote", /unterminated single quote/],
    ["unterminated \"quote", /unterminated double quote/],
    ["dangling\\", /unfinished escape/],
    ["runs\0list", /NUL/],
    ["runs\nlist", /one command at a time/],
  ])("reports malformed input clearly", (input, message) => {
    expect(() => tokenizeShellInput(input)).toThrow(message);
  });
});

describe("routeShellCommand", () => {
  it("routes ordinary input without a workflow prefix", () => {
    expect(routeShellCommand("runs list --json")).toEqual({
      args: ["runs", "list", "--json"],
    });
  });

  it("strips a redundant tt prefix", () => {
    expect(routeShellCommand("tt doctor")).toEqual({
      args: ["doctor"],
    });
  });

  it("strips the hidden local alias so the command runs at the top level", () => {
    expect(routeShellCommand("local runs list")).toEqual({
      args: ["runs", "list"],
    });
    expect(routeShellCommand("tt local doctor")).toEqual({
      args: ["doctor"],
    });
  });

  it("does not open a nested shell for a bare tt command", () => {
    expect(() => routeShellCommand("tt")).toThrow(/already open/);
  });
});

describe("isCatalogCommand", () => {
  it("recognizes existing CLI grammar without treating conversation as commands", () => {
    expect(isCatalogCommand(["runs", "list"])).toBe(true);
    expect(isCatalogCommand(["doctor"])).toBe(true);
    expect(isCatalogCommand(["hardware"])).toBe(true);
    expect(isCatalogCommand(["local", "runs", "list"])).toBe(true);
    expect(isCatalogCommand(["local", "doctor"])).toBe(true);
    expect(isCatalogCommand(["show", "my", "latest", "run"])).toBe(false);
    expect(isCatalogCommand(["runs", "please"])).toBe(false);
    expect(isCatalogCommand(["status", "of", "my", "run"])).toBe(false);
    expect(isCatalogCommand(["status"])).toBe(true);
  });
});

describe("parseSlashCommand", () => {
  it("parses the palette, slash commands, and question-mark help alias", () => {
    expect(parseSlashCommand("/")).toEqual({ name: "palette", args: [] });
    expect(parseSlashCommand("/model")).toEqual({ name: "model", args: [] });
    expect(parseSlashCommand("? runs")).toEqual({
      name: "help",
      args: ["runs"],
    });
  });

  it("rejects unknown slash commands and operators", () => {
    expect(() => parseSlashCommand("/wat")).toThrow(/Unknown session command/);
    expect(() => parseSlashCommand("/help | cat")).toThrow(/operator/);
  });

  it("rejects the removed mode-switching slash commands", () => {
    expect(() => parseSlashCommand("/mode cloud")).toThrow(/Unknown session command/);
    expect(() => parseSlashCommand("/cloud")).toThrow(/Unknown session command/);
    expect(() => parseSlashCommand("/local")).toThrow(/Unknown session command/);
  });

  it("parses /model and /login and suggests fixes for mistyped session commands", () => {
    expect(parseSlashCommand("/model")).toEqual({ name: "model", args: [] });
    expect(parseSlashCommand("/model abc123")).toEqual({
      name: "model",
      args: ["abc123"],
    });
    expect(parseSlashCommand("/login")).toEqual({ name: "login", args: [] });
    expect(parseSlashCommand("/login openrouter")).toEqual({
      name: "login",
      args: ["openrouter"],
    });
    expect(() => parseSlashCommand("/stat")).toThrow(/Did you mean \/status\?/);
    expect(() => parseSlashCommand("/models")).toThrow(/need no slash/);
  });
});

describe("command completion", () => {
  it("completes local, cloud, account, and slash commands", () => {
    const complete = createCommandCompleter();

    expect(complete("runs c")[0]).toContain("runs compare");
    expect(complete("/mo")[0]).toEqual(["/model"]);
    expect(complete("/lo")[0]).toEqual(["/login"]);
    expect(complete("cl")[0]).toContain("cloud runs list");
    expect(complete("auth")[0]).toContain("auth login");
    expect(complete("publish")[0]).toEqual(["publish"]);
    expect(complete("pipeline")[0]).toContain("pipeline init");
    expect(complete("doctor")[0]).toContain("doctor");
  });
});

describe("shouldStartInteractiveShell", () => {
  const base = {
    args: [] as string[],
    stdinIsTTY: true,
    stdoutIsTTY: true,
    env: { TERM: "xterm-256color" },
  };

  it("starts only for a bare human TTY invocation", () => {
    expect(shouldStartInteractiveShell(base)).toBe(true);
    expect(shouldStartInteractiveShell({ ...base, args: ["--help"] })).toBe(false);
    expect(shouldStartInteractiveShell({ ...base, stdinIsTTY: false })).toBe(false);
    expect(shouldStartInteractiveShell({ ...base, env: { CI: "1" } })).toBe(false);
    expect(shouldStartInteractiveShell({ ...base, env: { CI: "false" } })).toBe(true);
    expect(shouldStartInteractiveShell({ ...base, env: { TERM: "dumb" } })).toBe(false);
  });
});

describe("foreground SIGINT handoff", () => {
  it("keeps the parent pending, releases readline raw mode, and leaves SIGINT visible to the foreground owner", async () => {
    const signals = new EventEmitter();
    const childSawSigint = vi.fn();
    signals.on("SIGINT", childSawSigint);
    const pauseReadline = vi.fn();
    const resumeReadline = vi.fn();
    const input = {
      isRaw: true,
      setRawMode(mode: boolean) {
        this.isRaw = mode;
        return this;
      },
    };
    let release!: (value: string) => void;
    const child = new Promise<string>((resolve) => {
      release = resolve;
    });
    let settled = false;

    const pending = withForegroundSignalHandoff(
      {
        input: input as never,
        pauseReadline,
        resumeReadline,
        signals,
      },
      () => child,
    ).finally(() => {
      settled = true;
    });

    expect(pauseReadline).toHaveBeenCalledTimes(1);
    expect(input.isRaw).toBe(false);
    expect(signals.listenerCount("SIGINT")).toBe(2);

    signals.emit("SIGINT");
    await Promise.resolve();
    expect(childSawSigint).toHaveBeenCalledTimes(1);
    expect(settled).toBe(false);

    release("done");
    await expect(pending).resolves.toBe("done");
    expect(input.isRaw).toBe(true);
    expect(resumeReadline).toHaveBeenCalledTimes(1);
    expect(signals.listenerCount("SIGINT")).toBe(1);
  });
});

function fakeContext(cwd: string): ShellContext {
  return {
    cwd,
    projectName: cwd.split("/").filter(Boolean).at(-1) ?? cwd,
    local: {
      configPath: `${cwd}/local-runner.json`,
      artifactRoot: `${cwd}/.tuned-tensor/artifacts`,
      storeRoot: `${cwd}/.tuned-tensor/store`,
      activeModelId: "model_abc123",
    },
    warnings: [],
  };
}

const liveUsage: LiveUsage = {
  sampled_at: "2026-01-01T00:00:00.000Z",
  cpu: { model: "Ryzen 9 7950X", cores: 16, utilization_percent: 18, load_average: [2.1, 1.8, 1.5] },
  memory: { used_bytes: 26 * 1024 ** 3, total_bytes: 64 * 1024 ** 3 },
  gpus: [{
    index: 0,
    name: "NVIDIA GeForce RTX 4090",
    utilization_percent: 42,
    memory_used_bytes: 6 * 1024 ** 3,
    memory_total_bytes: 24 * 1024 ** 3,
    temperature_c: 55,
    unified_memory: false,
    vendor: "nvidia",
  }],
};

function specContext(cwd: string, spec: ShellContext["spec"]): ShellContext {
  return {
    ...fakeContext(cwd),
    spec,
    agent: { provider: "anthropic", model: "claude-sonnet-4-5" },
  };
}

describe("renderShellBanner", () => {
  it("shows Tardi, the heading, context, controls, and version", () => {
    const banner = renderShellBanner({
      mode: "local",
      modeSource: "default-local",
      cwd: "/tmp/local-project",
      context: fakeContext("/tmp/local-project"),
      version: "0.6.0",
    });
    const rows = banner.trimEnd().split("\n");
    expect(rows[0]).toContain("tt");
    expect(rows[0]).toContain("╭─────╮");
    expect(banner).toContain("v0.6.0");
    expect(banner).toContain("Tardi");
    expect(banner).toContain("agent not configured");
    expect(banner).toContain("workflow model model_abc123");
    expect(banner).toContain("ctrl+c stop/clear");
    expect(banner).toContain("/system machine");
    expect(banner).toContain("Use /login tunedtensor for managed inference");
    expect(banner).toContain("Workflow commands work now");
    expect(banner).not.toContain("MACHINE");
  });

  it("points at the spec once a provider and model are selected", () => {
    const banner = renderShellBanner({
      mode: "local",
      modeSource: "default-local",
      cwd: "/tmp/local-project",
      context: specContext("/tmp/local-project", {
        path: "/tmp/local-project/tunedtensor.json",
        name: "Support triage",
        baseModel: "Qwen/Qwen3.5-2B",
        exampleCount: 120,
        parseError: false,
      }),
      version: "0.6.0",
    });
    expect(banner).toContain("spec Support triage · Qwen/Qwen3.5-2B · 120 examples");
    expect(banner).toContain("Ask TT anything. Known commands run directly.");
    expect(banner).not.toContain("Use /login tunedtensor");
  });

  it.each([
    [undefined, /No tunedtensor\.json here yet/],
    [{ path: "/p/tunedtensor.json", parseError: true }, /doesn't parse/],
    [{ path: "/p/tunedtensor.json", name: "Tiny", exampleCount: 4, parseError: false }, /4 examples is a small set/],
  ])("lets Tardi suggest the next step for %o", (spec, message) => {
    const banner = renderShellBanner({
      mode: "local",
      modeSource: "default-local",
      cwd: "/p",
      context: specContext("/p", spec as ShellContext["spec"]),
    });
    expect(banner).toMatch(message);
  });

  it("adds live GPU, CPU and memory use with a fine-tune verdict", () => {
    const banner = renderShellBanner({
      mode: "local",
      modeSource: "default-local",
      cwd: "/p",
      context: specContext("/p", {
        path: "/p/tunedtensor.json",
        name: "Tiny",
        baseModel: "Qwen/Qwen3.5-2B",
        exampleCount: 40,
        parseError: false,
      }),
      usage: liveUsage,
    }, 100);
    expect(banner).toContain("MACHINE");
    expect(banner).toMatch(/GPU\s+GeForce RTX 4090/);
    expect(banner).toContain("42%");
    expect(banner).toContain("VRAM 6.0/24.0 GiB");
    expect(banner).toContain("16× Ryzen 9 7950X");
    expect(banner).toContain("26.0/64.0 GiB");
    expect(banner).toMatch(/LoRA Qwen3\.5-2B ready/);
    expect(banner).toContain("quick check");
  });

  it("omits the version when none is provided", () => {
    const banner = renderShellBanner({
      mode: "local",
      modeSource: "default-local",
      cwd: "/tmp/local-project",
      context: fakeContext("/tmp/local-project"),
    });
    expect(banner).toContain("tt");
    expect(banner).not.toContain("v0");
  });

  it("keeps every row within a narrow terminal", () => {
    const banner = renderShellBanner({
      mode: "local",
      modeSource: "default-local",
      cwd: "/p",
      context: fakeContext("/p"),
      usage: liveUsage,
      version: "0.6.0",
    }, 50);
    for (const row of banner.split("\n")) {
      expect(terminalWidth(row)).toBeLessThanOrEqual(50);
    }
  });
});

describe("renderShellPrompt", () => {
  it("keeps the active readline prompt self-contained", () => {
    const originalLevel = chalk.level;
    chalk.level = 3;
    try {
      const prompt = renderShellPrompt();
      expect(prompt).toContain("›");
      expect(prompt).not.toContain("\u001b[48;");
      expect(prompt).not.toContain("\u001b[K");
      expect(resetShellPromptStyle()).toBe("");
    } finally {
      chalk.level = originalLevel;
    }
  });

  it.each([
    [1, "\u001b[100m"],
    [2, "\u001b[48;5;238m"],
    [3, "\u001b[48;2;50;52;67m"],
  ] as const)("repaints submitted input at color level %i", (level, background) => {
    const originalLevel = chalk.level;
    chalk.level = level;
    try {
      const submitted = renderSubmittedShellInput("hello");
      expect(submitted).toContain("›");
      expect(submitted).toContain("hello");
      expect(submitted).toContain(background);
      expect(submitted).toContain("\u001b[K\u001b[0m\r\n");
      expect(submitted).toMatch(/^\u001b\[1A\r\u001b\[2K/);
      expect(renderSubmittedShellInput("too long", 5)).toBe("");
      const wrapped = renderSubmittedShellInput("a question that wraps", 10);
      expect(wrapped).toMatch(/^\u001b\[3A\r\u001b\[J/);
      expect(wrapped).toContain("a question that wraps");
      expect(renderSubmittedShellInput("safe\u001b[31m", 80)).not.toContain("[31m");
    } finally {
      chalk.level = originalLevel;
    }
  });

  it("falls back to a plain prompt when color is disabled", () => {
    const originalLevel = chalk.level;
    chalk.level = 0;
    try {
      expect(renderShellPrompt()).toBe("› ");
      expect(renderSubmittedShellInput("hello")).toBe("");
      expect(resetShellPromptStyle()).toBe("");
    } finally {
      chalk.level = originalLevel;
    }
  });
});

describe("TunedTensorShellSession", () => {  it("routes commands locally and recovers from parse errors", async () => {
    const requests: ShellCommandRequest[] = [];
    const stdout: string[] = [];
    const stderr: string[] = [];
    const io: ShellSessionIO = {
      write: (text) => stdout.push(text),
      writeError: (text) => stderr.push(text),
      clear: vi.fn(),
    };
    const session = await createShellSession({
      cwd: "/tmp/local-project",
      env: {},
      io,
      runner: async (request) => {
        requests.push(request);
        return { exitCode: 0 };
      },
      contextProvider: async ({ cwd }) => fakeContext(cwd),
    });

    expect(session.snapshot().mode).toBe("local");
    await session.handleLine("runs list");
    await session.handleLine("/mode cloud");
    await session.handleLine("runs list");
    await session.handleLine("runs list | cat");

    expect(requests).toEqual([
      { args: ["runs", "list"], cwd: "/tmp/local-project" },
      { args: ["runs", "list"], cwd: "/tmp/local-project" },
    ]);
    expect(stderr.join("")).toMatch(/Unknown session command/);
    expect(stderr.join("")).toMatch(/Shell operator/);
  });

  it("stays local even when the discovered context favours cloud", async () => {
    const stdout: string[] = [];
    const stderr: string[] = [];
    const session = await createShellSession({
      cwd: "/tmp/cloud-project",
      env: {},
      io: {
        write: (text) => stdout.push(text),
        writeError: (text) => stderr.push(text),
        clear: vi.fn(),
      },
      runner: async () => ({ exitCode: 0 }),
      contextProvider: async ({ cwd }) => fakeContext(cwd),
    });

    expect(session.snapshot().mode).toBe("local");

    await session.handleLine("/local");
    await session.handleLine("/cloud");
    await session.handleLine("/mode local");

    expect(session.snapshot().mode).toBe("local");
    expect(stdout.join("")).not.toContain("Workflow switched");
    expect(stderr.join("")).toMatch(/Unknown session command/);
  });

  it("sends natural language and agent slash commands to the in-shell agent", async () => {
    const run = vi.fn(async (_request: ShellCommandRequest) => ({
      exitCode: 0,
    }));
    const agent = {
      busy: false,
      handleLine: vi.fn(async (
        _input: string,
        _context?: { mode: "cloud" | "local"; workspaceRoot: string },
      ) => "continue" as const),
      interrupt: vi.fn(() => false),
    };
    const writeError = vi.fn();
    const session = await createShellSession({
      cwd: "/tmp/cloud-project",
      env: {},
      io: {
        write: vi.fn(),
        writeError,
        clear: vi.fn(),
      },
      runner: run,
      agent,
      contextProvider: async ({ cwd }) => fakeContext(cwd),
    });

    await session.handleLine("What happened in my latest training run?");
    await session.handleLine("/new");
    await session.handleLine("/approve action-123");
    await session.handleLine("Compare accuracy > latency & cost; summarize it.");
    await session.handleLine("status of my latest run");
    await session.handleLine("runs list");
    await session.handleLine("local runs list");
    await session.handleLine("runs list | cat");
    await session.handleLine(": doctor");
    await session.handleLine("cloud not-a-command");

    expect(agent.handleLine.mock.calls.map((call) => call[0])).toEqual([
      "What happened in my latest training run?",
      "/new",
      "/approve action-123",
      "Compare accuracy > latency & cost; summarize it.",
      "status of my latest run",
      "cloud not-a-command",
    ]);
    expect(agent.handleLine.mock.calls.map((call) => call[1])).toEqual([
      { mode: "local", workspaceRoot: "/tmp/cloud-project" },
      { mode: "local", workspaceRoot: "/tmp/cloud-project" },
      { mode: "local", workspaceRoot: "/tmp/cloud-project" },
      { mode: "local", workspaceRoot: "/tmp/cloud-project" },
      { mode: "local", workspaceRoot: "/tmp/cloud-project" },
      { mode: "local", workspaceRoot: "/tmp/cloud-project" },
    ]);
    expect(run.mock.calls.map((call) => call[0])).toEqual([
      {
        args: ["runs", "list"],
        cwd: "/tmp/cloud-project",
      },
      {
        args: ["runs", "list"],
        cwd: "/tmp/cloud-project",
      },
      {
        args: ["doctor"],
        cwd: "/tmp/cloud-project",
      },
    ]);
    expect(writeError).toHaveBeenCalledWith(expect.stringMatching(/Shell operator/));
  });

  it("implements help, context, status, clear, and exit without running work", async () => {
    const run = vi.fn();
    const clear = vi.fn();
    const stdout: string[] = [];
    const session = await createShellSession({
      cwd: "/tmp/cloud-project",
      env: {},
      io: {
        write: (text) => stdout.push(text),
        writeError: vi.fn(),
        clear,
      },
      runner: run,
      contextProvider: async ({ cwd }) => fakeContext(cwd),
    });

    await session.handleLine("/help runs");
    await session.handleLine("/context");
    await session.handleLine("/status");
    await session.handleLine("/clear");
    expect(await session.handleLine("exit")).toBe("exit");
    expect(await session.handleLine("/exit")).toBe("exit");

    const output = stdout.join("");
    expect(output).toContain("TT commands matching");
    expect(output).not.toContain("Cloud endpoint");
    expect(output).toContain("Host           not inventoried");
    expect(clear).toHaveBeenCalledTimes(1);
    expect(run).not.toHaveBeenCalled();
  });

  it("lists, searches, and changes the TT agent model through /model", async () => {
    const configRoot = mkdtempSync(join(tmpdir(), "tt-shell-model-"));
    process.env.TUNED_TENSOR_HOME = configRoot;
    const stdout: string[] = [];
    const stderr: string[] = [];
    const models = [
      { id: "claude-sonnet-4-5", provider: "anthropic", name: "Claude Sonnet 4.5", reasoning: true },
      { id: "claude-haiku-4-5", provider: "anthropic", name: "Claude Haiku 4.5", reasoning: false },
      { id: "gpt-5.2", provider: "openai", name: "GPT 5.2", reasoning: true },
      { id: "gpt-5.6-sol", provider: "openai", name: "GPT-5.6 Sol", reasoning: true },
      {
        id: "anthropic/claude-sonnet-5",
        provider: "openrouter",
        name: "Anthropic: Claude Sonnet 5",
        reasoning: true,
      },
      { id: "llama-3.3-70b", provider: "groq", name: "Llama 3.3 70B", reasoning: false },
    ];
    const modelRuntime = {
      getProviders: () => [
        { id: "anthropic", name: "Anthropic" },
        { id: "openai", name: "OpenAI" },
        { id: "openrouter", name: "OpenRouter" },
        { id: "groq", name: "Groq" },
      ],
      getModels: (provider?: string) =>
        provider ? models.filter((model) => model.provider === provider) : models,
      getModel: (provider: string, model: string) =>
        models.find((candidate) => candidate.provider === provider && candidate.id === model),
      hasConfiguredAuth: (provider: string) => provider === "anthropic",
    };
    const session = await createShellSession({
      cwd: "/tmp/local-project",
      env: { HOME: "/tmp/home" },
      io: {
        write: (text) => stdout.push(text),
        writeError: (text) => stderr.push(text),
        clear: vi.fn(),
      },
      runner: async () => ({ exitCode: 0 }),
      agentModelRuntime: async () => modelRuntime,
      contextProvider: async ({ cwd }) => fakeContext(cwd),
    });

    const run = async (line: string): Promise<string> => {
      stdout.length = 0;
      await session.handleLine(line);
      return stdout.join("");
    };

    try {
      let output = await run("/model");
      expect(output).toContain("Agent model");
      expect(output).toContain("Providers");
      expect(output).toContain("openai");
      expect(output).toContain("openrouter");
      const providersSection = output.split("Suggestions")[0]!;
      expect(providersSection).not.toContain("anthropic");
      expect(output).not.toContain("groq");
      expect(output).toContain("Other providers: /login <id> or /model <id>");
      expect(output).toMatch(/openai[\s\S]*auth required/);
      expect(output).toContain("Use /login tunedtensor for your TT token, or /login <provider> for a BYO key.");
      expect(output).toContain("/login <provider>");
      expect(output).toContain("Suggestions");
      expect(output).toContain("openai/gpt-5.6-sol");
      expect(output).toContain("openrouter/anthropic/claude-sonnet-5");
      expect(output).not.toContain("anthropic/claude-sonnet-4-5");
      expect(output).not.toContain("openai/gpt-5.2");
      expect(output).not.toContain("groq/llama-3.3-70b");
      expect(output).not.toMatch(/… \d+ more/);
      expect(output).toContain("Use /model <provider> to list models");

      output = await run("/model groq");
      expect(output).toContain("Models from groq");
      expect(output).toContain("groq/llama-3.3-70b");

      output = await run("/model anthropic");
      expect(output).toContain("Models from anthropic");
      expect(output).toContain("anthropic/claude-sonnet-4-5");
      expect(output).toContain("anthropic/claude-haiku-4-5");
      expect(output).not.toContain("openai/gpt-5.2");

      output = await run("/model sonnet");
      expect(output).toContain('Models matching "sonnet"');
      expect(output).toContain("anthropic/claude-sonnet-4-5");
      expect(output).not.toContain("anthropic/claude-haiku-4-5");

      output = await run("/model anthropic/claude-sonnet-4-5");
      expect(output).toContain("Agent model: anthropic/claude-sonnet-4-5 (thinking medium).");

      output = await run("/model anthropic/claude-haiku-4-5");
      expect(output).toContain("Agent model: anthropic/claude-haiku-4-5 (thinking off)");
      expect(output).toContain("thinking set to off for this model");
    } finally {
      delete process.env.TUNED_TENSOR_HOME;
      rmSync(configRoot, { recursive: true, force: true });
    }
  });

  it("notes when /model <provider> truncates a long catalog", async () => {
    const configRoot = mkdtempSync(join(tmpdir(), "tt-shell-model-trunc-"));
    process.env.TUNED_TENSOR_HOME = configRoot;
    const stdout: string[] = [];
    const models = Array.from({ length: 21 }, (_, index) => ({
      id: `model-${String(index + 1).padStart(2, "0")}`,
      provider: "anthropic",
      name: `Model ${index + 1}`,
      reasoning: true,
    }));
    const session = await createShellSession({
      cwd: "/tmp/local-project",
      env: { HOME: "/tmp/home" },
      io: {
        write: (text) => stdout.push(text),
        writeError: vi.fn(),
        clear: vi.fn(),
      },
      runner: async () => ({ exitCode: 0 }),
      agentModelRuntime: async () => ({
        getProviders: () => [{ id: "anthropic", name: "Anthropic" }],
        getModels: (provider?: string) =>
          provider ? models.filter((model) => model.provider === provider) : models,
        getModel: (provider: string, model: string) =>
          models.find((candidate) => candidate.provider === provider && candidate.id === model),
        hasConfiguredAuth: () => true,
      }),
      contextProvider: async ({ cwd }) => fakeContext(cwd),
    });

    try {
      stdout.length = 0;
      await session.handleLine("/model anthropic");
      const output = stdout.join("");
      expect(output).toContain("Models from anthropic");
      expect(output).toContain("anthropic/model-01");
      expect(output).toContain("anthropic/model-20");
      expect(output).not.toContain("anthropic/model-21");
      expect(output).toMatch(/… 1 more — \/model <query> to search/);
    } finally {
      delete process.env.TUNED_TENSOR_HOME;
      rmSync(configRoot, { recursive: true, force: true });
    }
  });

  it("saves a provider API key through /login without putting the secret on the command line", async () => {
    const configRoot = mkdtempSync(join(tmpdir(), "tt-shell-login-"));
    process.env.TUNED_TENSOR_HOME = configRoot;
    const stdout: string[] = [];
    const stderr: string[] = [];
    const keys: Array<{ provider: string; apiKey: string }> = [];
    const authenticated = new Set<string>();
    const models = [
      { id: "claude-sonnet-4-5", provider: "anthropic", name: "Claude Sonnet 4.5", reasoning: true },
      { id: "gpt-5.2", provider: "openai", name: "GPT 5.2", reasoning: true },
    ];
    const prompts: string[] = [];
    const session = await createShellSession({
      cwd: "/tmp/local-project",
      env: { HOME: "/tmp/home" },
      io: {
        write: (text) => stdout.push(text),
        writeError: (text) => stderr.push(text),
        clear: vi.fn(),
        promptLine: async (message) => {
          prompts.push(message);
          return "openai";
        },
        promptSecret: async (message) => {
          prompts.push(message);
          return "  sk-test-openai  ";
        },
      },
      runner: async () => ({ exitCode: 0 }),
      agentModelRuntime: async () => ({
        getProviders: () => [
          { id: "anthropic", name: "Anthropic" },
          { id: "openai", name: "OpenAI" },
          { id: "groq", name: "Groq" },
        ],
        getModels: (provider?: string) =>
          provider ? models.filter((model) => model.provider === provider) : models,
        getModel: (provider: string, model: string) =>
          models.find((candidate) => candidate.provider === provider && candidate.id === model),
        hasConfiguredAuth: (provider: string) => authenticated.has(provider),
        setRuntimeApiKey: async (provider, apiKey) => {
          keys.push({ provider, apiKey });
          authenticated.add(provider);
        },
      }),
      contextProvider: async ({ cwd }) => fakeContext(cwd),
    });

    try {
      stdout.length = 0;
      stderr.length = 0;
      prompts.length = 0;
      await session.handleLine("/login");
      expect(stdout.join("")).toContain("Providers");
      expect(stdout.join("")).toContain("openai");
      expect(stdout.join("")).not.toContain("anthropic");
      expect(stdout.join("")).not.toContain("groq");
      expect(stdout.join("")).toContain("Other providers: /login <id>");
      expect(prompts).toEqual(["Provider: ", "OpenAI API key: "]);
      expect(stderr.join("")).toBe("");
      expect(keys).toEqual([{ provider: "openai", apiKey: "sk-test-openai" }]);
      expect(stdout.join("")).toContain("Saved openai credentials");
      expect(JSON.parse(readFileSync(join(configRoot, "agent", "auth.json"), "utf-8")))
        .toEqual({ openai: { type: "api_key", key: "sk-test-openai" } });

      stdout.length = 0;
      stderr.length = 0;
      await session.handleLine("/login missing");
      expect(stderr.join("")).toMatch(/Unknown provider "missing".*\/login <id> or \/model <id>/);

      stdout.length = 0;
      stderr.length = 0;
      prompts.length = 0;
      await session.handleLine("/login anthropic");
      expect(prompts).toEqual(["Anthropic API key: "]);
      expect(stdout.join("")).toContain("Saved anthropic credentials");
      expect(keys).toEqual([
        { provider: "openai", apiKey: "sk-test-openai" },
        { provider: "anthropic", apiKey: "sk-test-openai" },
      ]);
      expect(JSON.parse(readFileSync(join(configRoot, "agent", "auth.json"), "utf-8")))
        .toEqual({
          openai: { type: "api_key", key: "sk-test-openai" },
          anthropic: { type: "api_key", key: "sk-test-openai" },
        });
      expect(stderr.join("")).toBe("");

      stdout.length = 0;
      await session.handleLine("/model openai/gpt-5.2");
      expect(stdout.join("")).toContain("Agent model: openai/gpt-5.2");

      stdout.length = 0;
      stderr.length = 0;
      prompts.length = 0;
      await session.handleLine("/login");
      expect(prompts[0]).toBe("Provider: ");
      expect(prompts).not.toContain("OpenRouter API key: ");
    } finally {
      delete process.env.TUNED_TENSOR_HOME;
      rmSync(configRoot, { recursive: true, force: true });
    }
  });

  it("refuses /login when the session cannot prompt for a key", async () => {
    const stdout: string[] = [];
    const stderr: string[] = [];
    const session = await createShellSession({
      cwd: "/tmp/local-project",
      env: {},
      io: {
        write: (text) => stdout.push(text),
        writeError: (text) => stderr.push(text),
        clear: vi.fn(),
      },
      runner: async () => ({ exitCode: 0 }),
      agentModelRuntime: async () => ({
        getProviders: () => [{ id: "openai", name: "OpenAI" }],
        getModels: () => [],
        getModel: () => undefined,
        hasConfiguredAuth: () => false,
        setRuntimeApiKey: async () => {},
      }),
      contextProvider: async ({ cwd }) => fakeContext(cwd),
    });

    await session.handleLine("/login openai");
    expect(stderr.join("")).toMatch(/interactive tt session/);
  });
});

describe("spec and machine views in the shell", () => {
  function session(cwd: string, systemProbe?: () => Promise<LiveUsage>) {
    const stdout: string[] = [];
    const stderr: string[] = [];
    const created = createShellSession({
      cwd,
      env: { HOME: cwd, TUNED_TENSOR_HOME: join(cwd, ".tt-home") },
      io: { write: (text) => stdout.push(text), writeError: (text) => stderr.push(text), clear: vi.fn() },
      runner: vi.fn(async () => ({ exitCode: 0 })),
      contextProvider: async ({ cwd: directory }) => fakeContext(directory),
      systemProbe,
      columns: () => 100,
    });
    return { created, stdout, stderr };
  }

  it("renders the readable overview, a section, and highlighted JSON", async () => {
    const root = mkdtempSync(join(tmpdir(), "tt-shell-spec-"));
    try {
      writeFileSync(join(root, "tunedtensor.json"), readFileSync(
        join(import.meta.dirname, "../../examples/single-spec/adapter/tunedtensor.json"),
        "utf8",
      ));
      const { created, stdout, stderr } = session(root);
      const shell = await created;
      await shell.handleLine("/spec");
      await shell.handleLine("/spec examples");
      await shell.handleLine("/spec show");
      await shell.handleLine("/spec tunedtensor.json");
      await shell.handleLine("/spec nonsense");
      const [overview, examples, source, byPath] = stdout.map((text) => stripVTControlCharacters(text));
      expect(overview).toContain("▍BEHAVIOR");
      expect(overview).toContain("… 1 more · /spec examples shows all");
      expect(examples).toContain("Support never replied.");
      expect(examples).not.toContain("▍BEHAVIOR");
      expect(source).toContain('"base_model": "Qwen/Qwen3.5-2B"');
      expect(byPath).toContain("▍TRAINING");
      expect(stderr.join("")).toMatch(/Usage: \/spec \[identity\|behavior/);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("explains how to create a missing spec", async () => {
    const root = mkdtempSync(join(tmpdir(), "tt-shell-nospec-"));
    try {
      const { created, stderr } = session(root);
      await (await created).handleLine("/spec");
      expect(stderr.join("")).toMatch(/No tunedtensor\.json in .*Run init, or ask TT to draft/);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("samples live usage for the banner and on /system", async () => {
    const probe = vi.fn(async () => liveUsage);
    const { created, stdout, stderr } = session("/tmp/local-project", probe);
    const shell = await created;
    expect(probe).toHaveBeenCalledTimes(1);
    expect(stripVTControlCharacters(shell.banner())).toContain("MACHINE");
    await shell.handleLine("/system");
    expect(probe).toHaveBeenCalledTimes(2);
    const text = stripVTControlCharacters(stdout.join(""));
    expect(text).toContain("CERTIFIED BASE MODELS");
    expect(text).toMatch(/✓ Qwen\/Qwen3\.5-2B/);
    expect(text).toContain("No full probe yet");
    expect(stderr).toEqual([]);
  });

  it("keeps working when the probe fails or is absent", async () => {
    const failing = session("/tmp/local-project", async () => { throw new Error("boom"); });
    const shell = await failing.created;
    expect(stripVTControlCharacters(shell.banner())).not.toContain("MACHINE");
    await shell.handleLine("/system");
    expect(failing.stderr.join("")).toMatch(/Live machine usage is unavailable/);
  });
});
