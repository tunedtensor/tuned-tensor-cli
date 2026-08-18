import { describe, expect, it, vi } from "vitest";
import { EventEmitter } from "node:events";
import { mkdtempSync, rmSync } from "node:fs";
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

  it("parses /model and suggests fixes for mistyped session commands", () => {
    expect(parseSlashCommand("/model")).toEqual({ name: "model", args: [] });
    expect(parseSlashCommand("/model abc123")).toEqual({
      name: "model",
      args: ["abc123"],
    });
    expect(() => parseSlashCommand("/stat")).toThrow(/Did you mean \/status\?/);
    expect(() => parseSlashCommand("/models")).toThrow(/need no slash/);
  });
});

describe("command completion", () => {
  it("completes local commands and slash commands", () => {
    const complete = createCommandCompleter();

    expect(complete("runs c")[0]).toContain("runs compare");
    expect(complete("/mo")[0]).toEqual(["/model"]);
    expect(complete("cl")[0].join(" ")).not.toMatch(/\bcloud\b/);
    expect(complete("auth s")[0]).toContain("auth status");
    expect(complete("publish")[0]).toContain("publish");
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
      artifactRoot: `${cwd}/.tt-local/artifacts`,
      storeRoot: `${cwd}/.tt-local/store`,
      activeModelId: "model_abc123",
    },
    warnings: [],
  };
}

describe("renderShellBanner", () => {
  it("shows a compact heading, context, controls, and version", () => {
    const banner = renderShellBanner({
      mode: "local",
      modeSource: "default-local",
      cwd: "/tmp/local-project",
      context: fakeContext("/tmp/local-project"),
      version: "0.6.0",
    });
    const rows = banner.trimEnd().split("\n");
    expect(rows).toHaveLength(5);
    expect(rows[0]).toContain("tt");
    expect(banner).toContain("v0.6.0");
    expect(banner).not.toMatch(/no spec/);
    expect(banner).toContain("agent");
    expect(banner).toContain("workflow model");
    expect(banner).toContain("ctrl+c stop/clear");
    expect(banner).toContain("Ask TT anything");
    expect(banner).not.toContain("██");
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
    expect(output).toContain("Host checks");
    expect(clear).toHaveBeenCalledTimes(1);
    expect(run).not.toHaveBeenCalled();
  });

  it("lists, searches, and changes the TT agent model through /model", async () => {
    const configRoot = mkdtempSync(join(tmpdir(), "tt-shell-model-"));
    process.env.XDG_CONFIG_HOME = configRoot;
    const stdout: string[] = [];
    const stderr: string[] = [];
    const modelRuntime = {
      getProviders: () => [{ id: "anthropic", name: "Anthropic" }],
      getModels: () => [
        { id: "claude-sonnet-4-5", provider: "anthropic", name: "Claude Sonnet 4.5", reasoning: true },
        { id: "claude-haiku-4-5", provider: "anthropic", name: "Claude Haiku 4.5", reasoning: false },
      ],
      getModel: (provider: string, model: string) => ({
        id: model,
        provider,
        name: model === "claude-sonnet-4-5" ? "Claude Sonnet 4.5" : "Claude Haiku 4.5",
        reasoning: model !== "claude-haiku-4-5",
      }),
      hasConfiguredAuth: () => true,
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
      expect(output).toContain("Available models");
      expect(output).toContain("anthropic/claude-sonnet-4-5");

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
      delete process.env.XDG_CONFIG_HOME;
      rmSync(configRoot, { recursive: true, force: true });
    }
  });
});
