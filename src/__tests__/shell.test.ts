import { describe, expect, it, vi } from "vitest";
import { EventEmitter } from "node:events";
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
  it("routes ordinary input to the selected mode", () => {
    expect(routeShellCommand("runs list --json", "local")).toEqual({
      target: "local",
      args: ["runs", "list", "--json"],
    });
  });

  it("supports explicit workflow prefixes and a redundant tt prefix", () => {
    expect(routeShellCommand("cloud runs list", "local")).toEqual({
      target: "cloud",
      args: ["runs", "list"],
    });
    expect(routeShellCommand("tt local doctor", "cloud")).toEqual({
      target: "local",
      args: ["doctor"],
    });
  });

  it("does not open a nested shell for a bare tt command", () => {
    expect(() => routeShellCommand("tt", "cloud")).toThrow(/already open/);
  });
});

describe("isCatalogCommand", () => {
  it("recognizes existing CLI grammar without treating conversation as commands", () => {
    expect(isCatalogCommand("cloud", ["runs", "list"])).toBe(true);
    expect(isCatalogCommand("local", ["doctor"])).toBe(true);
    expect(isCatalogCommand("cloud", ["show", "my", "latest", "run"])).toBe(false);
    expect(isCatalogCommand("cloud", ["runs", "please"])).toBe(false);
    expect(isCatalogCommand("cloud", ["status", "of", "my", "run"])).toBe(false);
    expect(isCatalogCommand("cloud", ["status", "--target", "local"])).toBe(true);
  });
});

describe("parseSlashCommand", () => {
  it("parses the palette, slash commands, and question-mark help alias", () => {
    expect(parseSlashCommand("/")).toEqual({ name: "palette", args: [] });
    expect(parseSlashCommand("/mode local")).toEqual({
      name: "mode",
      args: ["local"],
    });
    expect(parseSlashCommand("? runs")).toEqual({
      name: "help",
      args: ["runs"],
    });
  });

  it("rejects unknown slash commands and operators", () => {
    expect(() => parseSlashCommand("/wat")).toThrow(/Unknown session command/);
    expect(() => parseSlashCommand("/help | cat")).toThrow(/operator/);
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
  it("completes mode-aware commands, slash commands, and overrides", () => {
    let mode: "cloud" | "local" = "local";
    const complete = createCommandCompleter(() => mode);

    expect(complete("runs c")[0]).toContain("runs compare");
    expect(complete("cloud runs e")[0]).toContain("cloud runs estimate");
    expect(complete("/mo")[0]).toEqual([
      "/mode cloud",
      "/mode local",
      "/model",
    ]);
    expect(complete("cl")[0]).toContain("cloud ");

    mode = "cloud";
    expect(complete("auth s")[0]).toContain("auth status");
    expect(complete("doctor")[0]).not.toContain("doctor");
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

function fakeContext(cwd: string, target: "cloud" | "local"): ShellContext {
  return {
    cwd,
    projectName: cwd.split("/").filter(Boolean).at(-1) ?? cwd,
    inferredTarget: target,
    targetSource: target === "local" ? "adjacent-config" : "default-cloud",
    cloud: {
      authenticated: true,
      keyPrefix: "tt_test…",
      baseUrl: "https://tunedtensor.com",
      configPath: "/config.json",
      configFound: true,
    },
    local: {
      configPath: target === "local" ? `${cwd}/local-runner.json` : undefined,
      artifactRoot: `${cwd}/.tt-local/artifacts`,
      storeRoot: `${cwd}/.tt-local/store`,
      activeModelId: target === "local" ? "model_abc123" : undefined,
    },
    warnings: [],
  };
}

describe("renderShellBanner", () => {
  it("shows a compact heading, context, controls, and version", () => {
    const banner = renderShellBanner({
      mode: "cloud",
      modeSource: "default-cloud",
      cwd: "/tmp/cloud-project",
      context: fakeContext("/tmp/cloud-project", "cloud"),
      version: "0.6.0",
    });
    const rows = banner.trimEnd().split("\n");
    expect(rows).toHaveLength(5);
    expect(rows[0]).toContain("tt");
    expect(banner).toContain("v0.6.0");
    expect(banner).toContain("cloud");
    expect(banner).toContain("ctrl+c stop/clear");
    expect(banner).toContain("Ask TT anything");
    expect(banner).not.toContain("██");
  });

  it("omits the version when none is provided", () => {
    const banner = renderShellBanner({
      mode: "local",
      modeSource: "adjacent-config",
      cwd: "/tmp/local-project",
      context: fakeContext("/tmp/local-project", "local"),
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

describe("TunedTensorShellSession", () => {  it("routes commands, switches modes, and recovers from parse errors", async () => {
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
      contextProvider: async ({ cwd }) => fakeContext(cwd, "local"),
    });

    expect(session.snapshot().mode).toBe("local");
    await session.handleLine("runs list");
    await session.handleLine("cloud auth status");
    await session.handleLine("/mode cloud");
    await session.handleLine("runs list");
    await session.handleLine("runs list | cat");

    expect(requests).toEqual([
      { target: "local", args: ["runs", "list"], cwd: "/tmp/local-project" },
      { target: "cloud", args: ["auth", "status"], cwd: "/tmp/local-project" },
      { target: "cloud", args: ["runs", "list"], cwd: "/tmp/local-project" },
    ]);
    expect(stdout.join("")).toContain("Workflow switched to cloud");
    expect(stderr.join("")).toMatch(/Shell operator/);
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
      contextProvider: async ({ cwd }) => fakeContext(cwd, "cloud"),
    });

    await session.handleLine("What happened in my latest training run?");
    await session.handleLine("/new");
    await session.handleLine("/approve action-123");
    await session.handleLine("Compare accuracy > latency & cost; summarize it.");
    await session.handleLine("status of my latest run");
    await session.handleLine("runs list");
    await session.handleLine("runs list | cat");
    await session.handleLine(": balance");
    await session.handleLine("cloud not-a-command");

    expect(agent.handleLine.mock.calls.map((call) => call[0])).toEqual([
      "What happened in my latest training run?",
      "/new",
      "/approve action-123",
      "Compare accuracy > latency & cost; summarize it.",
      "status of my latest run",
    ]);
    expect(agent.handleLine.mock.calls.map((call) => call[1])).toEqual([
      { mode: "cloud", workspaceRoot: "/tmp/cloud-project" },
      { mode: "cloud", workspaceRoot: "/tmp/cloud-project" },
      { mode: "cloud", workspaceRoot: "/tmp/cloud-project" },
      { mode: "cloud", workspaceRoot: "/tmp/cloud-project" },
      { mode: "cloud", workspaceRoot: "/tmp/cloud-project" },
    ]);
    expect(run.mock.calls.map((call) => call[0])).toEqual([
      {
        target: "cloud",
        args: ["runs", "list"],
        cwd: "/tmp/cloud-project",
      },
      {
        target: "cloud",
        args: ["balance"],
        cwd: "/tmp/cloud-project",
      },
      {
        target: "cloud",
        args: ["not-a-command"],
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
      contextProvider: async ({ cwd }) => fakeContext(cwd, "cloud"),
    });

    await session.handleLine("/help runs");
    await session.handleLine("/context");
    await session.handleLine("/status");
    await session.handleLine("/clear");
    expect(await session.handleLine("exit")).toBe("exit");
    expect(await session.handleLine("/exit")).toBe("exit");

    const output = stdout.join("");
    expect(output).toContain("TT cloud commands matching");
    expect(output).toContain("Cloud endpoint");
    expect(output).toContain("Remote status");
    expect(clear).toHaveBeenCalledTimes(1);
    expect(run).not.toHaveBeenCalled();
  });

  it("shows and activates models through /model", async () => {
    const requests: ShellCommandRequest[] = [];
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
      runner: async (request) => {
        requests.push(request);
        return { exitCode: 0 };
      },
      contextProvider: async ({ cwd }) => fakeContext(cwd, "local"),
    });

    await session.handleLine("/model");
    expect(stdout.join("")).toContain("Active model");
    expect(stdout.join("")).toContain("model_abc123");

    await session.handleLine("/model model_def456");
    expect(requests).toEqual([
      {
        target: "local",
        args: ["models", "activate", "model_def456"],
        cwd: "/tmp/local-project",
      },
    ]);

    await session.handleLine("/mode cloud");
    await session.handleLine("/model");
    expect(stdout.join("")).toContain("Base model");
    await session.handleLine("/model model_def456");
    expect(stderr.join("")).toMatch(/local workflow action/);
    expect(requests).toHaveLength(1);
  });
});
