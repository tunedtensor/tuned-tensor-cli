import { afterEach, describe, expect, it, vi } from "vitest";
import { PassThrough } from "node:stream";
import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import {
  createProgram,
  extractPassthroughOptions,
  runCli,
  type SelfCommandRunner,
} from "../cli.js";
import type { ShellCommandRunner } from "../shell.js";
import { setJsonMode } from "../output.js";

const originalExitCode = process.exitCode;

afterEach(() => {
  setJsonMode(false);
  process.exitCode = originalExitCode;
  vi.restoreAllMocks();
});

describe("extractPassthroughOptions", () => {
  it("accepts presentation flags on either side of a namespace", () => {
    expect(
      extractPassthroughOptions(
        ["runs", "list", "--json", "--no-color"],
        {},
      ),
    ).toEqual({
      args: ["runs", "list"],
      json: true,
      color: false,
    });

    expect(
      extractPassthroughOptions(["doctor"], { json: true, color: false }),
    ).toEqual({
      args: ["doctor"],
      json: true,
      color: false,
    });
  });
});

describe("unified command routing", () => {
  it("configures the local agent and exposes agent model status without secret flags", async () => {
    const configRoot = mkdtempSync(join(tmpdir(), "tt-agent-cli-"));
    process.env.XDG_CONFIG_HOME = configRoot;
    const stdout = new PassThrough();
    let output = "";
    stdout.setEncoding("utf8");
    stdout.on("data", (chunk: string) => { output += chunk; });
    const modelRuntime = {
      getProviders: () => [{ id: "openai", name: "OpenAI" }],
      getModels: () => [{ id: "gpt-5.2", provider: "openai", name: "GPT 5.2", reasoning: true }],
      getModel: () => ({ id: "gpt-5.2", provider: "openai", name: "GPT 5.2", reasoning: true }),
      hasConfiguredAuth: () => true,
    };
    try {
      const program = createProgram("test", { stdout, modelRuntime });
      program.exitOverride();
      await program.parseAsync([
        "node", "tt", "agent", "configure", "--provider", "openai",
        "--model", "gpt-5.2", "--thinking", "high",
      ]);
      expect(output).toContain("openai/gpt-5.2");
      expect(program.commands.map((command) => command.name())).toContain("agent");
      const agent = program.commands.find((command) => command.name() === "agent")!;
      expect(agent.commands.find((command) => command.name() === "configure")?.options.map((option) => option.long))
        .not.toEqual(expect.arrayContaining(["--api-key", "--token", "--secret"]));
    } finally {
      delete process.env.XDG_CONFIG_HOME;
      rmSync(configRoot, { recursive: true, force: true });
    }
  });

  it("reports a selected model whose provider is not authenticated", async () => {
    const configRoot = mkdtempSync(join(tmpdir(), "tt-agent-status-"));
    process.env.XDG_CONFIG_HOME = configRoot;
    const stdout = new PassThrough();
    let output = "";
    stdout.setEncoding("utf8");
    stdout.on("data", (chunk: string) => { output += chunk; });
    try {
      const program = createProgram("test", {
        stdout,
        env: {
          TUNED_TENSOR_AGENT_PROVIDER: "openai",
          TUNED_TENSOR_AGENT_MODEL: "gpt-5.2",
          TUNED_TENSOR_AGENT_THINKING: "off",
        },
        modelRuntime: {
          getProviders: () => [{ id: "openai" }],
          getModels: () => [{ id: "gpt-5.2", provider: "openai", reasoning: true }],
          getModel: () => ({ id: "gpt-5.2", provider: "openai", reasoning: true }),
          hasConfiguredAuth: () => false,
        },
      });
      program.exitOverride();
      await program.parseAsync(["node", "tt", "--json", "agent", "status"]);
      expect(JSON.parse(output)).toMatchObject({
        execution: "local",
        provider: "openai",
        model: "gpt-5.2",
        authenticated: false,
      });
    } finally {
      delete process.env.XDG_CONFIG_HOME;
      rmSync(configRoot, { recursive: true, force: true });
    }
  });

  it("forwards the complete local grammar without Commander consuming it", async () => {
    const runLocalCommand = vi.fn(async (
      _args: string[],
      _options?: unknown,
    ) => ({
      exitCode: 0,
      signal: null,
      stdout: "",
      parsed: undefined,
      droppedKeys: [],
    }));
    const program = createProgram("test", {
      runLocalCommand: runLocalCommand as never,
    });
    program.exitOverride();

    await program.parseAsync([
      "node",
      "tt",
      "local",
      "models",
      "serve",
      "active",
      "--print-command",
      "--json",
    ]);

    expect(runLocalCommand).toHaveBeenCalledTimes(1);
    expect(runLocalCommand.mock.calls[0]?.[0]).toEqual([
      "models",
      "serve",
      "active",
      "--print-command",
    ]);
    expect(runLocalCommand.mock.calls[0]?.[1]).toMatchObject({
      jsonMode: true,
    });
  });

  it("promotes local commands to the root and hides hosted commands", async () => {
    const runLocalCommand = vi.fn(async (
      _args: string[],
      _options?: unknown,
    ) => ({
      exitCode: 0,
      signal: null,
      stdout: "",
      parsed: undefined,
      droppedKeys: [],
    }));
    const program = createProgram("test", {
      runLocalCommand: runLocalCommand as never,
    });
    program.exitOverride();

    const names = program.commands.map((command) => command.name());
    expect(names).toEqual(
      expect.arrayContaining(["runs", "models", "doctor", "init", "shell", "status"]),
    );
    expect(names).not.toEqual(
      expect.arrayContaining(["auth", "push", "balance", "topup", "cloud", "eval", "specs", "datasets", "label"]),
    );

    await program.parseAsync(["node", "tt", "runs", "list", "--json"]);
    expect(runLocalCommand.mock.calls[0]?.[0]).toEqual(["runs", "list"]);
    expect(runLocalCommand.mock.calls[0]?.[1]).toMatchObject({ jsonMode: true });
  });

  it("isolates shell commands in child tt invocations", async () => {
    const runSelfCommand = vi.fn(async (
      _args: string[],
      _options?: unknown,
    ) => ({
      exitCode: 0,
      signal: null,
    }));
    const startShell = vi.fn(async (options: {
      runner: ShellCommandRunner;
    }) => {
      await options.runner({
        args: ["doctor"],
        cwd: "/tmp/tt-project",
      });
      await options.runner({
        args: ["runs", "list"],
        cwd: "/tmp/tt-project",
      });
    });
    const program = createProgram("test", {
      runSelfCommand: runSelfCommand as SelfCommandRunner,
      startShell: startShell as never,
    });
    program.exitOverride();

    await program.parseAsync(["node", "tt", "shell"]);

    expect(runSelfCommand.mock.calls.map((call) => call[0])).toEqual([
      ["doctor"],
      ["runs", "list"],
    ]);
  });

  it("reports local project context from status", async () => {
    const stdout = new PassThrough();
    let output = "";
    stdout.setEncoding("utf8");
    stdout.on("data", (chunk: string) => {
      output += chunk;
    });
    const program = createProgram("test", {
      cwd: "/tmp",
      env: { HOME: "/tmp/tt-cli-status-home" },
      stdout,
    });
    program.exitOverride();

    await program.parseAsync([
      "node",
      "tt",
      "--json",
      "status",
    ]);

    expect(JSON.parse(output)).toMatchObject({
      context: {
        local: {
          artifactRoot: expect.any(String),
          storeRoot: expect.any(String),
        },
      },
    });
    expect(JSON.parse(output).context.cloud).toBeUndefined();
  });

  it("checks for updates before starting the bare interactive shell", async () => {
    const startShell = vi.fn(async () => {});
    const checkForUpdate = vi.fn(async () => ({
      currentVersion: "0.10.0",
      latestVersion: "0.11.0",
    }));
    const stderr = new PassThrough();
    let errors = "";
    stderr.setEncoding("utf8");
    stderr.on("data", (chunk: string) => { errors += chunk; });

    await runCli("0.10.0", {
      argv: ["node", "tt"],
      env: { TERM: "xterm-256color" },
      stdinIsTTY: true,
      stdoutIsTTY: true,
      stderr,
      checkForUpdate,
      startShell: startShell as never,
    });

    expect(checkForUpdate).toHaveBeenCalledWith("0.10.0");
    expect(errors).toContain("tt 0.10.0");
    expect(errors).toContain("npm install -g @tuned-tensor/cli@latest");
    expect(startShell).toHaveBeenCalledTimes(1);
    expect(checkForUpdate.mock.invocationCallOrder[0]).toBeLessThan(
      startShell.mock.invocationCallOrder[0]!,
    );
  });

  it("does not check for updates for explicit commands", async () => {
    const checkForUpdate = vi.fn(async () => null);
    const stdout = new PassThrough();
    let output = "";
    stdout.setEncoding("utf8");
    stdout.on("data", (chunk: string) => { output += chunk; });

    await runCli("0.10.0", {
      argv: ["node", "tt", "status", "--json"],
      cwd: "/tmp",
      env: { TERM: "xterm-256color", HOME: "/tmp/tt-cli-update-home" },
      stdinIsTTY: true,
      stdoutIsTTY: true,
      stdout,
      checkForUpdate,
    });

    expect(output).toContain('"storeRoot"');
    expect(checkForUpdate).not.toHaveBeenCalled();
  });

  it("starts the shell when update discovery throws", async () => {
    const startShell = vi.fn(async () => {});

    await runCli("0.10.0", {
      argv: ["node", "tt"],
      env: { TERM: "xterm-256color" },
      stdinIsTTY: true,
      stdoutIsTTY: true,
      checkForUpdate: async () => {
        throw new Error("registry unavailable");
      },
      startShell: startShell as never,
    });

    expect(startShell).toHaveBeenCalledTimes(1);
  });

  it("starts the shell only for a bare interactive invocation", async () => {
    const startShell = vi.fn(async () => {});

    await runCli("test", {
      argv: ["node", "tt"],
      env: { TERM: "xterm-256color" },
      stdinIsTTY: true,
      stdoutIsTTY: true,
      startShell: startShell as never,
    });

    expect(startShell).toHaveBeenCalledTimes(1);
  });

  it("uses the local agent client and never calls remote /agent endpoints", async () => {
    const configRoot = mkdtempSync(join(tmpdir(), "tt-agent-local-"));
    process.env.XDG_CONFIG_HOME = configRoot;
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const modelRuntime = {
      getProviders: () => [{ id: "openai" }],
      getModels: () => [{ id: "gpt-5.2", provider: "openai", reasoning: true }],
      getModel: () => ({ id: "gpt-5.2", provider: "openai", reasoning: true }),
      hasConfiguredAuth: () => true,
    };
    const startShell = vi.fn(async (options: { agent?: { handleLine(input: string): Promise<unknown> } }) => {
      await options.agent?.handleLine("hello locally");
    });
    try {
      await runCli("test", {
        argv: ["node", "tt"],
        env: {
          TERM: "xterm-256color",
          TUNED_TENSOR_API_KEY: "tt_must_not_reach_pi",
          TUNED_TENSOR_AGENT_PROVIDER: "openai",
          TUNED_TENSOR_AGENT_MODEL: "gpt-5.2",
          TUNED_TENSOR_AGENT_THINKING: "medium",
        },
        stdinIsTTY: true,
        stdoutIsTTY: true,
        stdout: new PassThrough(),
        stderr: new PassThrough(),
        modelRuntime,
        createPiAgent: (options) => ({
          ...(() => {
            expect(JSON.stringify(options)).not.toContain("tt_must_not_reach_pi");
            return {};
          })(),
          state: { messages: options.messages },
          subscribe: () => () => {},
          prompt: async () => {},
          abort: () => {},
        }),
        startShell: startShell as never,
      });
      expect(fetchSpy.mock.calls.some(([url]) => String(url).includes("/agent/"))).toBe(false);
    } finally {
      delete process.env.XDG_CONFIG_HOME;
      rmSync(configRoot, { recursive: true, force: true });
    }
  });

  it("retries agent client creation after configuring mid-session", async () => {
    const configRoot = mkdtempSync(join(tmpdir(), "tt-agent-reconfigure-"));
    process.env.XDG_CONFIG_HOME = configRoot;
    const { updateConfig } = await import("../config.js");
    const stderr = new PassThrough();
    let errors = "";
    stderr.setEncoding("utf8");
    stderr.on("data", (chunk: string) => {
      errors += chunk;
    });
    const modelRuntime = {
      getProviders: () => [{ id: "openai" }],
      getModels: () => [{ id: "gpt-5.2", provider: "openai", reasoning: true }],
      getModel: () => ({ id: "gpt-5.2", provider: "openai", reasoning: true }),
      hasConfiguredAuth: () => true,
    };
    const createPiAgent = vi.fn((options: { messages: unknown }) => ({
      state: { messages: options.messages },
      subscribe: () => () => {},
      prompt: async () => {},
      abort: () => {},
    }));
    const startShell = vi.fn(async (options: { agent?: { handleLine(input: string): Promise<unknown> } }) => {
      // Fails before the agent is configured.
      await options.agent?.handleLine("hello");
      expect(errors).toContain("agent is not configured");
      // Configure mid-session, as `tt agent configure` does.
      updateConfig({
        agent: { provider: "openai", model: "gpt-5.2", thinking: "medium" },
      });
      // Must retry client creation instead of replaying the stale rejection.
      await options.agent?.handleLine("hello again");
    });
    try {
      await runCli("test", {
        argv: ["node", "tt"],
        env: { TERM: "xterm-256color" },
        stdinIsTTY: true,
        stdoutIsTTY: true,
        stdout: new PassThrough(),
        stderr,
        modelRuntime,
        createPiAgent: createPiAgent as never,
        startShell: startShell as never,
      });
      expect(createPiAgent).toHaveBeenCalledTimes(1);
    } finally {
      delete process.env.XDG_CONFIG_HOME;
      rmSync(configRoot, { recursive: true, force: true });
    }
  });
});
