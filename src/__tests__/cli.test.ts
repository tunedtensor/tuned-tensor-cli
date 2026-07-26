import { afterEach, describe, expect, it, vi } from "vitest";
import { PassThrough } from "node:stream";
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

  it("keeps hosted commands at the root and offers an explicit cloud alias", async () => {
    const runSelfCommand = vi.fn(async (
      _args: string[],
      _options?: unknown,
    ) => ({
      exitCode: 0,
      signal: null,
    }));
    const program = createProgram("test", {
      runSelfCommand: runSelfCommand as SelfCommandRunner,
    });
    program.exitOverride();

    expect(program.commands.map((command) => command.name())).toEqual(
      expect.arrayContaining(["runs", "models", "local", "cloud", "shell"]),
    );

    await program.parseAsync([
      "node",
      "tt",
      "cloud",
      "runs",
      "list",
      "--json",
    ]);

    expect(runSelfCommand).toHaveBeenCalledWith(
      ["--json", "runs", "list"],
      expect.objectContaining({ cwd: expect.any(String) }),
    );
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
        target: "local",
        args: ["doctor"],
        cwd: "/tmp/tt-project",
      });
      await options.runner({
        target: "cloud",
        args: ["auth", "status"],
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
      ["local", "doctor"],
      ["auth", "status"],
    ]);
  });

  it("applies root cloud overrides to explicit shell context", async () => {
    const startShell = vi.fn(async () => {});
    const program = createProgram("test", {
      env: { HOME: "/tmp/tt-cli-shell-home" },
      startShell: startShell as never,
    });
    program.exitOverride();

    await program.parseAsync([
      "node",
      "tt",
      "--api-key",
      "tt_test_override",
      "--base-url",
      "https://example.test",
      "shell",
    ]);

    expect(startShell).toHaveBeenCalledWith(expect.objectContaining({
      env: expect.objectContaining({
        TUNED_TENSOR_API_KEY: "tt_test_override",
        TUNED_TENSOR_URL: "https://example.test",
      }),
    }));
  });

  it("applies root cloud overrides to status discovery", async () => {
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
      "--api-key",
      "tt_test_override",
      "--base-url",
      "https://example.test",
      "--json",
      "status",
    ]);

    expect(JSON.parse(output)).toMatchObject({
      target: "cloud",
      context: {
        cloud: {
          authenticated: true,
          keyPrefix: "tt_test_…",
          baseUrl: "https://example.test",
        },
      },
    });
  });

  it("labels an explicit status target as a command option", async () => {
    const stdout = new PassThrough();
    let output = "";
    stdout.setEncoding("utf8");
    stdout.on("data", (chunk: string) => {
      output += chunk;
    });
    const program = createProgram("test", {
      cwd: "/tmp",
      env: { HOME: "/tmp/tt-cli-status-target-home" },
      stdout,
    });
    program.exitOverride();

    await program.parseAsync([
      "node",
      "tt",
      "status",
      "--target",
      "local",
    ]);

    expect(output).toContain("Target         local (--target)");
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
});
