import { describe, expect, it, vi } from "vitest";
import { createProgram } from "../cli.js";

describe("local-only CLI surface", () => {
  it("does not register hosted commands", () => {
    const program = createProgram("test");
    const names = program.commands.map((command) => command.name());
    expect(names).not.toEqual(
      expect.arrayContaining([
        "push",
        "balance",
        "topup",
        "cloud",
        "eval",
        "specs",
        "datasets",
        "label",
      ]),
    );
    expect(names).toEqual(
      expect.arrayContaining([
        "info",
        "init",
        "doctor",
        "hardware",
        "validate",
        "run",
        "serve",
        "runs",
        "models",
        "agent",
        "pipeline",
        "status",
        "shell",
      ]),
    );
    expect(names).not.toEqual(
      expect.arrayContaining(["auth", "publish"]),
    );
  });

  it("omits cloud wording from root help", () => {
    const help = createProgram("test").helpInformation();
    expect(help).not.toMatch(/\bcloud\b/i);
    expect(help).not.toMatch(/--api-key/);
    expect(help).not.toMatch(/--base-url/);
    expect(help).not.toMatch(/\btt auth\b/);
    expect(help).not.toMatch(/\btt publish\b/);
    expect(help).not.toMatch(/^\s+run\b/m);
    expect(help).toMatch(/^\s+pipeline\b/m);
    expect(help).toMatch(/^\s+serve\b/m);
    expect(help.match(/Commands:/g)?.length).toBe(1);
  });

  it("does not expose hosted agent tools on the default client", async () => {
    const { createTunedTensorTools } = await import("../agent-tools.js");
    const names = createTunedTensorTools({
      get: async () => {
        throw new Error("This build of tt is local-only.");
      },
      postRead: async () => {
        throw new Error("This build of tt is local-only.");
      },
      propose: async (action) => action,
    }, { localOnly: true, workspaceRoot: process.cwd() }).map((tool) => tool.name);
    expect(names).not.toEqual(expect.arrayContaining([
      "list_specs",
      "list_runs",
      "get_balance",
      "list_transactions",
      "prepare_create_spec",
    ]));
    expect(names).toEqual(expect.arrayContaining([
      "examine_hardware",
      "describe_pipeline",
      "validate_pipeline",
      "prepare_create_local_spec",
      "prepare_pipeline_run",
    ]));
  });

  it("routes tt runs list to the local runtime", async () => {
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
    await program.parseAsync(["node", "tt", "runs", "list"]);
    expect(runLocalCommand).toHaveBeenCalledTimes(1);
    expect(runLocalCommand.mock.calls[0]?.[0]).toEqual(["runs", "list"]);
  });
});
