import { describe, expect, it, vi } from "vitest";
import { createProgram } from "../cli.js";

describe("unified local and cloud CLI surface", () => {
  it("keeps local commands at the root and cloud operations in an explicit namespace", () => {
    const program = createProgram("test");
    const names = program.commands.map((command) => command.name());
    expect(names).toEqual(expect.arrayContaining([
      "info", "init", "doctor", "hardware", "validate", "serve", "runs", "models",
      "agent", "pipeline", "status", "shell", "cloud", "auth", "balance", "topup", "usage", "publish",
    ]));
    expect(names).not.toEqual(expect.arrayContaining(["specs", "datasets", "push", "label"]));
    const cloud = program.commands.find((command) => command.name() === "cloud")!;
    expect(cloud.commands.map((command) => command.name())).toEqual(expect.arrayContaining([
      "push", "specs", "datasets", "label", "runs", "models",
    ]));
    const help = program.helpInformation();
    expect(help).toMatch(/cloud/);
    expect(help).toMatch(/TT access token/);
    expect(help).not.toMatch(/^\s+run\b/m);
    expect(help.match(/Commands:/g)?.length).toBe(1);
  });

  it("does not expose hosted agent tools on the default client", async () => {
    const { createTunedTensorTools } = await import("../agent-tools.js");
    const names = createTunedTensorTools({
      get: async () => {
        throw new Error("No TT access token is configured.");
      },
      postRead: async () => {
        throw new Error("No TT access token is configured.");
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
