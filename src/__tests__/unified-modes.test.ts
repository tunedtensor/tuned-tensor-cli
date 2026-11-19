import { describe, expect, it, vi } from "vitest";
import { createProgram } from "../cli.js";

describe("local-only CLI surface", () => {
  it("keeps workflow commands at the root and has no hosted-account commands", () => {
    const program = createProgram("test");
    const names = program.commands.map((command) => command.name());
    expect(names).toEqual(expect.arrayContaining([
      "info", "init", "doctor", "hardware", "validate", "serve", "runs", "models",
      "agent", "pipeline", "status", "shell", "upgrade",
    ]));
    for (const retired of ["cloud", "auth", "balance", "topup", "usage", "publish", "specs", "push", "label"]) {
      expect(names).not.toContain(retired);
    }
    const help = program.helpInformation();
    expect(help).not.toMatch(/--api-key|--base-url|TT access token|tt cloud|tt auth/);
    expect(help).not.toMatch(/^\s+run\b/m);
    expect(help.match(/Commands:/g)?.length).toBe(1);
  });

  it("rejects the retired hosted-service options", async () => {
    const program = createProgram("test");
    program.exitOverride();
    program.configureOutput({ writeErr: () => {}, writeOut: () => {} });
    await expect(program.parseAsync(["node", "tt", "--api-key", "tt_x", "status"])).rejects.toThrow(/unknown option/i);
  });

  it("exposes no hosted agent tools", async () => {
    const { createTunedTensorTools } = await import("../agent-tools.js");
    const names = createTunedTensorTools({
      propose: async (action) => action,
    }, { workspaceRoot: process.cwd() }).map((tool) => tool.name);
    for (const hosted of ["list_specs", "list_runs", "get_balance", "list_transactions", "get_agent_usage", "prepare_create_spec", "prepare_label_job"]) {
      expect(names).not.toContain(hosted);
    }
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
