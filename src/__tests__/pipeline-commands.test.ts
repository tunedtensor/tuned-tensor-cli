import { afterEach, describe, expect, it, vi } from "vitest";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { createProgram } from "../cli.js";
import { setJsonMode } from "../output.js";

afterEach(() => {
  setJsonMode(false);
  vi.restoreAllMocks();
});

describe("pipeline commands", () => {
  it("initializes a canonical recipe and emits a resolved dry-run JSON plan without execution", async () => {
    const dir = mkdtempSync(join(tmpdir(), "tt-pipeline-"));
    const file = join(dir, "pipeline.json");
    const log = vi.spyOn(console, "log").mockImplementation(() => undefined);
    try {
      const program = createProgram("test");
      program.exitOverride();
      await program.parseAsync(["node", "tt", "pipeline", "init", "--target", "cloud", "--file", file]);
      expect(JSON.parse(readFileSync(file, "utf8"))).toMatchObject({ version: 1, target: "cloud" });

      await program.parseAsync(["node", "tt", "--json", "pipeline", "run", "--dry-run", "--file", file]);
      const output = JSON.parse(log.mock.calls.at(-1)?.[0] as string);
      expect(output.dry_run).toBe(true);
      expect(output.steps[0]).toMatchObject({ id: "baseline", target: "cloud" });
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("rejects run requests without --dry-run in this side-effect-free slice", async () => {
    const program = createProgram("test");
    program.exitOverride();
    await expect(program.parseAsync(["node", "tt", "pipeline", "run"])).rejects.toBeDefined();
  });
});
