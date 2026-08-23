import { afterEach, describe, expect, it, vi } from "vitest";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
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
      await program.parseAsync(["node", "tt", "pipeline", "init", "--file", file]);
      expect(JSON.parse(readFileSync(file, "utf8"))).toMatchObject({ version: 1, name: "default-local" });

      await program.parseAsync(["node", "tt", "--json", "pipeline", "run", "--dry-run", "--file", file]);
      const output = JSON.parse(log.mock.calls.at(-1)?.[0] as string);
      expect(output.dry_run).toBe(true);
      expect(output.steps[0]).toMatchObject({ id: "baseline", target: "local" });
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("initializes a foundation recipe and dry-runs a spec-generated DAG", async () => {
    const dir = mkdtempSync(join(tmpdir(), "tt-pipeline-"));
    const file = join(dir, "pipeline.json");
    const spec = join(dir, "tunedtensor.json");
    const log = vi.spyOn(console, "log").mockImplementation(() => undefined);
    try {
      const program = createProgram("test");
      program.exitOverride();
      await program.parseAsync(["node", "tt", "pipeline", "init", "--engine", "foundation", "--file", file]);
      expect(JSON.parse(readFileSync(file, "utf8"))).toMatchObject({
        version: 1,
        runtime: { engine: "foundation" },
      });

      writeFileSync(spec, JSON.stringify({
        engine: "foundation",
        name: "Tiny GPT",
        system_prompt: "You are a helpful assistant.",
        guidelines: ["Answer directly."],
        constraints: [],
        examples: [
          { input: "Hello", output: "Hi there." },
          { input: "Thanks", output: "You're welcome." },
        ],
        foundation: {
          depth: 2,
          pretrain_steps: 2,
          finetune_steps: 2,
          rl_steps: 0,
          vocab_size: 256,
          max_chars: 20_000,
          sequence_length: 64,
          batch_size: 2,
          nproc_per_node: 1,
        },
      }));

      await program.parseAsync([
        "node", "tt", "--json", "pipeline", "run", "--dry-run",
        "--file", join(dir, "generated.pipeline.json"),
        "--spec", spec,
      ]);
      const output = JSON.parse(log.mock.calls.at(-1)?.[0] as string);
      expect(output.dry_run).toBe(true);
      expect(output.steps[0]).toMatchObject({ id: "tokenize", uses: "tokenize", target: "local" });
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("rejects a leftover adapter pipeline file when --spec is foundation", async () => {
    const dir = mkdtempSync(join(tmpdir(), "tt-pipeline-"));
    const spec = join(dir, "tunedtensor.json");
    const pipeline = join(dir, "tunedtensor.pipeline.json");
    writeFileSync(
      spec,
      JSON.stringify({
        engine: "foundation",
        name: "foundation-spec",
        system_prompt: "Answer questions.",
        guidelines: [],
        constraints: [],
        examples: [
          { input: "hello", output: "world" },
          { input: "thanks", output: "you are welcome" },
        ],
        foundation: {
          vocab_size: 256,
          max_chars: 20000,
          depth: 2,
          pretrain_steps: 2,
          finetune_steps: 2,
          rl_steps: 0,
          batch_size: 2,
          sequence_length: 64,
          nproc_per_node: 1,
        },
      }),
    );
    writeFileSync(
      pipeline,
      JSON.stringify({
        version: 1,
        name: "adapter",
        target: "local",
        steps: [{ id: "baseline", uses: "evaluate", with: { model: "base" } }],
      }),
    );
    const program = createProgram("test");
    program.exitOverride();
    try {
      await expect(
        program.parseAsync(["node", "tt", "pipeline", "validate", "--spec", spec, "--file", pipeline]),
      ).rejects.toThrow(/is an adapter recipe/);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("rejects a leftover foundation pipeline file when --spec is adapter", async () => {
    const dir = mkdtempSync(join(tmpdir(), "tt-pipeline-"));
    const spec = join(dir, "tunedtensor.json");
    const pipeline = join(dir, "tunedtensor.pipeline.json");
    writeFileSync(
      spec,
      JSON.stringify({
        name: "adapter-spec",
        base_model: "Qwen/Qwen3.5-2B",
        system_prompt: "Answer questions.",
        guidelines: [],
        constraints: [],
        examples: [
          { input: "hello", output: "world" },
          { input: "thanks", output: "you are welcome" },
        ],
      }),
    );
    const program = createProgram("test");
    program.exitOverride();
    try {
      await program.parseAsync(["node", "tt", "pipeline", "init", "--engine", "foundation", "--file", pipeline]);
      await expect(
        program.parseAsync(["node", "tt", "pipeline", "validate", "--spec", spec, "--file", pipeline]),
      ).rejects.toThrow(/is a foundation recipe/);
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
