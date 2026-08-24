import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { pipelineFromFoundationHyperparameters, createExecutionPlan } from "../pipeline.js";
import { runFoundationPipeline, type FoundationStepSpawn } from "../local-runtime/foundation-runner.js";
import type { LocalFoundationSpecFile } from "../local-runtime/contracts.js";

const spec: LocalFoundationSpecFile = {
  engine: "foundation",
  name: "tiny-gpt",
  description: "",
  system_prompt: "Answer briefly.",
  guidelines: [],
  constraints: [],
  examples: [
    { input: "hello", output: "world" },
    { input: "What is 2 + 2?", output: "4" },
  ],
  foundation: {
    vocab_size: 256,
    max_chars: 2000,
    depth: 2,
    pretrain_steps: 2,
    finetune_steps: 2,
    rl_steps: 0,
    batch_size: 2,
    sequence_length: 64,
    nproc_per_node: 1,
  },
};

const mockSpawn: FoundationStepSpawn = async ({ configPath, entrypoint }) => {
  const config = JSON.parse(await readFile(configPath, "utf8")) as { output_dir: string };
  await mkdir(config.output_dir, { recursive: true });
  if (entrypoint === "tokenize.py") {
    await writeFile(join(config.output_dir, "tokenizer.json"), "{}\n");
  } else if (entrypoint === "evaluate.py") {
    await writeFile(join(config.output_dir, "report.json"), JSON.stringify({ ok: true, evaluator: "bpb" }) + "\n");
  } else {
    await writeFile(join(config.output_dir, "model.safetensors"), "fake");
    await writeFile(join(config.output_dir, "config.json"), "{}\n");
  }
  await writeFile(join(config.output_dir, "metrics.json"), JSON.stringify({ ok: true, entrypoint }) + "\n");
};

describe("foundation pipeline runner", () => {
  const dirs: string[] = [];
  afterEach(async () => {
    await Promise.all(dirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })));
  });

  it("threads tokenizer and model artifacts and writes a hashed report", async () => {
    const dir = await mkdtemp(join(tmpdir(), "tt-foundation-"));
    dirs.push(dir);
    const specPath = join(dir, "tunedtensor.json");
    await writeFile(specPath, JSON.stringify(spec));
    const recipe = pipelineFromFoundationHyperparameters(spec.name, spec.foundation);
    const plan = createExecutionPlan(recipe);
    const result = await runFoundationPipeline({
      spec,
      plan,
      specPath,
      outputDir: join(dir, "run"),
      spawnStep: mockSpawn,
    });
    expect(result.status).toBe("succeeded");
    expect(result.steps.map((step) => step.id)).toEqual(plan.steps.map((step) => step.id));
    expect(result.steps[0]?.artifacts.tokenizer?.sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(result.steps[1]?.artifacts.model?.sha256).toMatch(/^[a-f0-9]{64}$/);
    const report = JSON.parse(await readFile(result.report_path, "utf8")) as { status: string; steps: unknown[] };
    expect(report.status).toBe("succeeded");
    expect(report.steps).toHaveLength(plan.steps.length);
  });
});
