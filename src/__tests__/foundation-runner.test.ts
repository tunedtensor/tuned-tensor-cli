import { access, mkdir, mkdtemp, readFile, rm, stat, symlink, writeFile } from "node:fs/promises";
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
  if (entrypoint === "train_tokenizer.py") {
    await writeFile(join(config.output_dir, "tokenizer.json"), "{}\n");
  } else if (entrypoint === "evaluate.py") {
    await writeFile(join(config.output_dir, "report.json"), JSON.stringify({ ok: true, evaluator: "bpb" }) + "\n");
  } else {
    await writeFile(join(config.output_dir, "model.safetensors"), "fake");
    await writeFile(join(config.output_dir, "config.json"), JSON.stringify({
      depth: 2,
      width: 64,
      heads: 4,
      vocab_size: 256,
      sequence_length: 64,
    }) + "\n");
  }
  await writeFile(join(config.output_dir, "metrics.json"), JSON.stringify({ ok: true, entrypoint }) + "\n");
};

describe("foundation pipeline runner", () => {
  const dirs: string[] = [];
  afterEach(async () => {
    await Promise.all(dirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })));
  });

  it("rejects RL contracts without numeric rewards before creating run state", async () => {
    const dir = await mkdtemp(join(tmpdir(), "tt-foundation-"));
    dirs.push(dir);
    const outputDir = join(dir, "run");
    const nonNumericSpec: LocalFoundationSpecFile = {
      ...spec,
      examples: [
        { input: "hello", output: "world" },
        { input: "goodbye", output: "moon" },
      ],
      foundation: { ...spec.foundation, rl_steps: 1 },
    };
    const plan = createExecutionPlan(
      pipelineFromFoundationHyperparameters(nonNumericSpec.name, nonNumericSpec.foundation),
    );
    let spawned = false;

    await expect(runFoundationPipeline({
      spec: nonNumericSpec,
      plan,
      specPath: join(dir, "tunedtensor.json"),
      outputDir,
      spawnStep: async (args) => {
        spawned = true;
        await mockSpawn(args);
      },
    })).rejects.toThrow(/RL.*numeric expected output/i);
    expect(spawned).toBe(false);
    await expect(access(outputDir)).rejects.toThrow();
  });

  it("rejects unsupported multi-process training before creating run state", async () => {
    const dir = await mkdtemp(join(tmpdir(), "tt-foundation-"));
    dirs.push(dir);
    const specPath = join(dir, "tunedtensor.json");
    const outputDir = join(dir, "run");
    const multiProcessSpec: LocalFoundationSpecFile = {
      ...spec,
      foundation: { ...spec.foundation, nproc_per_node: 2 },
    };
    const plan = createExecutionPlan(
      pipelineFromFoundationHyperparameters(multiProcessSpec.name, multiProcessSpec.foundation),
    );
    let spawned = false;

    await expect(runFoundationPipeline({
      spec: multiProcessSpec,
      plan,
      specPath,
      outputDir,
      spawnStep: async (args) => {
        spawned = true;
        await mockSpawn(args);
      },
    })).rejects.toThrow(/nproc_per_node.*only 1/i);
    expect(spawned).toBe(false);
    await expect(access(outputDir)).rejects.toThrow();
  });

  it("rejects unsupported compare steps before creating run state", async () => {
    const dir = await mkdtemp(join(tmpdir(), "tt-foundation-"));
    dirs.push(dir);
    const outputDir = join(dir, "run");
    const recipe = pipelineFromFoundationHyperparameters(spec.name, spec.foundation);
    recipe.steps.push({
      id: "compare",
      uses: "compare",
      target: "local",
      with: { before: { from: "bpb.report" }, after: { from: "chat.report" } },
    });
    const plan = createExecutionPlan(recipe);
    let spawned = false;

    await expect(runFoundationPipeline({
      spec,
      plan,
      specPath: join(dir, "tunedtensor.json"),
      outputDir,
      spawnStep: async () => {
        spawned = true;
      },
    })).rejects.toThrow(/does not implement compare/i);
    expect(spawned).toBe(false);
    await expect(access(outputDir)).rejects.toThrow();
  });

  it("rejects missing required step outputs", async () => {
    const dir = await mkdtemp(join(tmpdir(), "tt-foundation-"));
    dirs.push(dir);
    const recipe = pipelineFromFoundationHyperparameters(spec.name, spec.foundation);
    const plan = createExecutionPlan(recipe, { only: ["tokenize", "pretrain"] });

    await expect(runFoundationPipeline({
      spec,
      plan,
      specPath: join(dir, "tunedtensor.json"),
      outputDir: join(dir, "run"),
      spawnStep: async ({ configPath, entrypoint }) => {
        const config = JSON.parse(await readFile(configPath, "utf8")) as { output_dir: string };
        await mkdir(config.output_dir, { recursive: true });
        if (entrypoint === "train_tokenizer.py") {
          await writeFile(join(config.output_dir, "tokenizer.json"), "{}\n");
        } else {
          await writeFile(join(config.output_dir, "config.json"), JSON.stringify({
            depth: 2,
            width: 64,
            heads: 4,
            vocab_size: 256,
            sequence_length: 64,
          }) + "\n");
        }
        await writeFile(join(config.output_dir, "metrics.json"), "{\"ok\":true}\n");
      },
    })).rejects.toThrow(/pretrain.*model\.safetensors/i);
  });

  it.each([
    ["missing", undefined, /missing or unreadable/i],
    ["malformed", "{", /not valid JSON/i],
    ["unsuccessful", "{\"ok\":false}\n", /ok: true/i],
  ])("rejects %s step metrics", async (_case, metrics, expected) => {
    const dir = await mkdtemp(join(tmpdir(), "tt-foundation-"));
    dirs.push(dir);
    const recipe = pipelineFromFoundationHyperparameters(spec.name, spec.foundation);
    const plan = createExecutionPlan(recipe, { only: ["tokenize"] });

    await expect(runFoundationPipeline({
      spec,
      plan,
      specPath: join(dir, "tunedtensor.json"),
      outputDir: join(dir, "run"),
      spawnStep: async ({ configPath }) => {
        const config = JSON.parse(await readFile(configPath, "utf8")) as { output_dir: string };
        await mkdir(config.output_dir, { recursive: true });
        await writeFile(join(config.output_dir, "tokenizer.json"), "{}\n");
        if (metrics !== undefined) {
          await writeFile(join(config.output_dir, "metrics.json"), metrics);
        }
      },
    })).rejects.toThrow(expected);
  });

  it("rejects symlinks inside produced artifacts", async () => {
    const dir = await mkdtemp(join(tmpdir(), "tt-foundation-"));
    dirs.push(dir);
    const recipe = pipelineFromFoundationHyperparameters(spec.name, spec.foundation);
    const plan = createExecutionPlan(recipe, { only: ["tokenize"] });

    await expect(runFoundationPipeline({
      spec,
      plan,
      specPath: join(dir, "tunedtensor.json"),
      outputDir: join(dir, "run"),
      spawnStep: async ({ configPath }) => {
        const config = JSON.parse(await readFile(configPath, "utf8")) as { output_dir: string };
        await mkdir(config.output_dir, { recursive: true });
        await writeFile(join(config.output_dir, "tokenizer.json"), "{}\n");
        await writeFile(join(config.output_dir, "metrics.json"), "{\"ok\":true}\n");
        await symlink("/etc/hosts", join(config.output_dir, "external"));
      },
    })).rejects.toThrow(/symbolic link/i);
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
    expect((await stat(join(dir, "run"))).mode & 0o777).toBe(0o700);
    expect((await stat(join(dir, "run", "tokenize", "config.json"))).mode & 0o777).toBe(0o600);
    expect((await stat(result.report_path)).mode & 0o777).toBe(0o600);
    const report = JSON.parse(await readFile(result.report_path, "utf8")) as { status: string; steps: unknown[] };
    expect(report.status).toBe("succeeded");
    expect(report.steps).toHaveLength(plan.steps.length);
  });

  it("normalizes every generated step directory and artifact to private modes", async () => {
    const dir = await mkdtemp(join(tmpdir(), "tt-foundation-"));
    dirs.push(dir);
    const recipe = pipelineFromFoundationHyperparameters(spec.name, spec.foundation);
    const plan = createExecutionPlan(recipe, { only: ["tokenize", "pretrain"] });

    await runFoundationPipeline({
      spec,
      plan,
      specPath: join(dir, "tunedtensor.json"),
      outputDir: join(dir, "run"),
      spawnStep: mockSpawn,
    });

    if (process.platform !== "win32") {
      expect((await stat(join(dir, "run", "tokenize"))).mode & 0o777).toBe(0o700);
      expect((await stat(join(dir, "run", "tokenize", "output"))).mode & 0o777).toBe(0o700);
      expect((await stat(join(dir, "run", "tokenize", "output", "tokenizer.json"))).mode & 0o777).toBe(0o600);
      expect((await stat(join(dir, "run", "pretrain", "output", "config.json"))).mode & 0o777).toBe(0o600);
      expect((await stat(join(dir, "run", "pretrain", "output", "model.safetensors"))).mode & 0o777).toBe(0o600);
    }
  });
});
