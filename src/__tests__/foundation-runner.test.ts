import { createHash } from "node:crypto";
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
  guidelines: ["Give the answer directly."],
  constraints: ["Do not add commentary."],
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

  it("compiles the shared spec instruction into every foundation stage", async () => {
    const dir = await mkdtemp(join(tmpdir(), "tt-foundation-instruction-"));
    dirs.push(dir);
    const recipe = pipelineFromFoundationHyperparameters(spec.name, spec.foundation);
    const plan = createExecutionPlan(recipe);
    const instructions: string[] = [];

    await runFoundationPipeline({
      spec,
      plan,
      specPath: join(dir, "tunedtensor.json"),
      outputDir: join(dir, "run"),
      spawnStep: async (args) => {
        const config = JSON.parse(await readFile(args.configPath, "utf8")) as {
          system_prompt: string;
        };
        instructions.push(config.system_prompt);
        await mockSpawn(args);
      },
    });

    expect(instructions).toHaveLength(plan.steps.length);
    expect(new Set(instructions)).toEqual(new Set([
      "Answer briefly.\n\nGuidelines:\n- Give the answer directly."
      + "\n\nConstraints:\n- Do not add commentary.",
    ]));
  });

  it("rejects validation data nested inside the training corpus before creating run state", async () => {
    const dir = await mkdtemp(join(tmpdir(), "tt-foundation-overlap-"));
    dirs.push(dir);
    const training = join(dir, "training");
    const validation = join(training, "held-out.jsonl");
    await mkdir(training);
    await writeFile(join(training, "train.txt"), "training text\n");
    await writeFile(validation, '{"text":"validation text"}\n');
    const outputDir = join(dir, "run");
    const overlappingSpec: LocalFoundationSpecFile = {
      ...spec,
      foundation: {
        ...spec.foundation,
        corpus_path: training,
        validation_path: validation,
      },
    };
    const plan = createExecutionPlan(
      pipelineFromFoundationHyperparameters(overlappingSpec.name, overlappingSpec.foundation),
      { only: ["tokenize"] },
    );
    let spawned = false;

    await expect(runFoundationPipeline({
      spec: overlappingSpec,
      plan,
      specPath: join(dir, "tunedtensor.json"),
      outputDir,
      spawnStep: async () => {
        spawned = true;
      },
    })).rejects.toThrow(/training.*validation.*overlap/i);
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

  it("rejects a symlinked resume root before spawning or writing through it", async () => {
    const dir = await mkdtemp(join(tmpdir(), "tt-foundation-resume-link-"));
    dirs.push(dir);
    const target = join(dir, "target");
    const outputDir = join(dir, "run");
    await mkdir(target);
    await symlink(target, outputDir, "dir");
    const recipe = pipelineFromFoundationHyperparameters(spec.name, spec.foundation);
    const plan = createExecutionPlan(recipe, { only: ["tokenize"] });
    let spawned = false;

    await expect(runFoundationPipeline({
      spec,
      plan,
      specPath: join(dir, "tunedtensor.json"),
      outputDir,
      resume: true,
      spawnStep: async () => {
        spawned = true;
      },
    })).rejects.toThrow(/symbolic link/i);
    expect(spawned).toBe(false);
    await expect(access(join(target, "tokenize"))).rejects.toThrow();
  });

  it("rejects preserved symlinks before reading resume configuration", async () => {
    const dir = await mkdtemp(join(tmpdir(), "tt-foundation-resume-entry-link-"));
    dirs.push(dir);
    const outputDir = join(dir, "run");
    const stepDir = join(outputDir, "tokenize");
    const target = join(dir, "external-config.json");
    await mkdir(join(stepDir, "output"), { recursive: true });
    await writeFile(target, "preserve me\n");
    await symlink(target, join(stepDir, "config.json"));
    const recipe = pipelineFromFoundationHyperparameters(spec.name, spec.foundation);
    const plan = createExecutionPlan(recipe, { only: ["tokenize"] });
    let spawned = false;

    await expect(runFoundationPipeline({
      spec,
      plan,
      specPath: join(dir, "tunedtensor.json"),
      outputDir,
      resume: true,
      spawnStep: async () => {
        spawned = true;
      },
    })).rejects.toThrow(/symbolic link/i);
    expect(spawned).toBe(false);
    expect(await readFile(target, "utf8")).toBe("preserve me\n");
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

  it("resumes verified stages and reruns only an interrupted pretrain step", async () => {
    const dir = await mkdtemp(join(tmpdir(), "tt-foundation-resume-"));
    dirs.push(dir);
    const specPath = join(dir, "tunedtensor.json");
    const outputDir = join(dir, "run");
    await writeFile(specPath, JSON.stringify(spec));
    const plan = createExecutionPlan(
      pipelineFromFoundationHyperparameters(spec.name, spec.foundation),
      { only: ["tokenize", "pretrain"] },
    );
    const firstCalls: string[] = [];
    await expect(runFoundationPipeline({
      spec,
      plan,
      specPath,
      outputDir,
      spawnStep: async (args) => {
        firstCalls.push(args.entrypoint);
        if (args.entrypoint === "pretrain.py") throw new Error("injected interruption");
        await mockSpawn(args);
      },
    })).rejects.toThrow(/injected interruption/);
    expect(firstCalls).toEqual(["train_tokenizer.py", "pretrain.py"]);

    const resumedCalls: string[] = [];
    const result = await runFoundationPipeline({
      spec,
      plan,
      specPath,
      outputDir,
      resume: true,
      spawnStep: async (args) => {
        resumedCalls.push(args.entrypoint);
        await mockSpawn(args);
      },
    });
    expect(result.status).toBe("succeeded");
    expect(resumedCalls).toEqual(["pretrain.py"]);
  });

  it("resumes completed beta.6 stages with a semantically identical raw system prompt", async () => {
    const dir = await mkdtemp(join(tmpdir(), "tt-foundation-legacy-resume-"));
    dirs.push(dir);
    const specPath = join(dir, "tunedtensor.json");
    const outputDir = join(dir, "run");
    const legacyCompatibleSpec: LocalFoundationSpecFile = {
      ...spec,
      system_prompt: "  Answer briefly.  ",
      guidelines: [],
      constraints: [],
    };
    const plan = createExecutionPlan(
      pipelineFromFoundationHyperparameters(legacyCompatibleSpec.name, legacyCompatibleSpec.foundation),
      { only: ["tokenize"] },
    );
    await runFoundationPipeline({
      spec: legacyCompatibleSpec,
      plan,
      specPath,
      outputDir,
      spawnStep: mockSpawn,
    });

    const configPath = join(outputDir, "tokenize", "config.json");
    const completionPath = join(outputDir, "tokenize", "completion.json");
    const config = JSON.parse(await readFile(configPath, "utf8")) as Record<string, unknown>;
    config.system_prompt = legacyCompatibleSpec.system_prompt;
    const legacyConfig = `${JSON.stringify(config, null, 2)}\n`;
    await writeFile(configPath, legacyConfig);
    const completion = JSON.parse(await readFile(completionPath, "utf8")) as Record<string, unknown>;
    completion.config_sha256 = createHash("sha256").update(legacyConfig).digest("hex");
    await writeFile(completionPath, `${JSON.stringify(completion, null, 2)}\n`);

    let spawned = false;
    await runFoundationPipeline({
      spec: legacyCompatibleSpec,
      plan,
      specPath,
      outputDir,
      resume: true,
      spawnStep: async () => {
        spawned = true;
      },
    });

    expect(spawned).toBe(false);
  });

  it("rejects beta.6 corpus-backed stages that did not ingest the instruction", async () => {
    const dir = await mkdtemp(join(tmpdir(), "tt-foundation-legacy-corpus-"));
    dirs.push(dir);
    const specPath = join(dir, "tunedtensor.json");
    const outputDir = join(dir, "run");
    const corpusPath = join(dir, "corpus.txt");
    await writeFile(corpusPath, "alpha beta gamma");
    const corpusSpec: LocalFoundationSpecFile = {
      ...spec,
      system_prompt: "Answer briefly.",
      guidelines: [],
      constraints: [],
      foundation: {
        ...spec.foundation,
        corpus_path: corpusPath,
      },
    };
    const plan = createExecutionPlan(
      pipelineFromFoundationHyperparameters(corpusSpec.name, corpusSpec.foundation),
      { only: ["tokenize"] },
    );
    await runFoundationPipeline({ spec: corpusSpec, plan, specPath, outputDir, spawnStep: mockSpawn });

    const configPath = join(outputDir, "tokenize", "config.json");
    const completionPath = join(outputDir, "tokenize", "completion.json");
    const config = JSON.parse(await readFile(configPath, "utf8")) as Record<string, unknown>;
    delete config.instruction_corpus_version;
    config.system_prompt = corpusSpec.system_prompt;
    const legacyConfig = `${JSON.stringify(config, null, 2)}\n`;
    await writeFile(configPath, legacyConfig);
    const completion = JSON.parse(await readFile(completionPath, "utf8")) as Record<string, unknown>;
    completion.config_sha256 = createHash("sha256").update(legacyConfig).digest("hex");
    await writeFile(completionPath, `${JSON.stringify(completion, null, 2)}\n`);

    await expect(runFoundationPipeline({
      spec: corpusSpec,
      plan,
      specPath,
      outputDir,
      resume: true,
      spawnStep: mockSpawn,
    })).rejects.toThrow(/configuration changed/i);
  });

  it("rejects beta.6 resume when legacy stages ignored current instruction lists", async () => {
    const dir = await mkdtemp(join(tmpdir(), "tt-foundation-legacy-drift-"));
    dirs.push(dir);
    const specPath = join(dir, "tunedtensor.json");
    const outputDir = join(dir, "run");
    const legacySpec: LocalFoundationSpecFile = {
      ...spec,
      guidelines: [],
      constraints: [],
    };
    const plan = createExecutionPlan(
      pipelineFromFoundationHyperparameters(legacySpec.name, legacySpec.foundation),
      { only: ["tokenize"] },
    );
    await runFoundationPipeline({ spec: legacySpec, plan, specPath, outputDir, spawnStep: mockSpawn });

    const configPath = join(outputDir, "tokenize", "config.json");
    const completionPath = join(outputDir, "tokenize", "completion.json");
    const config = JSON.parse(await readFile(configPath, "utf8")) as Record<string, unknown>;
    config.system_prompt = legacySpec.system_prompt;
    const legacyConfig = `${JSON.stringify(config, null, 2)}\n`;
    await writeFile(configPath, legacyConfig);
    const completion = JSON.parse(await readFile(completionPath, "utf8")) as Record<string, unknown>;
    completion.config_sha256 = createHash("sha256").update(legacyConfig).digest("hex");
    await writeFile(completionPath, `${JSON.stringify(completion, null, 2)}\n`);

    await expect(runFoundationPipeline({
      spec: {
        ...legacySpec,
        guidelines: ["This instruction was never used by the legacy stage."],
      },
      plan,
      specPath,
      outputDir,
      resume: true,
      spawnStep: mockSpawn,
    })).rejects.toThrow(/configuration changed/i);
  });

  it("reruns downstream stages after an earlier completion manifest is missing", async () => {
    const dir = await mkdtemp(join(tmpdir(), "tt-foundation-resume-dag-"));
    dirs.push(dir);
    const specPath = join(dir, "tunedtensor.json");
    const outputDir = join(dir, "run");
    const plan = createExecutionPlan(
      pipelineFromFoundationHyperparameters(spec.name, spec.foundation),
      { only: ["tokenize", "pretrain"] },
    );
    await runFoundationPipeline({ spec, plan, specPath, outputDir, spawnStep: mockSpawn });
    await rm(join(outputDir, "tokenize", "completion.json"));
    const resumedCalls: string[] = [];

    await runFoundationPipeline({
      spec,
      plan,
      specPath,
      outputDir,
      resume: true,
      spawnStep: async (args) => {
        resumedCalls.push(args.entrypoint);
        await mockSpawn(args);
      },
    });

    expect(resumedCalls).toEqual(["train_tokenizer.py", "pretrain.py"]);
  });

  it("refuses resume when the persisted step configuration changed", async () => {
    const dir = await mkdtemp(join(tmpdir(), "tt-foundation-resume-config-"));
    dirs.push(dir);
    const specPath = join(dir, "tunedtensor.json");
    const outputDir = join(dir, "run");
    const plan = createExecutionPlan(
      pipelineFromFoundationHyperparameters(spec.name, spec.foundation),
      { only: ["tokenize"] },
    );
    await runFoundationPipeline({ spec, plan, specPath, outputDir, spawnStep: mockSpawn });
    await expect(runFoundationPipeline({
      spec: { ...spec, system_prompt: "Changed after the first run." },
      plan,
      specPath,
      outputDir,
      resume: true,
      spawnStep: mockSpawn,
    })).rejects.toThrow(/configuration changed/i);
  });

  it("refuses resume when a completed stage artifact changed", async () => {
    const dir = await mkdtemp(join(tmpdir(), "tt-foundation-resume-integrity-"));
    dirs.push(dir);
    const specPath = join(dir, "tunedtensor.json");
    const outputDir = join(dir, "run");
    const plan = createExecutionPlan(
      pipelineFromFoundationHyperparameters(spec.name, spec.foundation),
      { only: ["tokenize"] },
    );
    await runFoundationPipeline({ spec, plan, specPath, outputDir, spawnStep: mockSpawn });
    await writeFile(
      join(outputDir, "tokenize", "output", "tokenizer.json"),
      '{"changed":true}\n',
    );
    let spawned = false;

    await expect(runFoundationPipeline({
      spec,
      plan,
      specPath,
      outputDir,
      resume: true,
      spawnStep: async () => {
        spawned = true;
      },
    })).rejects.toThrow(/integrity.*changed/i);
    expect(spawned).toBe(false);
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
