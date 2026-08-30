import { createHash, randomUUID } from "node:crypto";
import { createReadStream } from "node:fs";
import { chmod, lstat, mkdir, open, readdir, readFile, rename, rm } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { DEFAULT_FOUNDATION_RUNS_DIR } from "../paths.js";
import type { LocalFoundationSpecFile } from "./contracts.js";
import { buildSystemMessage } from "./dataset.js";
import { localPathsOverlap } from "./local-project.js";
import type { ExecutionPlan } from "../pipeline.js";
import {
  buildFoundationPythonCommand,
  runLoggedProcess,
  withFoundationPythonEnvironment,
  type FoundationPythonEntrypoint,
} from "./process-runner.js";

export interface FoundationStepArtifact {
  path: string;
  sha256: string;
}

export interface FoundationStepResult {
  id: string;
  uses: string;
  metrics: Record<string, unknown>;
  artifacts: Record<string, FoundationStepArtifact>;
}

export interface FoundationPipelineResult {
  status: "succeeded";
  name?: string;
  report_path: string;
  steps: FoundationStepResult[];
}

export interface FoundationStepSpawnArgs {
  entrypoint: FoundationPythonEntrypoint;
  configPath: string;
  logPath: string;
  stepId: string;
}

export type FoundationStepSpawn = (args: FoundationStepSpawnArgs) => Promise<void>;

interface ProducedArtifacts {
  tokenizer?: string;
  model?: string;
  report?: string;
}

const INSTRUCTION_CORPUS_VERSION = 1;

function isRef(value: unknown): value is { from: string } {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value)
    && typeof (value as { from?: unknown }).from === "string";
}

function resumeConfigMatches(
  previous: unknown,
  current: Record<string, unknown>,
  legacySystemPrompt: string,
): boolean {
  if (JSON.stringify(previous) === JSON.stringify(current)) return true;
  if (!previous || typeof previous !== "object" || Array.isArray(previous)) return false;
  const legacyInstruction = legacySystemPrompt.trim();
  if (!legacyInstruction || current.system_prompt !== legacyInstruction) return false;
  const migrated = { ...(previous as Record<string, unknown>) };
  if (migrated.system_prompt !== legacySystemPrompt) return false;
  migrated.system_prompt = legacyInstruction;
  return JSON.stringify(migrated) === JSON.stringify(current);
}

function parseRef(value: { from: string }): { stepId: string; kind: string } {
  const index = value.from.lastIndexOf(".");
  if (index <= 0) throw new Error(`Invalid artifact reference: ${value.from}`);
  return { stepId: value.from.slice(0, index), kind: value.from.slice(index + 1) };
}

async function hashPath(path: string): Promise<string> {
  const info = await lstat(path);
  if (info.isSymbolicLink()) {
    throw new Error(`Foundation artifacts must not contain symbolic links: ${path}`);
  }
  const hash = createHash("sha256");
  if (info.isDirectory()) {
    for (const name of (await readdir(path)).sort()) {
      hash.update(name);
      hash.update(await hashPath(join(path, name)));
    }
    return hash.digest("hex");
  }
  if (!info.isFile()) {
    throw new Error(`Foundation artifacts must contain only regular files and directories: ${path}`);
  }
  await new Promise<void>((done, reject) => {
    const stream = createReadStream(path);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.on("error", reject);
    stream.on("end", () => done());
  });
  return hash.digest("hex");
}

async function makePrivateTree(path: string): Promise<void> {
  const info = await lstat(path);
  if (info.isSymbolicLink()) {
    throw new Error(`Foundation artifacts must not contain symbolic links: ${path}`);
  }
  if (info.isDirectory()) {
    await chmod(path, 0o700);
    for (const name of await readdir(path)) {
      await makePrivateTree(join(path, name));
    }
    return;
  }
  if (info.isFile()) {
    await chmod(path, 0o600);
    return;
  }
  throw new Error(`Foundation artifacts must contain only regular files and directories: ${path}`);
}

async function readRequiredJsonObject(
  path: string,
  description: string,
): Promise<Record<string, unknown>> {
  let content: string;
  try {
    content = await readFile(path, "utf8");
  } catch (error) {
    throw new Error(`${description} is missing or unreadable: ${path}`, { cause: error });
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(content) as unknown;
  } catch (error) {
    throw new Error(`${description} is not valid JSON: ${path}`, { cause: error });
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`${description} must be a JSON object: ${path}`);
  }
  return parsed as Record<string, unknown>;
}

async function writePrivateTextAtomic(path: string, content: string): Promise<void> {
  const temporary = join(dirname(path), `.${basename(path)}.${randomUUID()}.tmp`);
  const file = await open(temporary, "wx", 0o600);
  let closed = false;
  try {
    await file.chmod(0o600);
    await file.writeFile(content, "utf8");
    await file.sync();
    await file.close();
    closed = true;
    await rename(temporary, path);
  } finally {
    if (!closed) await file.close().catch(() => undefined);
    await rm(temporary, { force: true }).catch(() => undefined);
  }
}

async function artifactFor(path: string): Promise<FoundationStepArtifact> {
  return { path, sha256: await hashPath(path) };
}

async function requireRegularNonemptyFile(path: string, description: string): Promise<void> {
  let info;
  try {
    info = await lstat(path);
  } catch (error) {
    throw new Error(`${description} is missing: ${path}`, { cause: error });
  }
  if (info.isSymbolicLink() || !info.isFile() || info.size === 0) {
    throw new Error(`${description} must be a non-empty regular file: ${path}`);
  }
}

function requirePositiveIntegerFields(
  value: Record<string, unknown>,
  fields: string[],
  description: string,
): void {
  for (const field of fields) {
    if (!Number.isInteger(value[field]) || Number(value[field]) <= 0) {
      throw new Error(`${description}.${field} must be a positive integer.`);
    }
  }
}

function stepConfig(step: ExecutionPlan["steps"][number]): Record<string, unknown> {
  if (step.uses === "train") return {};
  if (step.uses === "tokenize") return { ...(step.with ?? {}) };
  return { ...step.with };
}

function assertFoundationPlanSupported(
  spec: LocalFoundationSpecFile,
  plan: ExecutionPlan,
): void {
  const supported = new Set(["tokenize", "pretrain", "finetune", "rl", "evaluate"]);
  const unsupported = plan.steps.find((step) => !supported.has(step.uses));
  if (unsupported) {
    throw new Error(
      `Foundation runner does not implement ${unsupported.uses} (step ${unsupported.id}).`,
    );
  }
  if (
    spec.foundation.corpus_path
    && spec.foundation.validation_path
    && localPathsOverlap(spec.foundation.corpus_path, spec.foundation.validation_path)
  ) {
    throw new Error(
      "Foundation training and validation paths overlap; use disjoint corpora for held-out evaluation.",
    );
  }
  if (plan.steps.some((step) => step.uses === "rl")) {
    const invalidReward = spec.examples.findIndex(
      (example) => !/-?\d+(?:\.\d+)?/.test(example.output.replaceAll(",", "")),
    );
    if (invalidReward >= 0) {
      throw new Error(
        `Foundation RL requires a numeric expected output for every example; examples[${invalidReward}].output has none.`,
      );
    }
  }
  for (const step of plan.steps) {
    if (step.uses !== "pretrain") continue;
    const processCount = step.with.nprocPerNode ?? spec.foundation.nproc_per_node;
    if (processCount !== 1) {
      throw new Error(
        `Foundation nproc_per_node is ${processCount}; only 1 is supported by the current single-process runtime.`,
      );
    }
  }
}

export async function defaultFoundationSpawn(args: FoundationStepSpawnArgs): Promise<void> {
  const launched = buildFoundationPythonCommand(args.entrypoint, ["--config", args.configPath]);
  const result = await runLoggedProcess({
    command: launched.command,
    commandArgs: launched.commandArgs,
    env: withFoundationPythonEnvironment(process.env),
    logPath: args.logPath,
    stage: args.stepId,
    shutdownGraceMs: 120_000,
    appendLog: true,
  });
  if (result.exitCode !== 0) {
    throw new Error(
      `Foundation step "${args.stepId}" failed (exit ${result.exitCode}).${result.stderr ? `\n${result.stderr}` : ""}`,
    );
  }
}

export async function runFoundationPipeline(args: {
  spec: LocalFoundationSpecFile;
  plan: ExecutionPlan;
  specPath: string;
  outputDir?: string;
  resume?: boolean;
  spawnStep?: FoundationStepSpawn;
}): Promise<FoundationPipelineResult> {
  assertFoundationPlanSupported(args.spec, args.plan);
  const outputDir = resolve(
    args.outputDir ?? join(dirname(resolve(args.specPath)), DEFAULT_FOUNDATION_RUNS_DIR, randomUUID()),
  );
  let outputExists = false;
  try {
    const outputInfo = await lstat(outputDir);
    if (!args.resume) {
      throw new Error(`Foundation output directory already exists; pass --resume ${outputDir} to recover it.`);
    }
    if (outputInfo.isSymbolicLink()) {
      throw new Error(`Foundation resume directory must not be a symbolic link: ${outputDir}`);
    }
    if (!outputInfo.isDirectory()) {
      throw new Error(`Foundation resume path must be a directory: ${outputDir}`);
    }
    outputExists = true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }
  if (!outputExists) {
    await mkdir(outputDir, { recursive: true, mode: 0o700 });
  }
  await makePrivateTree(outputDir);
  const spawnStep = args.spawnStep ?? defaultFoundationSpawn;
  const produced = new Map<string, ProducedArtifacts>();
  const steps: FoundationStepResult[] = [];
  const examples = args.spec.examples.map((example) => ({ input: example.input, output: example.output }));
  const systemInstruction = buildSystemMessage(args.spec);
  const hp = args.spec.foundation;
  let tokenizerDir: string | undefined;
  let modelDir: string | undefined;
  let upstreamExecuted = false;

  const resolveKind = (stepId: string, kind: string): string => {
    const path = produced.get(stepId)?.[kind as keyof ProducedArtifacts];
    if (!path) throw new Error(`Step ${stepId} did not produce ${kind}.`);
    return path;
  };

  const resolveFrom = (value: unknown, kind: string, fallback?: string): string => {
    if (isRef(value)) return resolveKind(parseRef(value).stepId, parseRef(value).kind);
    if (fallback) return fallback;
    throw new Error(`Missing ${kind} artifact.`);
  };

  for (const step of args.plan.steps) {
    const stepDir = join(outputDir, step.id);
    const output = join(stepDir, "output");
    await mkdir(stepDir, { recursive: true, mode: 0o700 });
    await chmod(stepDir, 0o700);
    await mkdir(output, { recursive: true, mode: 0o700 });
    await chmod(output, 0o700);
    const configPath = join(stepDir, "config.json");
    const logPath = join(stepDir, "step.log");
    const metricsPath = join(output, "metrics.json");
    const completionPath = join(stepDir, "completion.json");
    const fields = stepConfig(step);
    let entrypoint: FoundationPythonEntrypoint;
    let config: Record<string, unknown>;

    if (step.uses === "tokenize") {
      entrypoint = "train_tokenizer.py";
      config = {
        output_dir: output,
        vocab_size: fields.vocabSize ?? hp.vocab_size,
        max_chars: hp.tokenizer_max_chars ?? fields.maxChars ?? hp.max_chars,
        corpus_path: hp.corpus_path,
        ...(hp.corpus_path ? { instruction_corpus_version: INSTRUCTION_CORPUS_VERSION } : {}),
        system_prompt: systemInstruction,
        examples,
      };
    } else if (step.uses === "pretrain") {
      tokenizerDir = resolveFrom(fields.tokenizer, "tokenizer", tokenizerDir);
      entrypoint = "pretrain.py";
      config = {
        tokenizer_dir: tokenizerDir,
        output_dir: output,
        work_dir: join(stepDir, "recovery"),
        depth: fields.depth ?? hp.depth,
        steps: fields.steps ?? hp.pretrain_steps,
        batch_size: fields.batchSize ?? hp.batch_size,
        sequence_length: fields.sequenceLength ?? hp.sequence_length,
        nproc_per_node: fields.nprocPerNode ?? hp.nproc_per_node,
        max_chars: hp.max_chars,
        corpus_path: hp.corpus_path,
        ...(hp.corpus_path ? { instruction_corpus_version: INSTRUCTION_CORPUS_VERSION } : {}),
        seed: hp.seed,
        learning_rate: hp.learning_rate,
        weight_decay: hp.weight_decay,
        warmup_steps: hp.warmup_steps,
        min_lr_ratio: hp.min_lr_ratio,
        gradient_accumulation_steps: hp.gradient_accumulation_steps,
        gradient_clip: hp.gradient_clip,
        bf16: hp.bf16,
        checkpoint_interval_steps: hp.checkpoint_interval_steps,
        checkpoint_interval_seconds: hp.checkpoint_interval_seconds,
        keep_checkpoints: hp.keep_checkpoints,
        log_interval_steps: hp.log_interval_steps,
        checkpoint_backup_dir: hp.checkpoint_backup_dir,
        resume: true,
        system_prompt: systemInstruction,
        examples,
      };
    } else if (step.uses === "finetune") {
      modelDir = resolveFrom(fields.model, "model", modelDir);
      entrypoint = "finetune.py";
      config = {
        model_dir: modelDir,
        tokenizer_dir: tokenizerDir,
        output_dir: output,
        steps: fields.steps ?? hp.finetune_steps,
        batch_size: fields.batchSize ?? hp.batch_size,
        sequence_length: hp.sequence_length,
        system_prompt: systemInstruction,
        examples,
      };
    } else if (step.uses === "rl") {
      modelDir = resolveFrom(fields.model, "model", modelDir);
      entrypoint = "rl.py";
      config = {
        model_dir: modelDir,
        tokenizer_dir: tokenizerDir,
        output_dir: output,
        steps: fields.steps ?? hp.rl_steps,
        system_prompt: systemInstruction,
        examples,
      };
    } else if (step.uses === "evaluate") {
      modelDir = resolveFrom(fields.model, "model", modelDir);
      entrypoint = "evaluate.py";
      config = {
        evaluator: fields.evaluator ?? "bpb",
        model_dir: modelDir,
        tokenizer_dir: tokenizerDir,
        output_dir: output,
        sequence_length: hp.sequence_length,
        max_chars: hp.max_chars,
        validation_path: hp.validation_path,
        system_prompt: systemInstruction,
        examples,
      };
    } else {
      throw new Error(`Foundation runner does not implement ${step.uses} (step ${step.id}).`);
    }

    if ((step.uses === "finetune" || step.uses === "rl" || step.uses === "evaluate") && !tokenizerDir) {
      throw new Error(`Step ${step.id} needs a tokenizer from an earlier tokenize step.`);
    }

    let completedStep = false;
    if (args.resume) {
      try {
        const previousConfig = JSON.parse(await readFile(configPath, "utf8")) as unknown;
        // beta.6 persisted only the raw system_prompt. Accept that exact legacy
        // shape without rewriting completed state; interrupted stages are
        // rewritten below with the compiled instruction before they run.
        if (!resumeConfigMatches(previousConfig, config, args.spec.system_prompt)) {
          throw new Error(
            `Foundation step "${step.id}" configuration changed; resume requires the original spec and pipeline.`,
          );
        }
        if (!upstreamExecuted) {
          await lstat(metricsPath);
          await lstat(completionPath);
          completedStep = true;
        }
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      }
    }
    if (!completedStep) {
      await writePrivateTextAtomic(configPath, `${JSON.stringify(config, null, 2)}\n`);
      await spawnStep({ entrypoint, configPath, logPath, stepId: step.id });
      upstreamExecuted = true;
    }
    await makePrivateTree(stepDir);

    const metrics = await readRequiredJsonObject(
      metricsPath,
      `Foundation step "${step.id}" metrics`,
    );
    if (metrics.ok !== true) {
      throw new Error(`Foundation step "${step.id}" metrics must report ok: true.`);
    }
    const artifacts: Record<string, FoundationStepArtifact> = {};
    const record: ProducedArtifacts = {};
    if (step.uses === "tokenize") {
      const tokenizerPath = join(output, "tokenizer.json");
      await requireRegularNonemptyFile(tokenizerPath, `Foundation step "${step.id}" tokenizer.json`);
      await readRequiredJsonObject(tokenizerPath, `Foundation step "${step.id}" tokenizer.json`);
      record.tokenizer = output;
      tokenizerDir = output;
      artifacts.tokenizer = await artifactFor(output);
    } else if (step.uses === "pretrain" || step.uses === "finetune" || step.uses === "rl") {
      const modelPath = join(output, "model.safetensors");
      const modelConfigPath = join(output, "config.json");
      await requireRegularNonemptyFile(modelPath, `Foundation step "${step.id}" model.safetensors`);
      const modelConfig = await readRequiredJsonObject(
        modelConfigPath,
        `Foundation step "${step.id}" config.json`,
      );
      requirePositiveIntegerFields(
        modelConfig,
        ["depth", "width", "heads", "vocab_size", "sequence_length"],
        `Foundation step "${step.id}" config.json`,
      );
      record.model = output;
      modelDir = output;
      artifacts.model = await artifactFor(output);
    } else if (step.uses === "evaluate") {
      const reportPath = join(output, "report.json");
      const stepReport = await readRequiredJsonObject(
        reportPath,
        `Foundation step "${step.id}" report.json`,
      );
      if (stepReport.ok !== true) {
        throw new Error(`Foundation step "${step.id}" report.json must report ok: true.`);
      }
      record.report = reportPath;
      artifacts.report = await artifactFor(reportPath);
    }
    const completion = {
      version: 1,
      config_sha256: await hashPath(configPath),
      metrics_sha256: await hashPath(metricsPath),
      artifacts,
    };
    if (completedStep) {
      const previousCompletion = await readRequiredJsonObject(
        completionPath,
        `Foundation step "${step.id}" completion manifest`,
      );
      if (JSON.stringify(previousCompletion) !== JSON.stringify(completion)) {
        throw new Error(
          `Foundation step "${step.id}" integrity changed after completion; use a new run directory.`,
        );
      }
    } else {
      await writePrivateTextAtomic(completionPath, `${JSON.stringify(completion, null, 2)}\n`);
    }
    produced.set(step.id, record);
    steps.push({ id: step.id, uses: step.uses, metrics, artifacts });
  }

  const report = {
    status: "succeeded" as const,
    ...(args.plan.name ? { name: args.plan.name } : {}),
    spec: args.specPath,
    steps,
  };
  const reportPath = join(outputDir, "report.json");
  await writePrivateTextAtomic(reportPath, `${JSON.stringify(report, null, 2)}\n`);
  return {
    status: "succeeded",
    ...(args.plan.name ? { name: args.plan.name } : {}),
    report_path: reportPath,
    steps,
  };
}
