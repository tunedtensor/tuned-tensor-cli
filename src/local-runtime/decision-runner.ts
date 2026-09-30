import { randomUUID } from "node:crypto";
import { lstat, mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { DEFAULT_DECISION_RUNS_DIR } from "../paths.js";
import type { ExecutionPlan } from "../pipeline.js";
import { fileUri } from "./artifacts.js";
import {
  decisionLabels,
  evalReportSchema,
  type DecisionQuestion,
  type EvalReport,
  type LocalDecisionSpecFile,
} from "./contracts.js";
import { buildSystemMessage } from "./dataset.js";
import { resolveDecisionModel } from "./decision-models.js";
import { compareEvalReports, deriveSampleSeed, splitSpecExamples } from "./evaluation.js";
import {
  buildDecisionPythonCommand,
  runLoggedProcess,
  withDecisionPythonEnvironment,
  type DecisionPythonEntrypoint,
} from "./process-runner.js";

export interface DecisionStepSpawnArgs {
  entrypoint: DecisionPythonEntrypoint;
  configPath: string;
  logPath: string;
  stepId: string;
  onProgress?: (message: string) => void;
}

export type DecisionStepSpawn = (args: DecisionStepSpawnArgs) => Promise<void>;

/** Calibration-aware metrics a label accuracy alone would hide. */
export interface DecisionMetrics {
  accuracy: number;
  log_loss: number;
  brier: number;
  mean_confidence: number;
  per_label: Record<string, { support: number; correct: number; recall: number }>;
}

export interface DecisionStepResult {
  id: string;
  uses: string;
  model?: string;
  report?: EvalReport;
  decision_metrics?: DecisionMetrics;
  training?: Record<string, unknown>;
  comparison?: ReturnType<typeof compareEvalReports> & { log_loss_delta: number; brier_delta: number };
}

export interface DecisionPipelineResult {
  status: "succeeded";
  engine: "decision";
  name?: string;
  run_dir: string;
  report_path: string;
  model_dir?: string;
  steps: DecisionStepResult[];
}

interface Prediction {
  id: string;
  prediction: string;
  probabilities: Record<string, number>;
  confidence: number;
  latency_ms: number;
}

const PROBABILITY_FLOOR = 1e-6;

function assertDecisionPlanSupported(plan: ExecutionPlan): void {
  for (const step of plan.steps) {
    if (step.uses === "evaluate" && step.with.evaluator !== "behavior") {
      throw new Error(`Decision runner evaluates with the behavior evaluator; step ${step.id} uses ${step.with.evaluator}.`);
    }
    if (!["train", "evaluate", "compare"].includes(step.uses)) {
      throw new Error(`Decision runner does not implement ${step.uses} (step ${step.id}).`);
    }
  }
}

/** The typed question the model answers; the shared behavior fields become its instructions. */
export function decisionQuestion(spec: LocalDecisionSpecFile): DecisionQuestion & { instructions: string } {
  return { ...spec.decision, instructions: buildSystemMessage(spec) };
}

function canonicalLabel(labels: string[], output: string): string {
  const normalized = output.trim().toLowerCase();
  const label = labels.find((candidate) => candidate.trim().toLowerCase() === normalized);
  if (!label) throw new Error(`Example output ${JSON.stringify(output)} is not a decision label (${labels.join(", ")}).`);
  return label;
}

async function readPredictions(path: string, ids: string[], labels: string[]): Promise<Prediction[]> {
  const rows = (await readFile(path, "utf8")).split("\n").filter((line) => line.trim())
    .map((line) => JSON.parse(line) as Prediction);
  const byId = new Map(rows.map((row) => [row.id, row]));
  return ids.map((id) => {
    const row = byId.get(id);
    if (!row) throw new Error(`Decision predictions are missing row ${id}: ${path}`);
    if (!labels.includes(row.prediction)) throw new Error(`Decision prediction for ${id} is not a label: ${row.prediction}`);
    return row;
  });
}

/** Join predictions to trusted references and score them in the shared report shape. */
export function scoreDecisionPredictions(args: {
  kind: "baseline" | "candidate";
  modelId: string;
  labels: string[];
  examples: Array<{ input: string; output: string }>;
  predictions: Prediction[];
  outputPath: string;
  sampleSeed: number;
}): { report: EvalReport; metrics: DecisionMetrics } {
  const perLabel: DecisionMetrics["per_label"] = Object.fromEntries(
    args.labels.map((label) => [label, { support: 0, correct: 0, recall: 0 }]),
  );
  let logLoss = 0;
  let brier = 0;
  let confidence = 0;
  const results = args.examples.map((example, index) => {
    const prediction = args.predictions[index]!;
    const passed = prediction.prediction === example.output;
    const expectedProbability = prediction.probabilities[example.output] ?? 0;
    logLoss -= Math.log(Math.max(expectedProbability, PROBABILITY_FLOOR));
    brier += args.labels.reduce((sum, label) => {
      const target = label === example.output ? 1 : 0;
      return sum + ((prediction.probabilities[label] ?? 0) - target) ** 2;
    }, 0);
    confidence += prediction.confidence;
    const entry = perLabel[example.output]!;
    entry.support += 1;
    if (passed) entry.correct += 1;
    return {
      prompt: example.input,
      expected: example.output,
      actual: prediction.prediction,
      passed,
      score: passed ? 1 : 0,
      reasoning: `p(expected)=${expectedProbability.toFixed(4)}; confidence=${prediction.confidence.toFixed(4)}`,
      latency_ms: Math.max(0, Math.round(prediction.latency_ms)),
      scored_by: "exact_match" as const,
    };
  });
  for (const entry of Object.values(perLabel)) entry.recall = entry.support ? entry.correct / entry.support : 0;
  const total = results.length;
  const mean = (value: number) => (total ? value / total : 0);
  const accuracy = mean(results.filter((result) => result.passed).length);
  const report = evalReportSchema.parse({
    kind: args.kind,
    model_id: args.modelId,
    total,
    eval_examples_total: total,
    eval_examples_used: total,
    eval_truncated: false,
    eval_split: "spec_holdout",
    eval_sample_seed: args.sampleSeed,
    avg_score: accuracy,
    pass_rate: accuracy,
    exact_match_rate: accuracy,
    avg_latency_ms: Math.round(mean(results.reduce((sum, result) => sum + result.latency_ms, 0))),
    results,
    artifact_uri: fileUri(args.outputPath),
    scoring_method: "exact_match",
    scoring_mode: "exact_match",
  });
  return {
    report,
    metrics: {
      accuracy,
      log_loss: mean(logLoss),
      brier: mean(brier),
      mean_confidence: mean(confidence),
      per_label: perLabel,
    },
  };
}

export async function defaultDecisionSpawn(args: DecisionStepSpawnArgs): Promise<void> {
  const launched = buildDecisionPythonCommand(args.entrypoint, ["--config", args.configPath]);
  const result = await runLoggedProcess({
    command: launched.command,
    commandArgs: launched.commandArgs,
    env: withDecisionPythonEnvironment(process.env),
    logPath: args.logPath,
    stage: args.stepId,
    appendLog: true,
    onLine: (line, stream) => {
      if (stream !== "stdout" || !args.onProgress) return;
      try {
        const event = JSON.parse(line) as { event?: string; epoch?: number; step?: number; total_steps?: number; loss?: number };
        if (event.event === "step") {
          args.onProgress(`${args.stepId}: step ${event.step}/${event.total_steps} (epoch ${event.epoch}) loss ${event.loss}`);
        }
      } catch {
        // Only structured progress lines are relayed; everything else stays in the log.
      }
    },
  });
  if (result.exitCode !== 0) {
    throw new Error(
      `Decision step "${args.stepId}" failed (exit ${result.exitCode}). Log: ${args.logPath}${result.stderr ? `\n${result.stderr}` : ""}`,
    );
  }
}

async function requireRegularNonemptyFile(path: string, description: string): Promise<void> {
  const info = await lstat(path).catch(() => null);
  if (!info || info.isSymbolicLink() || !info.isFile() || info.size === 0) {
    throw new Error(`${description} must be a non-empty regular file: ${path}`);
  }
}

async function readJsonObject(path: string, description: string): Promise<Record<string, unknown>> {
  const parsed = JSON.parse(await readFile(path, "utf8").catch((error) => {
    throw new Error(`${description} is missing or unreadable: ${path}`, { cause: error });
  })) as unknown;
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`${description} must be a JSON object: ${path}`);
  }
  return parsed as Record<string, unknown>;
}

/** Examples and predictions can be sensitive; keep every run file owner-only. */
async function writePrivateAtomic(path: string, content: string): Promise<void> {
  await mkdir(dirname(path), { recursive: true, mode: 0o700 });
  const temporary = `${path}.${process.pid}.${randomUUID()}.tmp`;
  try {
    await writeFile(temporary, content, { encoding: "utf8", mode: 0o600, flag: "wx" });
    await rename(temporary, path);
  } finally {
    await rm(temporary, { force: true });
  }
}

async function writePrivateJson(path: string, value: unknown): Promise<void> {
  await writePrivateAtomic(path, `${JSON.stringify(value, null, 2)}\n`);
}

function producer(reference: string): string {
  return reference.slice(0, reference.lastIndexOf("."));
}

function jsonl(rows: unknown[]): string {
  return rows.map((row) => JSON.stringify(row)).join("\n") + "\n";
}

/**
 * Execute the adapter-vocabulary pipeline (evaluate base, train, evaluate the
 * tuned model, compare) for a typed decision model. Training rows and held-out
 * rows come from one deterministic split of the spec examples; the Python
 * evaluator only ever sees held-out IDs and inputs.
 */
export async function runDecisionPipeline(args: {
  spec: LocalDecisionSpecFile;
  plan: ExecutionPlan;
  specPath: string;
  outputDir?: string;
  spawnStep?: DecisionStepSpawn;
  onProgress?: (message: string) => void;
}): Promise<DecisionPipelineResult> {
  assertDecisionPlanSupported(args.plan);
  const model = resolveDecisionModel(args.spec.base_model);
  const specDir = dirname(resolve(args.specPath));
  const outputDir = resolve(args.outputDir ?? (args.spec.runtime?.artifactRoot
    ? join(resolve(specDir, args.spec.runtime.artifactRoot), "decision-runs", randomUUID())
    : join(specDir, DEFAULT_DECISION_RUNS_DIR, randomUUID())));
  if (await lstat(outputDir).catch(() => null)) {
    throw new Error(`Decision output directory already exists: ${outputDir}`);
  }
  await mkdir(outputDir, { recursive: true, mode: 0o700 });

  const labels = decisionLabels(args.spec.decision);
  const question = decisionQuestion(args.spec);
  const examples = args.spec.examples.map((example) => ({
    input: example.input,
    output: canonicalLabel(labels, example.output),
  }));
  const seed = deriveSampleSeed(args.spec.id ?? args.spec.name);
  const split = splitSpecExamples(examples, seed);
  if (!split.train.length || !split.holdout.length) {
    throw new Error("Decision runs need at least one training and one held-out example.");
  }
  const hp = args.spec.hyperparameters ?? {};
  const device = hp.device ?? "auto";
  const evalIds = split.holdout.map((_, index) => `eval-${index}`);
  const dataDir = join(outputDir, "data");
  await mkdir(dataDir, { recursive: true, mode: 0o700 });
  const trainPath = join(dataDir, "train.jsonl");
  const evalInputsPath = join(dataDir, "eval-inputs.jsonl");
  await writePrivateAtomic(trainPath, jsonl(split.train));
  await writePrivateAtomic(evalInputsPath, jsonl(split.holdout.map((example, index) => ({ id: evalIds[index], input: example.input }))));
  await writePrivateJson(join(outputDir, "resolved-workflow.json"), {
    spec: args.spec,
    plan: args.plan,
    model: { id: model.id, revision: model.revision },
    question,
    split: { seed, train: split.train.length, eval: split.holdout.length },
  });

  const spawnStep = args.spawnStep ?? defaultDecisionSpawn;
  const producedModels = new Map<string, string>();
  const reports = new Map<string, { report: EvalReport; metrics: DecisionMetrics }>();
  const steps: DecisionStepResult[] = [];
  let tunedModelDir: string | undefined;

  for (const step of args.plan.steps) {
    const stepDir = join(outputDir, step.id);
    await mkdir(stepDir, { recursive: true, mode: 0o700 });
    const configPath = join(stepDir, "config.json");
    const logPath = join(stepDir, "step.log");
    const metricsPath = join(stepDir, "metrics.json");

    if (step.uses === "train") {
      const modelDir = join(stepDir, "model");
      args.onProgress?.(`${step.id}: fine-tuning ${model.id} on ${split.train.length} examples (log: ${logPath})`);
      await writePrivateJson(configPath, {
        model: model.id,
        revision: model.revision,
        question,
        train_path: trainPath,
        output_dir: modelDir,
        metrics_path: metricsPath,
        epochs: hp.n_epochs ?? model.defaultEpochs,
        learning_rate: hp.learning_rate ?? model.defaultLearningRate,
        batch_size: hp.batch_size ?? model.defaultBatchSize,
        freeze_encoder: hp.freeze_encoder ?? false,
        shuffle_options: hp.shuffle_options ?? true,
        seed: hp.seed ?? 0,
        device,
      });
      await spawnStep({ entrypoint: "train.py", configPath, logPath, stepId: step.id, onProgress: args.onProgress });
      const metrics = await readJsonObject(metricsPath, `Decision step "${step.id}" metrics`);
      if (metrics.ok !== true) throw new Error(`Decision step "${step.id}" metrics must report ok: true.`);
      await requireRegularNonemptyFile(join(modelDir, "model.safetensors"), `Decision step "${step.id}" model.safetensors`);
      await readJsonObject(join(modelDir, "rl_agent_config.json"), `Decision step "${step.id}" rl_agent_config.json`);
      producedModels.set(step.id, modelDir);
      tunedModelDir = modelDir;
      steps.push({ id: step.id, uses: step.uses, model: modelDir, training: metrics });
      continue;
    }

    if (step.uses === "evaluate") {
      const source = step.with.model;
      const tuned = source !== "base";
      const modelDir = tuned ? producedModels.get(producer(source.from)) : undefined;
      if (tuned && !modelDir) throw new Error(`Step ${step.id} needs a model from ${source.from}.`);
      const predictionsPath = join(stepDir, "predictions.jsonl");
      args.onProgress?.(`${step.id}: evaluating ${tuned ? "tuned" : "base"} model on ${split.holdout.length} held-out examples`);
      await writePrivateJson(configPath, {
        model: modelDir ?? model.id,
        ...(tuned ? {} : { revision: model.revision }),
        question,
        inputs_path: evalInputsPath,
        output_path: predictionsPath,
        metrics_path: metricsPath,
        device,
      });
      await spawnStep({ entrypoint: "evaluate.py", configPath, logPath, stepId: step.id, onProgress: args.onProgress });
      const metrics = await readJsonObject(metricsPath, `Decision step "${step.id}" metrics`);
      if (metrics.ok !== true) throw new Error(`Decision step "${step.id}" metrics must report ok: true.`);
      const reportPath = join(stepDir, "report.json");
      const scored = scoreDecisionPredictions({
        kind: tuned ? "candidate" : "baseline",
        modelId: modelDir ? fileUri(modelDir) : model.id,
        labels,
        examples: split.holdout,
        predictions: await readPredictions(predictionsPath, evalIds, labels),
        outputPath: reportPath,
        sampleSeed: seed,
      });
      await writePrivateJson(reportPath, { ...scored.report, decision_metrics: scored.metrics });
      reports.set(step.id, scored);
      steps.push({ id: step.id, uses: step.uses, model: modelDir ?? model.id, report: scored.report, decision_metrics: scored.metrics });
      continue;
    }

    if (step.uses !== "compare") throw new Error(`Decision runner does not implement ${step.uses} (step ${step.id}).`);
    const before = reports.get(producer(step.with.before.from));
    const after = reports.get(producer(step.with.after.from));
    if (!before || !after) throw new Error(`Step ${step.id} needs both evaluation reports.`);
    const comparison = {
      ...compareEvalReports(before.report, after.report),
      log_loss_delta: after.metrics.log_loss - before.metrics.log_loss,
      brier_delta: after.metrics.brier - before.metrics.brier,
    };
    await writePrivateJson(join(stepDir, "comparison.json"), comparison);
    steps.push({ id: step.id, uses: step.uses, comparison });
  }

  const reportPath = join(outputDir, "report.json");
  const report = {
    status: "succeeded" as const,
    engine: "decision" as const,
    ...(args.plan.name ? { name: args.plan.name } : {}),
    spec: resolve(args.specPath),
    base_model: model.id,
    base_model_revision: model.revision,
    question: { type: question.type, labels },
    split: { seed, train: split.train.length, eval: split.holdout.length, eval_split: "spec_holdout" },
    ...(tunedModelDir ? { model_dir: tunedModelDir } : {}),
    steps,
    created_at: new Date().toISOString(),
  };
  await writePrivateJson(reportPath, report);
  return {
    status: "succeeded",
    engine: "decision",
    ...(args.plan.name ? { name: args.plan.name } : {}),
    run_dir: outputDir,
    report_path: reportPath,
    ...(tunedModelDir ? { model_dir: tunedModelDir } : {}),
    steps,
  };
}
