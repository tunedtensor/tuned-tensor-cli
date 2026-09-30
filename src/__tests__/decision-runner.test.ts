import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { canonicalFoundationPipeline, canonicalPipeline, createExecutionPlan, pipelineForRunInput } from "../pipeline.js";
import {
  decisionQuestion,
  runDecisionPipeline,
  scoreDecisionPredictions,
  type DecisionStepSpawn,
} from "../local-runtime/decision-runner.js";
import { LAYA_REVISION } from "../local-runtime/decision-models.js";
import { initLocalSpecFile, parseLocalRunInput, validateBehaviorSpec } from "../local-runtime/local-project.js";
import type { LocalDecisionSpecFile } from "../local-runtime/contracts.js";

const labels = ["billing", "technical", "account"];
const rawSpec = {
  engine: "decision",
  id: "7d0f7a2e-8d5c-4f0e-9a51-0b8d2f6c1e11",
  name: "Ticket router",
  system_prompt: "Which team should handle this support ticket?",
  guidelines: ["Refund and invoice questions go to billing."],
  base_model: "ConvaiInnovations/Laya",
  decision: {
    type: "choice",
    criteria: {
      billing: "invoices, payments, refunds",
      technical: "bugs, errors, outages",
      account: "login, password, profile",
    },
  },
  examples: Array.from({ length: 10 }, (_, index) => ({
    input: `ticket ${index}`,
    output: labels[index % 3]!.toUpperCase(),
  })),
  hyperparameters: { n_epochs: 2, device: "cpu" },
};

function decisionSpec(overrides: Record<string, unknown> = {}): LocalDecisionSpecFile {
  const input = parseLocalRunInput({ ...rawSpec, ...overrides }, "/work/tunedtensor.json");
  if (input.kind !== "decision-spec") throw new Error(`expected a decision spec, got ${input.kind}`);
  return input.spec;
}

/** Stand-in Python: the base model always answers billing; the tuned model is always right. */
function mockSpawn(seen: Array<{ entrypoint: string; config: Record<string, unknown>; inputs?: string }>, truth: Map<string, string>): DecisionStepSpawn {
  return async ({ entrypoint, configPath }) => {
    const config = JSON.parse(await readFile(configPath, "utf8")) as Record<string, string>;
    if (entrypoint === "train.py") {
      seen.push({ entrypoint, config });
      await mkdir(config.output_dir!, { recursive: true });
      await writeFile(join(config.output_dir!, "model.safetensors"), "weights");
      await writeFile(join(config.output_dir!, "rl_agent_config.json"), "{}\n");
      await writeFile(config.metrics_path!, JSON.stringify({ ok: true, steps: 4, final_loss: 0.1 }));
      return;
    }
    const inputs = await readFile(config.inputs_path!, "utf8");
    seen.push({ entrypoint, config, inputs });
    const tuned = !String(config.model).startsWith("convaiinnovations/");
    const rows = inputs.trim().split("\n").map((line) => JSON.parse(line) as { id: string; input: string });
    const predictions = rows.map((row) => {
      const label = tuned ? truth.get(row.input)! : "billing";
      const probabilities = Object.fromEntries(labels.map((item) => [item, item === label ? 0.8 : 0.1]));
      return JSON.stringify({ id: row.id, prediction: label, probabilities, confidence: 0.8, latency_ms: 3 });
    });
    await writeFile(config.output_path!, predictions.join("\n") + "\n");
    await writeFile(config.metrics_path!, JSON.stringify({ ok: true }));
  };
}

describe("decision specs", () => {
  it("parse as their own engine with a canonical, pinned model", () => {
    const spec = decisionSpec();
    expect(spec.base_model).toBe("convaiinnovations/laya");
    const input = parseLocalRunInput(rawSpec, "/work/tunedtensor.json");
    expect(validateBehaviorSpec(input)).toMatchObject({ valid: true, errors: [] });
    expect(decisionQuestion(spec)).toMatchObject({
      type: "choice",
      instructions: expect.stringContaining("Refund and invoice questions go to billing."),
    });
  });

  it("reuse the adapter pipeline vocabulary and reject foundation recipes", () => {
    const input = parseLocalRunInput(rawSpec, "/work/tunedtensor.json");
    expect(pipelineForRunInput(input)).toEqual(canonicalPipeline("local"));
    expect(() => pipelineForRunInput(input, canonicalFoundationPipeline())).toThrow(/foundation recipe, but the behavior spec is a decision spec/);
  });

  it("reject outputs that are not decision labels", () => {
    const input = parseLocalRunInput({
      ...rawSpec,
      examples: [...rawSpec.examples, { input: "ticket x", output: "sales" }],
    }, "/work/tunedtensor.json");
    const validation = validateBehaviorSpec(input);
    expect(validation.valid).toBe(false);
    expect(validation.errors.join("\n")).toMatch(/examples\[10\]\.output must be one of the decision labels: billing, technical, account/);
  });

  it("use true/false labels for yes/no and level indices for scores", () => {
    const noul = parseLocalRunInput({
      ...rawSpec,
      decision: { type: "noul" },
      examples: [{ input: "a", output: "true" }, { input: "b", output: "FALSE" }],
    }, "/work/tunedtensor.json");
    expect(validateBehaviorSpec(noul).errors).toEqual([]);
    const score = parseLocalRunInput({
      ...rawSpec,
      decision: { type: "score", criteria: ["low", "medium", "high"] },
      examples: [{ input: "a", output: "0" }, { input: "b", output: "3" }],
    }, "/work/tunedtensor.json");
    expect(validateBehaviorSpec(score).errors.join("\n")).toMatch(/examples\[1\]\.output must be one of the decision labels: 0, 1, 2/);
  });

  it("reject unsupported models and settings other engines own", () => {
    expect(() => parseLocalRunInput({ ...rawSpec, base_model: "Qwen/Qwen3.5-2B" }, "/work/tunedtensor.json")).toThrow(/Unsupported decision model/);
    expect(() => parseLocalRunInput({
      ...rawSpec,
      runtime: { gpu: { provider: "aws", instanceId: "i-0123456789abcdef0", user: "ubuntu" } },
    }, "/work/tunedtensor.json")).toThrow(/runtime.artifactRoot only/);
  });

  it("scaffold a spec that stays invalid until its placeholders are replaced", async () => {
    const dir = await mkdtemp(join(tmpdir(), "tt-decision-init-"));
    try {
      const path = join(dir, "tunedtensor.json");
      const spec = await initLocalSpecFile({ name: "Router", engine: "decision", outputPath: path });
      expect(spec).toMatchObject({ engine: "decision", base_model: "convaiinnovations/laya" });
      const validation = validateBehaviorSpec(parseLocalRunInput(JSON.parse(await readFile(path, "utf8")), path));
      expect(validation.valid).toBe(false);
      expect(validation.errors.join("\n")).toMatch(/decision\.criteria still contains generated placeholder text/);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});

describe("decision pipeline runner", () => {
  const dirs: string[] = [];
  afterEach(async () => {
    await Promise.all(dirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })));
  });

  it("trains on one split, evaluates held-out rows without labels, and compares base with tuned", async () => {
    const dir = await mkdtemp(join(tmpdir(), "tt-decision-"));
    dirs.push(dir);
    const spec = decisionSpec();
    const truth = new Map(spec.examples.map((example) => [example.input, example.output.toLowerCase()]));
    const seen: Array<{ entrypoint: string; config: Record<string, unknown>; inputs?: string }> = [];
    const result = await runDecisionPipeline({
      spec,
      plan: createExecutionPlan(pipelineForRunInput(parseLocalRunInput(rawSpec, join(dir, "tunedtensor.json")))),
      specPath: join(dir, "tunedtensor.json"),
      spawnStep: mockSpawn(seen, truth),
    });

    expect(seen.map((call) => call.entrypoint)).toEqual(["evaluate.py", "train.py", "evaluate.py"]);
    const [baseline, train, candidate] = seen;
    expect(baseline!.config).toMatchObject({ model: "convaiinnovations/laya", revision: LAYA_REVISION, device: "cpu" });
    expect(candidate!.config.model).toBe(join(result.run_dir, "train", "model"));
    expect(candidate!.config).not.toHaveProperty("revision");
    expect(train!.config).toMatchObject({ epochs: 2, batch_size: 8, learning_rate: 0.00002, shuffle_options: true });

    // Held-out rows reach Python as opaque IDs and inputs only, and never appear in training.
    const evalRows = baseline!.inputs!.trim().split("\n").map((line) => JSON.parse(line) as Record<string, string>);
    expect(evalRows.length).toBe(2);
    for (const row of evalRows) expect(Object.keys(row).sort()).toEqual(["id", "input"]);
    const trainRows = (await readFile(String(train!.config.train_path), "utf8")).trim().split("\n")
      .map((line) => JSON.parse(line) as { input: string; output: string });
    expect(trainRows.length).toBe(8);
    expect(trainRows.every((row) => labels.includes(row.output))).toBe(true);
    expect(trainRows.some((row) => evalRows.some((evalRow) => evalRow.input === row.input))).toBe(false);

    const [baseStep, , tunedStep, compare] = result.steps;
    expect(tunedStep!.decision_metrics!.accuracy).toBe(1);
    expect(compare!.comparison!.pass_rate_delta).toBeCloseTo(1 - baseStep!.decision_metrics!.accuracy);
    expect(compare!.comparison!.log_loss_delta).toBeLessThanOrEqual(0);
    expect(result.model_dir).toBe(join(result.run_dir, "train", "model"));

    const report = JSON.parse(await readFile(result.report_path, "utf8")) as Record<string, unknown>;
    expect(report).toMatchObject({
      engine: "decision",
      base_model: "convaiinnovations/laya",
      base_model_revision: LAYA_REVISION,
      question: { type: "choice", labels },
      split: { train: 8, eval: 2 },
    });
    expect((await stat(String(train!.config.train_path))).mode & 0o077).toBe(0);
  });

  it("refuses to reuse an existing run directory", async () => {
    const dir = await mkdtemp(join(tmpdir(), "tt-decision-"));
    dirs.push(dir);
    await expect(runDecisionPipeline({
      spec: decisionSpec(),
      plan: createExecutionPlan(canonicalPipeline("local")),
      specPath: join(dir, "tunedtensor.json"),
      outputDir: dir,
      spawnStep: async () => { throw new Error("must not spawn"); },
    })).rejects.toThrow(/already exists/);
  });

  it.each<{ probabilities?: Record<string, number>; prediction?: string; confidence?: number; latency_ms?: number }>([
    { probabilities: { no: -0.1, yes: 1.1 } },
    { probabilities: { no: 0.2, yes: 0.2 } },
    { probabilities: { yes: 1 } },
    { probabilities: { no: Number.NaN, yes: 1 } },
    { prediction: "no" },
    { confidence: 0.1 },
    { latency_ms: -1 },
  ])("rejects malformed model output: %j", (override) => {
    expect(() => scoreDecisionPredictions({
      kind: "baseline", modelId: "m", labels: ["no", "yes"],
      examples: [{ input: "a", output: "yes" }],
      predictions: [{ id: "0", prediction: "yes", probabilities: { no: 0.2, yes: 0.8 },
        confidence: 0.8, latency_ms: 1, ...override }],
      outputPath: "/tmp/report.json", sampleSeed: 1,
    })).toThrow(/Decision/);
  });

  it("rejects duplicate evaluation IDs even when every expected ID is present", async () => {
    const dir = await mkdtemp(join(tmpdir(), "tt-decision-"));
    dirs.push(dir);
    const spec = decisionSpec();
    const truth = new Map(spec.examples.map((example) => [example.input, example.output.toLowerCase()]));
    const spawn = mockSpawn([], truth);
    await expect(runDecisionPipeline({
      spec, plan: createExecutionPlan(canonicalPipeline("local")), specPath: join(dir, "tunedtensor.json"),
      spawnStep: async (args) => {
        await spawn(args);
        if (args.entrypoint === "evaluate.py") {
          const config = JSON.parse(await readFile(args.configPath, "utf8"));
          const output = await readFile(config.output_path, "utf8");
          await writeFile(config.output_path, output + output.split("\n")[0] + "\n");
        }
      },
    })).rejects.toThrow(/exactly one row per evaluation ID/);
  });

  it("scores log-loss and Brier against the expected label", () => {
    const { report, metrics } = scoreDecisionPredictions({
      kind: "baseline",
      modelId: "m",
      labels: ["no", "yes"],
      examples: [{ input: "a", output: "yes" }, { input: "b", output: "no" }],
      predictions: [
        { id: "0", prediction: "yes", probabilities: { no: 0.25, yes: 0.75 }, confidence: 0.75, latency_ms: 1 },
        { id: "1", prediction: "yes", probabilities: { no: 0.4, yes: 0.6 }, confidence: 0.6, latency_ms: 1 },
      ],
      outputPath: "/tmp/report.json",
      sampleSeed: 1,
    });
    expect(report.pass_rate).toBe(0.5);
    expect(metrics.log_loss).toBeCloseTo((-Math.log(0.75) - Math.log(0.4)) / 2);
    expect(metrics.brier).toBeCloseTo(((0.25 ** 2 + 0.25 ** 2) + (0.6 ** 2 + 0.6 ** 2)) / 2);
    expect(metrics.per_label).toEqual({
      no: { support: 1, correct: 0, recall: 0 },
      yes: { support: 1, correct: 1, recall: 1 },
    });
  });
});
