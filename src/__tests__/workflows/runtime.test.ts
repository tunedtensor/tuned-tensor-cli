import { spawnSync } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getActiveModel } from "../../local-runtime/active-model.js";
import { fineTuneRunRequestSchema, localRunnerConfigSchema } from "../../local-runtime/contracts.js";
import { runLocalFineTune } from "../../local-runtime/orchestrator.js";
import { runLoggedProcess } from "../../local-runtime/process-runner.js";
import { createLocalStore } from "../../local-runtime/store.js";

// Keep orchestration, dataset splitting, scoring, artifact verification, the
// store, and CLI real. Only the costly Python model process is simulated.
vi.mock("../../local-runtime/process-runner.js", async (importOriginal) => ({
  ...await importOriginal<typeof import("../../local-runtime/process-runner.js")>(),
  runLoggedProcess: vi.fn(),
}));

const firstRunId = "11111111-1111-4111-8111-111111111111";
const secondRunId = "33333333-3333-4333-8333-333333333333";
const cliPath = fileURLToPath(new URL("../../local-runtime/index.ts", import.meta.url));
const tsxLoader = import.meta.resolve("tsx");

const behaviorExamples = [
  { input: "Classify: good", output: "positive" },
  { input: "Classify: excellent", output: "positive" },
  { input: "Classify: bad", output: "negative" },
  { input: "Classify: terrible", output: "negative" },
];
const generalExamples = [
  { input: "What is 2 + 2?", output: "4" },
  { input: "Return the word blue.", output: "blue" },
];
const answers = new Map([...behaviorExamples, ...generalExamples].map(({ input, output }) => [input, output]));

function request(runId = firstRunId) {
  return fineTuneRunRequestSchema.parse({
    run_id: runId,
    user_id: "local-user",
    behavior_spec_id: "22222222-2222-4222-8222-222222222222",
    run_number: runId === firstRunId ? 1 : 2,
    spec_snapshot: {
      name: "Sentiment labels",
      system_prompt: "Return the requested label only.",
      base_model: "Qwen/Qwen3.5-2B",
      examples: behaviorExamples,
    },
    hyperparameters: { n_epochs: 1 },
  });
}

async function json(path: string) {
  return JSON.parse(await readFile(path, "utf8"));
}

let root: string;
let regressGeneral: boolean;
let trainedPrompts: string[];
let trainedModelDir: string | undefined;

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), "tt-workflow-"));
  regressGeneral = false;
  trainedPrompts = [];
  trainedModelDir = undefined;
  vi.mocked(runLoggedProcess).mockReset();
  vi.mocked(runLoggedProcess).mockImplementation(async (args) => {
    expect(args.command).toBe("uv");
    expect(args.env?.HF_HUB_OFFLINE).toBe("1");
    expect(args.env?.TRANSFORMERS_OFFLINE).toBe("1");
    if (args.logPath) await writeFile(args.logPath, "Simulated model process for workflow testing.\n");

    if (args.stage === "training") {
      const dataset = await readFile(join(args.env!.SM_CHANNEL_TRAINING!, "training.jsonl"), "utf8");
      trainedPrompts = dataset.trim().split("\n").map((row) => JSON.parse(row).messages[1].content);
      const modelDir = args.env!.SM_MODEL_DIR!;
      trainedModelDir = modelDir;
      await mkdir(modelDir, { recursive: true });
      // These bytes exercise the artifact handoff; they are not usable weights
      // and this suite makes no claim about actual model quality or GPU execution.
      await writeFile(join(modelDir, "adapter_model.safetensors"), "test adapter bytes");
      await writeFile(join(modelDir, "adapter_config.json"), JSON.stringify({ peft_type: "LORA" }));
      await writeFile(join(modelDir, "training-metrics.json"), JSON.stringify({ loss: 0.1 }));
      return { exitCode: 0, stderr: "" };
    }

    expect(args.stage).toMatch(/^evaluating_(baseline|candidate)$/);
    const inputPath = args.commandArgs[args.commandArgs.indexOf("--input") + 1]!;
    const outputPath = args.commandArgs[args.commandArgs.indexOf("--output") + 1]!;
    const input = await json(inputPath);
    expect(input.protocol_version).toBe(2);
    expect(input.adapter_path).toBe(input.kind === "candidate" ? trainedModelDir : undefined);
    const results = input.examples.map((example: { id: string; input: string }) => {
      // The inference boundary receives prompts, never held-out answers.
      expect(Object.keys(example).sort()).toEqual(["id", "input"]);
      expect(answers.has(example.input)).toBe(true);
      const general = generalExamples.some((entry) => entry.input === example.input);
      const wrong = general ? regressGeneral && input.kind === "candidate" : input.kind === "baseline";
      return { id: example.id, actual: wrong ? "incorrect" : answers.get(example.input), latency_ms: 2 };
    });
    await writeFile(outputPath, JSON.stringify({ results }));
    return { exitCode: 0, stderr: "" };
  });
});

afterEach(async () => {
  await rm(root, { recursive: true, force: true });
});

async function setup(options: { dryRun?: boolean } = {}) {
  const generalPath = join(root, "general.jsonl");
  await writeFile(generalPath, generalExamples.map(({ input, output }) => JSON.stringify({
    messages: [{ role: "user", content: input }, { role: "assistant", content: output }],
  })).join("\n") + "\n");
  const config = localRunnerConfigSchema.parse({
    artifactRoot: join(root, "artifacts"),
    storeRoot: join(root, "store"),
    dryRun: options.dryRun ?? false,
    paths: { modelCache: join(root, "model-cache") },
    evaluation: {
      inference: { device: "cuda" },
      generalRegression: { dataset: generalPath, maxScoreDrop: 0, maxPassRateDrop: 0 },
    },
  });
  const configPath = join(root, "local-runner.json");
  await writeFile(configPath, JSON.stringify(config));
  const store = createLocalStore(config.storeRoot);

  function cli(...args: string[]) {
    const result = spawnSync(process.execPath, ["--import", tsxLoader, cliPath, ...args, "--config", configPath], {
      cwd: root,
      encoding: "utf8",
      timeout: 10_000,
      env: { ...process.env, TT_LOCAL_HOME: config.storeRoot, TUNED_TENSOR_HOME: join(root, "tt-home") },
    });
    expect(result.error).toBeUndefined();
    expect(result.signal).toBeNull();
    return result;
  }

  function cliJson(...args: string[]) {
    const result = cli(...args);
    expect(result.status, result.stderr).toBe(0);
    return JSON.parse(result.stdout);
  }

  return { config, store, cli, cliJson };
}

describe("local adapter workflow contract", () => {
  it("trains on separate data, assesses results, activates, resolves serving, and reuses verified work", async () => {
    const { config, store, cliJson } = await setup();
    const input = request();
    const result = await runLocalFineTune({ request: input, config });
    const modelId = `local-${input.run_id}`;
    const heldOut = result.report.baseline.results.map((entry) => entry.prompt);

    expect(trainedPrompts).toHaveLength(3);
    expect(heldOut).toHaveLength(1);
    expect(trainedPrompts).not.toContain(heldOut[0]);
    expect(result.report.candidate.results.map((entry) => entry.prompt)).toEqual(heldOut);
    expect(result.report).toMatchObject({
      status: "completed",
      baseline: { eval_split: "spec_holdout", avg_score: 0 },
      candidate: { eval_split: "spec_holdout", avg_score: 1 },
      comparison: { avg_score_delta: 1, improvements: 1, regressions: 0 },
      general_regression: { passed: true, baseline: { total: 2 }, candidate: { total: 2 } },
    });
    expect(await json(result.reportPath)).toEqual(result.report);
    expect(await store.getRunReport(input.run_id)).toEqual(result.report);
    expect(await store.getRun(input.run_id)).toMatchObject({ status: "completed", model_id: modelId });
    expect(cliJson("runs", "report", input.run_id)).toEqual(result.report);
    expect(cliJson("models", "verify", modelId)).toMatchObject({ ok: true, model: { id: modelId } });
    expect(cliJson("models", "activate", modelId)).toMatchObject({ active: modelId });
    expect(cliJson("serve", "active", "--print-command")).toMatchObject({
      ok: true,
      model_id: modelId,
      artifact_path: fileURLToPath(result.report.training.model_artifact_uri!),
      manifest_path: join(result.artifactDir, "artifact-manifest.json"),
      url: "http://127.0.0.1:8000",
    });

    // A repeated request should pay for neither training nor inference again.
    expect(vi.mocked(runLoggedProcess).mock.calls.filter(([args]) => args.stage === "training")).toHaveLength(1);
    expect(runLoggedProcess).toHaveBeenCalledTimes(5);
    const repeated = await runLocalFineTune({ request: input, config });
    expect(repeated.report.candidate).toEqual(result.report.candidate);
    expect(runLoggedProcess).toHaveBeenCalledTimes(5);
    expect(await store.listModels()).toHaveLength(1);
    expect(cliJson("models", "rollback")).toMatchObject({ active: "base" });
    expect((await getActiveModel(store)).model).toBeNull();
  }, 20_000);

  it("keeps the approved serving target when a better task score hides general regression", async () => {
    const { config, store, cli, cliJson } = await setup();
    await runLocalFineTune({ request: request(), config });
    const approvedId = `local-${firstRunId}`;
    cliJson("models", "activate", approvedId);

    regressGeneral = true;
    const regressed = await runLocalFineTune({ request: request(secondRunId), config });
    expect(regressed.report).toMatchObject({
      status: "completed",
      comparison: { avg_score_delta: 1 },
      general_regression: { passed: false, comparison: { avg_score_delta: -1, regressions: 2 } },
    });
    const activation = cli("models", "activate", `local-${secondRunId}`);
    expect(activation.status).toBe(1);
    expect(activation.stderr).toMatch(/failed general regression/i);
    expect((await getActiveModel(store)).model?.id).toBe(approvedId);
    expect(cliJson("serve", "active", "--print-command")).toMatchObject({ model_id: approvedId });
  }, 20_000);

  it("rejects changed adapter bytes and evaluation evidence before they can be trusted again", async () => {
    const { config, cli, cliJson } = await setup();
    const input = request();
    const result = await runLocalFineTune({ request: input, config });
    const modelId = `local-${input.run_id}`;
    cliJson("models", "activate", modelId);
    const weightsPath = join(fileURLToPath(result.report.training.model_artifact_uri!), "adapter_model.safetensors");
    const originalWeights = await readFile(weightsPath);
    const manifestPath = join(result.artifactDir, "artifact-manifest.json");
    const originalManifest = await readFile(manifestPath, "utf8");
    await writeFile(weightsPath, "changed adapter bytes");

    for (const args of [
      ["models", "verify", modelId],
      ["models", "activate", modelId],
      ["serve", "active", "--print-command"],
    ]) {
      const rejected = cli(...args);
      expect(rejected.status, args.join(" ")).toBe(1);
      expect(rejected.stderr).toMatch(/integrity verification failed/i);
    }
    const processCalls = vi.mocked(runLoggedProcess).mock.calls.length;
    await expect(runLocalFineTune({ request: input, config })).rejects.toThrow(/integrity verification failed/i);
    expect(runLoggedProcess).toHaveBeenCalledTimes(processCalls);
    expect(await readFile(manifestPath, "utf8")).toBe(originalManifest);

    await writeFile(weightsPath, originalWeights);
    const evaluationPath = fileURLToPath(result.report.artifact_uris.general_candidate_eval!);
    await writeFile(evaluationPath, JSON.stringify({ ...await json(evaluationPath), avg_score: 0 }));
    await expect(runLocalFineTune({ request: input, config })).rejects.toThrow(/integrity verification failed.*general-candidate-eval/i);
    expect(runLoggedProcess).toHaveBeenCalledTimes(processCalls);
    expect(await readFile(manifestPath, "utf8")).toBe(originalManifest);
  }, 20_000);

  it("records a dry-run rehearsal without registering an activatable or servable model", async () => {
    const { config, store, cli } = await setup({ dryRun: true });
    const input = request();
    const result = await runLocalFineTune({ request: input, config });
    expect(result.report).toMatchObject({ status: "completed", training: { metrics: { dry_run: true } } });
    expect(runLoggedProcess).not.toHaveBeenCalled();
    expect(await store.listModels()).toEqual([]);
    expect((await store.getRun(input.run_id)).model_id).toBeUndefined();

    for (const args of [
      ["models", "activate", `local-${input.run_id}`],
      ["serve", `local-${input.run_id}`, "--print-command"],
      ["serve", "active", "--print-command"],
    ]) {
      const rejected = cli(...args);
      expect(rejected.status, args.join(" ")).toBe(1);
      expect(rejected.stderr).toMatch(/model not found|no adapter is active/i);
    }
  }, 20_000);
});
