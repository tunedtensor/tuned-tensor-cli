import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { localRunnerConfigSchema, fineTuneRunRequestSchema } from "../local-runtime/contracts.js";
import { resolveRunArtifacts } from "../local-runtime/artifacts.js";
import { launchProcessTraining } from "../local-runtime/process-training.js";
import { defaultFoundationSpawn } from "../local-runtime/foundation-runner.js";
import { runGpuProcess } from "../local-runtime/gpu-executor.js";
import { evaluateExamples, INFERENCE_PROTOCOL_VERSION } from "../local-runtime/evaluation.js";
import { QWEN_3_5_2B_REVISION } from "../local-runtime/model-registry.js";
vi.mock("../local-runtime/gpu-executor.js", async (original) => ({
  ...await original<typeof import("../local-runtime/gpu-executor.js")>(), runGpuProcess: vi.fn(),
}));
let root: string;
const request = fineTuneRunRequestSchema.parse({ run_id: "11111111-1111-4111-8111-111111111111", user_id: "local-user",
  behavior_spec_id: "22222222-2222-4222-8222-222222222222", run_number: 1,
  spec_snapshot: { name: "Classifier", base_model: "Qwen/Qwen3.5-2B", examples: [{ input: "hello", output: "greeting" }] } });
const gpu = { provider: "aws", instanceId: "i-1234567890abcdef0", user: "ubuntu" };
beforeEach(async () => { root = await mkdtemp(join(tmpdir(), "tt-stages-")); vi.clearAllMocks(); });
afterEach(async () => { await rm(root, { recursive: true, force: true }); });

it.each([false, true])("adapter training preserves local artifacts and dryRun=%s", async (dryRun) => {
  const config = localRunnerConfigSchema.parse({ artifactRoot: root, dryRun, gpu });
  const artifacts = resolveRunArtifacts({ artifactRoot: root, prefix: "run" });
  await mkdir(artifacts.trainingInputDir, { recursive: true });
  await mkdir(artifacts.trainingConfigDir, { recursive: true });
  await mkdir(artifacts.trainingModelDir, { recursive: true });
  await writeFile(artifacts.trainingJsonl, '{"messages":[]}\n');
  vi.mocked(runGpuProcess).mockImplementation(async () => {
    await writeFile(join(artifacts.trainingModelDir, "training-metrics.json"), '{"loss":0.5}');
    return { exitCode: 0, stderr: "" };
  });
  const report = await launchProcessTraining({ request, config, artifacts, baseModelRevision: QWEN_3_5_2B_REVISION });
  expect(report.provider).toBe(dryRun ? "local-uv" : "aws-ssh");
  expect(report.model_artifact_uri).toBe(`file://${artifacts.trainingModelDir}`);
  expect(runGpuProcess).toHaveBeenCalledTimes(dryRun ? 0 : 1);
});

it("runs inference remotely but scores predictions locally", async () => {
  const config = localRunnerConfigSchema.parse({ artifactRoot: root, gpu, evaluation: { baselineCache: false } });
  vi.mocked(runGpuProcess).mockImplementation(async (args) => {
    const output = args.files.find((file) => file.direction === "output")!;
    await writeFile(output.path, JSON.stringify({ protocol_version: INFERENCE_PROTOCOL_VERSION, results: [{ id: "0", actual: "greeting", latency_ms: 10 }] }));
    return { exitCode: 0, stderr: "" };
  });
  const report = await evaluateExamples({ kind: "baseline", modelId: "Qwen/Qwen3.5-2B", baseModelRevision: QWEN_3_5_2B_REVISION,
    examples: [{ input: "hello", output: "greeting" }], system: "Classify", config, outputPath: join(root, "eval.json") });
  expect(report.avg_score).toBe(1);
  expect(report.scoring_method).toBe("exact_match");
  expect(runGpuProcess).toHaveBeenCalledOnce();
  expect(JSON.parse(await readFile(join(root, "eval.json"), "utf8")).pass_rate).toBe(1);
});

it("foundation transfers corpus directories and checkpoints with the existing shutdown grace", async () => {
  const config = localRunnerConfigSchema.parse({ gpu });
  const corpus = join(root, "corpus");
  await mkdir(corpus);
  const path = join(root, "config.json");
  await writeFile(path, JSON.stringify({ corpus_path: corpus, output_dir: join(root, "output"), work_dir: join(root, "recovery") }));
  vi.mocked(runGpuProcess).mockResolvedValue({ exitCode: 0, stderr: "" });
  await defaultFoundationSpawn({ entrypoint: "pretrain.py", configPath: path, logPath: join(root, "log"), stepId: "pretrain", gpu: config.gpu });
  expect(runGpuProcess).toHaveBeenCalledWith(expect.objectContaining({ shutdownGraceMs: 120_000,
    files: expect.arrayContaining([{ path: corpus, direction: "input", directory: true }, { path: join(root, "recovery"), direction: "both", directory: true }]) }));
});
