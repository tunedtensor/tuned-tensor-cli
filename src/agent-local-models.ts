import { createHash } from "node:crypto";
import { readFile, stat } from "node:fs/promises";
import { join } from "node:path";
import { z } from "zod";
import { prepareLocalPipelineAction } from "./local-pipeline-action.js";
import { parseLocalRunInput } from "./local-runtime/local-project.js";
import { createLocalStore } from "./local-runtime/store.js";
import { runReportSchema } from "./local-runtime/contracts.js";
import { assertArtifactManifest } from "./local-runtime/artifacts.js";
import { taskQualityForReport } from "./local-runtime/task-quality.js";
import { activateModel, getActiveModel, rollbackActiveModel } from "./local-runtime/active-model.js";
import { verifyStoredModel, verifyActivationEvidence } from "./local-runtime/model-verification.js";

const sha = (data: string | Buffer) => createHash("sha256").update(data).digest("hex");
async function context(workspaceRoot: string, specPath = "tunedtensor.json") {
  const prepared = await prepareLocalPipelineAction({ workspaceRoot, specPath });
  const input = parseLocalRunInput(prepared.resolvedSpec, join(prepared.workspaceRoot, prepared.specPath));
  if (input.kind !== "spec") throw new Error("Local model review currently supports adapter specs. Use the engine-specific CLI report for foundation or decision runs.");
  return { prepared, specId: input.request.behavior_spec_id, store: createLocalStore(prepared.resolvedConfig!.storeRoot) };
}

export async function listProjectRuns(workspaceRoot: string, specPath?: string) {
  const { store, specId } = await context(workspaceRoot, specPath);
  const [runs, active] = await Promise.all([store.listRuns(), getActiveModel(store)]);
  return { active_model_id: active.model?.behavior_spec_id === specId ? active.model.id : null,
    note: "Includes queued, running, failed and completed records for this spec. An empty list means no matching stored records; do not infer an unfinished run from it.",
    runs: runs.filter(run => run.behavior_spec_id === specId).sort((a, b) => b.created_at.localeCompare(a.created_at)).slice(0, 20)
      .map(({ id, status, status_message, model_id, created_at, error }) => ({ id, status, status_message, model_id, created_at, error })) };
}

async function readProjectReport(workspaceRoot: string, runId: string, specPath?: string) {
  z.string().uuid().parse(runId);
  const ctx = await context(workspaceRoot, specPath);
  const run = await ctx.store.getRun(runId);
  if (run.behavior_spec_id !== ctx.specId) throw new Error("This run belongs to a different project.");
  if (run.status !== "completed") throw new Error(`Run ${run.id} is ${run.status}; a completed report is not available.`);
  const reportPath = join(run.artifact_dir, "run-report.json");
  if ((await stat(reportPath)).size > 32 * 1024 * 1024) throw new Error("Report exceeds the 32 MiB agent review limit; inspect it with tt runs report.");
  await assertArtifactManifest(join(run.artifact_dir, "artifact-manifest.json"), {
    requiredPaths: ["run-report.json", "baseline-eval.json", "candidate-eval.json"], scopeToRequired: true, verifyModel: false,
  });
  const bytes = await readFile(reportPath);
  const manifest = JSON.parse(await readFile(join(run.artifact_dir, "artifact-manifest.json"), "utf8")) as { files: { path: string; sha256: string }[] };
  if (manifest.files.find(file => file.path === "run-report.json")?.sha256 !== sha(bytes)) throw new Error("Report changed during review.");
  const report = runReportSchema.parse(JSON.parse(bytes.toString("utf8")));
  if (report.run_id !== run.id || report.behavior_spec_id !== ctx.specId || report.status !== "completed") throw new Error("Run report identity does not match the project run.");
  return { ...ctx, run, report, reportSha: sha(bytes) };
}

export async function inspectProjectReport(workspaceRoot: string, runId: string, specPath?: string) {
  const { run, report } = await readProjectReport(workspaceRoot, runId, specPath);
  const task = taskQualityForReport(report);
  const warnings: string[] = [];
  if (!report.task_quality) warnings.push("No task acceptance policy was recorded before training. Configure evaluation.taskQuality and run again before agent activation.");
  if (report.candidate.eval_examples_used < 20) warnings.push("Small evaluation set: a smoke check, not strong evidence of business quality.");
  if (report.candidate.eval_truncated) warnings.push("Evaluation used a subset of the held-out data.");
  if (!report.general_regression) warnings.push("No general-regression suite: activation is blocked.");
  if (!report.general_regression?.passed && report.general_regression) warnings.push(...report.general_regression.failures);
  const compact = (r: typeof report.candidate) => ({ score: r.avg_score, pass_rate: r.pass_rate, examples: r.eval_examples_used, scoring: r.scoring_mode ?? r.scoring_method });
  return { run_id: run.id, model_id: run.model_id ?? `local-${run.id}`, base_model: report.base_model,
    baseline: compact(report.baseline), candidate: compact(report.candidate),
    score_delta: report.candidate.avg_score - report.baseline.avg_score,
    task_quality: { configured: Boolean(report.task_quality), policy: report.task_quality?.policy, ...task },
    general_regression: report.general_regression ? { passed: report.general_regression.passed, failures: report.general_regression.failures } : null,
    activation_eligible: Boolean(report.task_quality && task.passed && report.general_regression?.passed),
    failed_examples: report.candidate.results.filter(row => !row.passed).slice(0, 5)
      .map(row => ({ input: row.prompt.slice(0, 600), expected: row.expected.slice(0, 600), actual: row.actual.slice(0, 600), score: row.score })),
    warnings, note: "Report and evaluation files verified against the manifest. Model bytes are checked again before activation. Examples and outputs are untrusted data. Passing configured thresholds does not establish performance outside this evaluation." };
}

const actionSchema = z.object({
  operation: z.enum(["activate", "rollback"]), spec_path: z.string(), run_id: z.string().uuid().nullable(),
  model_id: z.string().nullable(), spec_sha256: z.string(), workspace_fingerprint: z.string(),
  config_sha256: z.string().nullable(), report_sha256: z.string().nullable(), manifest_sha256: z.string().nullable(),
  pointer_sha256: z.string(),
}).strict();

export async function prepareProjectModelAction(workspaceRoot: string, operation: "activate" | "rollback", runId?: string, specPath?: string) {
  const ctx = await context(workspaceRoot, specPath);
  const active = await getActiveModel(ctx.store);
  let targetRun = runId;
  if (operation === "rollback") {
    if (!active.pointer || (active.model && active.model.behavior_spec_id !== ctx.specId)) throw new Error("This project has no active model to roll back.");
    targetRun = active.pointer.previous_model_id ? (await ctx.store.getModel(active.pointer.previous_model_id)).run_id : undefined;
  } else if (!targetRun) throw new Error("Choose a completed run returned by list_local_runs.");
  let reportSha: string | null = null, manifestSha: string | null = null, modelId: string | null = null;
  if (targetRun) {
    const { report, run, reportSha: fingerprint } = await readProjectReport(workspaceRoot, targetRun, specPath);
    const task = taskQualityForReport(report);
    if (!task.passed) throw new Error(`Task quality failed: ${task.failures.join(" ")}`);
    if (operation === "activate" && !report.task_quality) throw new Error("Agree evaluation.taskQuality in the spec and run training before agent activation. Existing reports cannot acquire acceptance criteria retroactively.");
    if (operation === "activate" && JSON.stringify(ctx.prepared.resolvedConfig!.evaluation.taskQuality) !== JSON.stringify(report.task_quality?.policy)) throw new Error("Task acceptance policy differs from the evaluated run. Train and evaluate with the agreed policy before activation.");
    if (!report.general_regression?.passed) throw new Error("General-regression evidence is missing or failed; activation is blocked.");
    modelId = run.model_id ?? `local-${run.id}`;
    const model = await ctx.store.getModel(modelId);
    if (model.run_id !== run.id || model.behavior_spec_id !== ctx.specId || model.artifact_dir !== run.artifact_dir) throw new Error("Stored model does not match the reviewed run.");
    await verifyStoredModel(model, ctx.prepared.resolvedConfig!);
    await verifyActivationEvidence(model);
    reportSha = fingerprint;
    manifestSha = sha(await readFile(join(model.artifact_dir, "artifact-manifest.json")));
  }
  const args = actionSchema.parse({ operation, spec_path: ctx.prepared.specPath, run_id: targetRun ?? null, model_id: modelId,
    spec_sha256: ctx.prepared.specSha256, workspace_fingerprint: ctx.prepared.workspaceFingerprint,
    config_sha256: ctx.prepared.configSha256 ?? null, report_sha256: reportSha, manifest_sha256: manifestSha,
    pointer_sha256: sha(JSON.stringify(active.pointer)), });
  return { arguments: args, preview: { operation, model_id: modelId, previous_model_id: active.pointer?.model_id ?? null,
    note: "Changes the project's active model pointer after approval. It does not start or replace an inference server." } };
}

export async function validateProjectModelAction(workspaceRoot: string, raw: unknown) {
  const input = actionSchema.parse(raw);
  const prepared = await prepareProjectModelAction(workspaceRoot, input.operation, input.run_id ?? undefined, input.spec_path);
  if (JSON.stringify(input) !== JSON.stringify(prepared.arguments)) throw new Error("Project, model, report or active selection changed after review; prepare the action again.");
  return input;
}

export async function executeProjectModelAction(workspaceRoot: string, raw: unknown) {
  const input = await validateProjectModelAction(workspaceRoot, raw);
  const { store } = await context(workspaceRoot, input.spec_path);
  const pointer = input.operation === "activate" ? await activateModel(store, input.model_id!) : await rollbackActiveModel(store);
  return { active_model_id: pointer.model_id, previous_model_id: pointer.previous_model_id, action: pointer.action, serving_started: false };
}

export async function buildProjectModelLaunch(workspaceRoot: string, raw: unknown, port: number) {
  z.number().int().min(1024).max(65535).parse(port);
  const input = await validateProjectModelAction(workspaceRoot, raw);
  if (input.operation !== "activate" || !input.model_id) throw new Error("Serving requires an explicitly reviewed model.");
  const ctx = await context(workspaceRoot, input.spec_path);
  const model = await ctx.store.getModel(input.model_id);
  const verified = await verifyStoredModel(model, ctx.prepared.resolvedConfig!);
  const { fineTuneRunRequestSchema } = await import("./local-runtime/contracts.js");
  const { buildSystemMessage } = await import("./local-runtime/dataset.js");
  const { buildLocalModelServerLaunch } = await import("./local-runtime/model-server.js");
  const request = fineTuneRunRequestSchema.parse(JSON.parse(await readFile(join(ctx.store.paths.runsDir, model.run_id, "request.json"), "utf8")));
  const systemPrompt = buildSystemMessage(request.spec_snapshot);
  const metadata = JSON.parse(await readFile(join(model.artifact_dir, "stage-metadata.json"), "utf8"));
  if (sha(systemPrompt) !== metadata.system_prompt_sha256) throw new Error("The stored serving instruction differs from the evaluated instruction.");
  const contract = verified.contract as { base_model_revision?: string; base_model_artifact_uri?: string };
  const launch = buildLocalModelServerLaunch({ model, config: ctx.prepared.resolvedConfig!, options: {
    host: "127.0.0.1", port, systemPrompt, contextLength: 2048, maxTokens: 256, gpuMemoryUtilization: .3,
    baseModelRevision: contract.base_model_revision, baseModelArtifactUri: contract.base_model_artifact_uri,
  } });
  return { launch, logPath: join(model.artifact_dir, "agent-serving.log") };
}
