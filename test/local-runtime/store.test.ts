import assert from "node:assert/strict";
import { chmod, mkdir, mkdtemp, readFile, rm, stat, symlink, unlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fineTuneRunRequestSchema, runReportSchema } from "../../src/local-runtime/contracts.js";
import { createLocalStore, isTerminalRunState } from "../../src/local-runtime/store.js";

const runId = "33333333-3333-4333-8333-333333333333";
const specId = "44444444-4444-4444-8444-444444444444";

function requestFixture() {
  return fineTuneRunRequestSchema.parse({
    run_id: runId,
    user_id: "local-user",
    behavior_spec_id: specId,
    run_number: 1,
    spec_snapshot: {
      name: "Local Store Spec",
      description: "",
      system_prompt: "Return labels.",
      guidelines: [],
      constraints: [],
      base_model: "Qwen/Qwen3.5-2B",
      examples: [{ input: "Classify: good", output: "positive" }],
    },
    hyperparameters: {
      n_epochs: 1,
    },
  });
}

function reportFixture(reportPath: string) {
  const evalReport = {
    kind: "baseline",
    model_id: "Qwen/Qwen3.5-2B",
    total: 1,
    eval_examples_total: 1,
    eval_examples_used: 1,
    eval_truncated: false,
    avg_score: 0,
    pass_rate: 0,
    exact_match_rate: 0,
    avg_latency_ms: 0,
    results: [{
      prompt: "Classify: good",
      expected: "positive",
      actual: "",
      passed: false,
      score: 0,
      reasoning: "test",
      latency_ms: 0,
    }],
    artifact_uri: `file://${reportPath}`,
    scoring_method: "heuristic",
  };
  return runReportSchema.parse({
    run_id: runId,
    behavior_spec_id: specId,
    user_id: "local-user",
    run_number: 1,
    base_model: "Qwen/Qwen3.5-2B",
    fine_tuned_model_id: `file://${reportPath}`,
    status: "completed",
    baseline: evalReport,
    candidate: { ...evalReport, kind: "candidate", model_id: `file://${reportPath}` },
    comparison: {
      avg_score_delta: 0,
      pass_rate_delta: 0,
      exact_match_rate_delta: 0,
      regressions: 0,
      improvements: 0,
      regressed_examples: [],
    },
    training: {
      provider: "local-uv",
      training_job_name: "test-job",
      model_artifact_uri: `file://${reportPath}`,
      metrics: { loss: 0.1 },
      exit_code: 0,
      log_uri: `file://${reportPath}`,
    },
    artifact_uris: {
      dataset: `file://${reportPath}`,
      baseline_eval: `file://${reportPath}`,
      candidate_eval: `file://${reportPath}`,
      report: `file://${reportPath}`,
    },
    run_metadata: {
      base_model: "Qwen/Qwen3.5-2B",
      fine_tuned_model_id: `file://${reportPath}`,
      dataset_prebuilt: false,
      dataset_uri: `file://${reportPath}`,
      spec_example_count: 1,
      training_example_count: 1,
      eval_examples_total: 1,
      eval_examples_used: 1,
      started_at: "2026-01-01T00:00:00.000Z",
      completed_at: "2026-01-01T00:00:01.000Z",
      elapsed_ms: 1000,
      elapsed_seconds: 1,
    },
    created_at: "2026-01-01T00:00:01.000Z",
  });
}

async function exists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

test("local store persists runs, events, reports, specs, and model records", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-store-test-"));
  try {
    const store = createLocalStore(join(root, "store"));
    const artifactDir = join(root, "artifacts", runId);
    const request = requestFixture();
    const reportPath = join(artifactDir, "run-report.json");
    await store.startRun({ request, artifactDir });
    await store.updateRun({
      runId,
      status: "training",
      stage: "training",
      message: "Training.",
      details: { dry_run: true },
    });

    const report = reportFixture(reportPath);
    await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
    await store.completeRun(report, artifactDir, reportPath);

    const runs = await store.listRuns();
    assert.equal(runs[0]?.status, "completed");
    assert.equal((await store.getRun(runId.slice(0, 8))).id, runId);
    assert.equal((await store.getRunEvents(runId)).length, 4);
    assert.equal((await store.getRunReport(runId)).run_id, runId);
    assert.equal((await store.listModels())[0]?.run_id, runId);
    assert.equal((await store.getModel(`local-${runId.slice(0, 8)}`)).run_id, runId);
    assert.equal((await store.getSpec(specId.slice(0, 8))).spec.name, "Local Store Spec");
    assert.equal(await exists(join(store.root, "catalog")), false);

    assert.equal((await store.listRuns())[0]?.id, runId);
    assert.match(await readFile(join(artifactDir, "progress.jsonl"), "utf8"), /Training/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("local store keeps shared state directories and files private", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-store-permissions-test-"));
  try {
    const sharedRoot = join(root, ".tuned-tensor");
    const store = createLocalStore(join(sharedRoot, "store"));
    await store.startRun({
      request: requestFixture(),
      artifactDir: join(root, "artifacts", runId),
    });
    await store.cancelRun(runId);

    if (process.platform !== "win32") {
      assert.equal((await stat(sharedRoot)).mode & 0o777, 0o700);
      assert.equal((await stat(store.root)).mode & 0o777, 0o700);
      assert.equal((await stat(join(store.root, "runs"))).mode & 0o777, 0o700);
      assert.equal((await stat(join(store.root, "runs", runId, "state.json"))).mode & 0o777, 0o600);
      assert.equal((await stat(join(store.root, "runs", runId, "progress.jsonl"))).mode & 0o777, 0o600);
      assert.equal((await stat(join(store.root, "runs", runId, "cancel.requested"))).mode & 0o777, 0o600);
      assert.equal((await stat(join(store.root, "specs", specId, "spec.json"))).mode & 0o777, 0o600);
    }
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("local store repairs preserved state permissions before reading it", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-store-repair-permissions-test-"));
  try {
    const store = createLocalStore(join(root, ".tuned-tensor-local"));
    await store.startRun({
      request: requestFixture(),
      artifactDir: join(root, "artifacts", runId),
    });
    const directories = [
      store.root,
      join(store.root, "runs"),
      join(store.root, "runs", runId),
      join(store.root, "specs", specId),
    ];
    const files = [
      join(store.root, "runs", runId, "state.json"),
      join(store.root, "runs", runId, "progress.jsonl"),
      join(store.root, "specs", specId, "spec.json"),
    ];
    await Promise.all(directories.map((path) => chmod(path, 0o775)));
    await Promise.all(files.map((path) => chmod(path, 0o664)));

    await createLocalStore(store.root).getRun(runId);

    if (process.platform !== "win32") {
      for (const path of directories) assert.equal((await stat(path)).mode & 0o777, 0o700);
      for (const path of files) assert.equal((await stat(path)).mode & 0o777, 0o600);
    }
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("local store refuses preserved symbolic links instead of following them", {
  skip: process.platform === "win32",
}, async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-store-symlink-test-"));
  try {
    const store = createLocalStore(join(root, ".tuned-tensor-local"));
    await store.startRun({
      request: requestFixture(),
      artifactDir: join(root, "artifacts", runId),
    });
    const statePath = join(store.root, "runs", runId, "state.json");
    const externalPath = join(root, "external-state.json");
    await writeFile(externalPath, await readFile(statePath));
    await unlink(statePath);
    await symlink(externalPath, statePath);

    await assert.rejects(
      createLocalStore(store.root).getRun(runId),
      /symbolic link/i,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("completeRun repairs the store before reading a preserved request", {
  skip: process.platform === "win32",
}, async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-store-complete-symlink-test-"));
  try {
    const store = createLocalStore(join(root, ".tuned-tensor-local"));
    const artifactDir = join(root, "artifacts", runId);
    await store.startRun({ request: requestFixture(), artifactDir });
    const requestPath = join(store.root, "runs", runId, "request.json");
    const externalPath = join(root, "external-request.json");
    await writeFile(externalPath, "not json\n");
    await unlink(requestPath);
    await symlink(externalPath, requestPath);

    await assert.rejects(
      createLocalStore(store.root).completeRun(
        reportFixture(join(artifactDir, "run-report.json")),
        artifactDir,
        join(artifactDir, "run-report.json"),
      ),
      /symbolic link/i,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("completeRun leaves a run nonterminal when storing its report fails", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-store-complete-report-"));
  try {
    const store = createLocalStore(join(root, "store"));
    const artifactDir = join(root, "artifacts", runId);
    await store.startRun({ request: requestFixture(), artifactDir });
    const reportDirectory = join(root, "report-directory");
    await mkdir(reportDirectory);

    await assert.rejects(
      store.completeRun(reportFixture(reportDirectory), artifactDir, reportDirectory),
    );
    assert.notEqual((await store.getRun(runId)).status, "completed");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("completeRun leaves a run nonterminal when its report source is missing", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-store-complete-missing-report-"));
  try {
    const store = createLocalStore(join(root, "store"));
    const artifactDir = join(root, "artifacts", runId);
    const reportPath = join(artifactDir, "missing-report.json");
    await store.startRun({ request: requestFixture(), artifactDir });

    await assert.rejects(
      store.completeRun(reportFixture(reportPath), artifactDir, reportPath),
      /report.*not found/i,
    );
    assert.notEqual((await store.getRun(runId)).status, "completed");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("cancellation preservation removes partially persisted model and report state", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-store-cancel-cleanup-"));
  try {
    const store = createLocalStore(join(root, "store"));
    const artifactDir = join(root, "artifacts", runId);
    await store.startRun({ request: requestFixture(), artifactDir });
    const modelDir = join(store.paths.modelsDir, `local-${runId}`);
    const storedReport = join(store.paths.runsDir, runId, "run-report.json");
    await mkdir(modelDir, { recursive: true });
    await writeFile(join(modelDir, "model.json"), "{}\n");
    await writeFile(storedReport, "{}\n");
    await store.cancelRun(runId);

    await store.completeRun(
      reportFixture(join(artifactDir, "run-report.json")),
      artifactDir,
      join(artifactDir, "run-report.json"),
    );
    await assert.rejects(stat(modelDir));
    await assert.rejects(stat(storedReport));
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("local store scans canonical files without a metadata index", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-store-rebuild-test-"));
  try {
    const store = createLocalStore(join(root, "store"));
    const artifactDir = join(root, "artifacts", runId);
    const request = requestFixture();
    const reportPath = join(artifactDir, "run-report.json");
    await store.startRun({ request, artifactDir });
    await store.updateRun({
      runId,
      status: "training",
      stage: "training",
      message: "Training.",
      details: { dry_run: true },
    });

    const report = reportFixture(reportPath);
    await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
    await store.completeRun(report, artifactDir, reportPath);

    const reopened = createLocalStore(store.root);
    assert.equal((await reopened.listRuns())[0]?.id, runId);
    assert.equal((await reopened.getRunEvents(runId)).length, 4);
    assert.equal((await reopened.listSpecs())[0]?.id, specId);
    assert.equal((await reopened.listModels())[0]?.id, `local-${runId}`);

    assert.equal((await reopened.listRuns())[0]?.id, runId);
    assert.equal((await reopened.getRunEvents(runId)).length, 4);
    assert.equal(await exists(join(reopened.root, "catalog")), false);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("dry-run completion does not create a model record", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-store-dry-model-"));
  try {
    const store = createLocalStore(join(root, "store"));
    const request = requestFixture();
    const artifactDir = join(root, "artifacts", runId);
    const reportPath = join(artifactDir, "run-report.json");
    await store.startRun({ request, artifactDir });
    const real = reportFixture(reportPath);
    const report = runReportSchema.parse({
      ...real,
      training: { ...real.training, metrics: { dry_run: true } },
    });
    await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
    const state = await store.completeRun(report, artifactDir, reportPath);
    assert.equal(state.model_id, undefined);
    assert.equal((await store.listModels()).length, 0);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("resuming work clears stale failure and completion fields", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-store-resume-"));
  try {
    const store = createLocalStore(join(root, "store"));
    const request = requestFixture();
    const artifactDir = join(root, "artifacts", runId);
    await store.startRun({ request, artifactDir });
    const failed = await store.failRun(runId, "old failure");
    assert.equal(failed.error, "old failure");
    assert.ok(failed.completed_at);

    const resumed = await store.updateRun({
      runId,
      status: "preparing",
      stage: "preparing",
      message: "Resuming.",
    });
    assert.equal(resumed.error, undefined);
    assert.equal(resumed.completed_at, undefined);
    const persisted = await store.getRun(runId);
    assert.equal(persisted.error, undefined);
    assert.equal(persisted.completed_at, undefined);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("a cancellation marker wins over late progress, failure, and completion writes", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-store-cancel-race-"));
  try {
    const store = createLocalStore(join(root, "store"));
    const request = requestFixture();
    const artifactDir = join(root, "artifacts", runId);
    const reportPath = join(artifactDir, "run-report.json");
    await store.startRun({ request, artifactDir });
    await store.cancelRun(runId);

    const progressed = await store.updateRun({
      runId,
      status: "training",
      stage: "training",
      message: "Late training update.",
    });
    assert.equal(progressed.status, "cancelled");
    assert.equal(progressed.current_stage, "cancel_requested");
    assert.equal(isTerminalRunState(progressed), false);

    const failed = await store.failRun(runId, "late failure");
    assert.equal(failed.status, "cancelled");
    assert.equal(failed.error, undefined);

    const report = reportFixture(reportPath);
    await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
    const completed = await store.completeRun(report, artifactDir, reportPath);
    assert.equal(completed.status, "cancelled");
    assert.equal(completed.current_stage, "cancel_requested");
    assert.equal((await store.listModels()).length, 0);

    const finalized = await store.finalizeCancellation(runId);
    assert.equal(finalized.current_stage, "cancelled");
    assert.equal(isTerminalRunState(finalized), true);
    await store.cancelRun(runId);
    const unchanged = await store.getRun(runId);
    assert.equal(unchanged.status, finalized.status);
    assert.equal(unchanged.current_stage, finalized.current_stage);
    assert.equal(unchanged.updated_at, finalized.updated_at);
    assert.equal(unchanged.completed_at, finalized.completed_at);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
