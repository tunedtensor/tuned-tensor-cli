import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { hostname, tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  createBackgroundRunStore,
  currentRequester,
  maxParallelRuns,
  superviseBackgroundRun,
  summarizePipelineResult,
  workspaceSnapshotPath,
  type BackgroundRunStore,
  type NewBackgroundRun,
} from "../local-runtime/background-runs.js";

let root: string;
let store: BackgroundRunStore;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "tt-background-runs-"));
  store = createBackgroundRunStore(join(root, "runs"));
});
afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

function newRun(overrides: Partial<NewBackgroundRun> = {}): NewBackgroundRun {
  return {
    reason: "Check whether the new negative examples fix sarcasm.",
    requested_by: currentRequester("agent", { thread_id: "thread-1", action_id: "action-1" }),
    choices: {
      engine: "adapter",
      spec_path: "./tunedtensor.json",
      spec_sha256: "a".repeat(64),
      config_path: null,
      config_sha256: null,
      steps: [{ id: "train", uses: "train", target: "local" }],
      target: "local",
      workspace_root: root,
    },
    sealed_spec: { name: "Sealed" },
    ...overrides,
  };
}

/** A stand-in `tt` entrypoint: records its argv, prints progress on stderr and a JSON result on stdout. */
function fakeEntrypoint(body: string): string {
  const path = join(root, `fake-tt-${Math.random().toString(16).slice(2)}.mjs`);
  writeFileSync(path, `import { readFileSync, writeFileSync } from "node:fs";
if (process.channel) {
  process.on("message", message => {
    if (message?.ttSignal === "SIGTERM") process.emit("SIGTERM");
  });
  process.channel.unref();
}
${body}\n`);
  return path;
}

describe("background run store", () => {
  it("records why, who and what before launch, with a verifiable event chain", async () => {
    const run = await store.create(newRun(), { maxParallel: 1 });
    expect(run).toMatchObject({
      status: "starting",
      reason: "Check whether the new negative examples fix sarcasm.",
      requested_by: { via: "agent", thread_id: "thread-1", action_id: "action-1", host: hostname() },
      choices: { spec_sha256: "a".repeat(64), target: "local" },
    });
    expect(JSON.parse(readFileSync(store.paths(run.id).spec, "utf8"))).toEqual({ name: "Sealed" });
    const { events, chain } = await store.events(run.id.slice(0, 8));
    expect(events.map((event) => event.type)).toEqual(["requested"]);
    expect(chain.status).toBe("verified");
  });

  it("refuses a launch beyond the parallel limit and allows it when raised", async () => {
    const first = await store.create(newRun(), { maxParallel: 1 });
    await store.update(first.id, (run) => { run.process = { host: hostname(), supervisor_pid: process.pid }; });
    await expect(store.create(newRun(), { maxParallel: 1 })).rejects.toThrow(/already active \(limit 1/);
    await expect(store.create(newRun(), { maxParallel: 2 })).resolves.toMatchObject({ status: "starting" });
    expect(await store.active()).toHaveLength(2);
  });

  it("marks a run lost when its supervisor process is gone", async () => {
    const run = await store.create(newRun());
    await store.update(run.id, (current) => {
      current.status = "running";
      current.process = { host: hostname(), supervisor_pid: 2 ** 22 + 12345 };
    });
    const reconciled = await store.get(run.id);
    expect(reconciled.status).toBe("lost");
    expect(reconciled.outcome?.error).toMatch(/supervisor stopped/);
    expect((await store.events(run.id)).events.at(-1)?.type).toBe("lost");
    // A lost run no longer counts against the limit.
    await expect(store.create(newRun(), { maxParallel: 1 })).resolves.toBeDefined();
  });

  it("rechecks a disappeared supervisor under lock before appending a lost event", async () => {
    const run = await store.create(newRun());
    const deadPid = 2 ** 22 + 12345;
    const running = await store.update(run.id, (current) => {
      current.status = "running";
      current.process = { host: hostname(), supervisor_pid: deadPid };
    });
    const originalKill = process.kill.bind(process);
    const kill = vi.spyOn(process, "kill").mockImplementation((pid, signal) => {
      if (pid === deadPid) {
        // The supervisor finishes between the reader's snapshot and PID check.
        writeFileSync(store.paths(run.id).record, JSON.stringify({ ...running, status: "succeeded", outcome: { exit_code: 0 } }));
        throw Object.assign(new Error("Gone"), { code: "ESRCH" });
      }
      return originalKill(pid, signal);
    });
    try {
      expect(await store.get(run.id)).toMatchObject({ status: "succeeded", outcome: { exit_code: 0 } });
      expect((await store.events(run.id)).events.map(event => event.type)).toEqual(["requested"]);
    } finally {
      kill.mockRestore();
    }
  });

  it("records a disappeared supervisor once when readers reconcile concurrently", async () => {
    const run = await store.create(newRun());
    await store.update(run.id, current => {
      current.status = "running";
      current.process = { host: hostname(), supervisor_pid: 2 ** 22 + 12345 };
    });
    expect((await Promise.all([store.get(run.id), store.get(run.id)])).map(current => current.status)).toEqual(["lost", "lost"]);
    expect((await store.events(run.id)).events.map(event => event.type)).toEqual(["requested", "lost"]);
  });

  it("records the cancel reason and refuses to cancel a finished run", async () => {
    const run = await store.create(newRun());
    const cancelled = await store.requestCancel(run.id, { reason: "Wrong dataset", requested_by: currentRequester("cli") });
    expect(cancelled).toMatchObject({ status: "cancel_requested", cancel: { reason: "Wrong dataset", requested_by: { via: "cli" } } });
    await store.update(run.id, (current) => { current.status = "cancelled"; });
    await expect(store.requestCancel(run.id, { reason: "again", requested_by: currentRequester("cli") })).rejects.toThrow(/already cancelled/);
    expect((await store.events(run.id)).chain.status).toBe("verified");
  });

  it("requires a reason and an unambiguous ID", async () => {
    await expect(store.create(newRun({ reason: "   " }))).rejects.toThrow(/needs a reason/);
    await expect(store.get("zzzz")).rejects.toThrow(/Invalid background run ID/);
    await expect(store.get("0000")).rejects.toThrow(/No background run matches/);
  });

  it("returns a bounded, control-free log tail", async () => {
    const run = await store.create(newRun());
    writeFileSync(store.paths(run.id).log, "one\n\u001b[32mtwo\u001b[0m\rthree\nfour\n");
    expect(await store.logTail(run.id, 2)).toBe("three\nfour");
    expect(await store.logTail(run.id)).toBe("one\ntwo\nthree\nfour");
  });

  it("validates TT_MAX_PARALLEL_RUNS", () => {
    expect(maxParallelRuns({})).toBe(1);
    expect(maxParallelRuns({ TT_MAX_PARALLEL_RUNS: "3" })).toBe(3);
    expect(() => maxParallelRuns({ TT_MAX_PARALLEL_RUNS: "0" })).toThrow(/1 to 8/);
    expect(() => maxParallelRuns({ TT_MAX_PARALLEL_RUNS: "lots" })).toThrow(/1 to 8/);
  });
});

describe("pipeline result summaries", () => {
  it("keeps adapter evaluation deltas and the model ID", () => {
    expect(summarizePipelineResult({
      status: "completed",
      request: { run_id: "11111111-1111-4111-8111-111111111111" },
      report: {
        fine_tuned_model_id: "file:///runs/artifacts/model",
        baseline: { pass_rate: 0.5, average_score: 0.6, results: [{ huge: true }] },
        candidate: { pass_rate: 0.8, average_score: 0.85 },
        comparison: { pass_rate_delta: 0.3, regressions: 1, details: [] },
        general_regression: { passed: true },
      },
      reportPath: "/runs/report.json",
      artifactDir: "/runs/artifacts",
    })).toEqual({
      pipeline_status: "completed",
      local_run_id: "11111111-1111-4111-8111-111111111111",
      model_id: "local-11111111-1111-4111-8111-111111111111",
      report_path: "/runs/report.json",
      output_dir: "/runs/artifacts",
      metrics: {
        baseline: { pass_rate: 0.5, average_score: 0.6 },
        candidate: { pass_rate: 0.8, average_score: 0.85 },
        comparison: { pass_rate_delta: 0.3, regressions: 1 },
        general_regression_passed: true,
      },
    });
  });

  it("does not invent a model for evaluation-only or dry-run results", () => {
    const result = {
      status: "stage_completed",
      request: { run_id: "11111111-1111-4111-8111-111111111111" },
      outputs: { baseline: { report: { pass_rate: 1 } } },
    };
    expect(summarizePipelineResult(result).model_id).toBeUndefined();
    expect(summarizePipelineResult({
      ...result,
      report: { fine_tuned_model_id: "dry-run", training: { metrics: { dry_run: true } } },
    }).model_id).toBeUndefined();
  });

  it("links a train-only recipe to its registered model", () => {
    expect(summarizePipelineResult({
      status: "stage_completed",
      request: { run_id: "11111111-1111-4111-8111-111111111111" },
      outputs: { custom_train: { modelArtifactUri: "file:///runs/artifacts/model" } },
    }).model_id).toBe("local-11111111-1111-4111-8111-111111111111");
  });

  it("keeps per-step metrics for foundation and decision results", () => {
    expect(summarizePipelineResult({
      status: "succeeded",
      engine: "decision",
      run_dir: "/d/run",
      report_path: "/d/run/report.json",
      model_dir: "/d/run/model",
      steps: [
        { id: "baseline", decision_metrics: { accuracy: 0.6, log_loss: 0.9 } },
        { id: "compare", comparison: { pass_rate_delta: 0.2, log_loss_delta: -0.3 } },
      ],
    })).toMatchObject({
      report_path: "/d/run/report.json",
      model_id: "/d/run/model",
      metrics: { baseline: { accuracy: 0.6, log_loss: 0.9 }, compare: { comparison: { pass_rate_delta: 0.2 } } },
    });
  });
});

describe("background supervisor", () => {
  it("runs the sealed spec beside the reviewed one, records the outcome and removes the snapshot", async () => {
    writeFileSync(join(root, "tunedtensor.json"), "{}");
    const run = await store.create(newRun());
    const argvPath = join(root, "argv.json");
    const entrypoint = fakeEntrypoint(`
      const spec = process.argv[process.argv.indexOf("--spec") + 1];
      writeFileSync(${JSON.stringify(argvPath)}, JSON.stringify({ argv: process.argv.slice(2), spec: readFileSync(spec, "utf8") }));
      process.stderr.write("training 50%\\n");
      process.stdout.write(JSON.stringify({ status: "completed", request: { run_id: "22222222-2222-4222-8222-222222222222" } }));
    `);
    const finished = await superviseBackgroundRun({ store, id: run.id, entrypoint });
    const seen = JSON.parse(readFileSync(argvPath, "utf8"));
    expect(seen.argv.slice(0, 4)).toEqual(["--json", "pipeline", "run", "--spec"]);
    expect(seen.argv[4]).toBe(workspaceSnapshotPath(run));
    expect(JSON.parse(seen.spec)).toEqual({ name: "Sealed" });
    expect(existsSync(workspaceSnapshotPath(run))).toBe(false);
    expect(finished).toMatchObject({
      status: "succeeded",
      outcome: { exit_code: 0, local_run_id: "22222222-2222-4222-8222-222222222222", pipeline_status: "completed" },
    });
    expect(finished.started_at).toBeDefined();
    expect(await store.logTail(run.id)).toBe("training 50%");
    const { events, chain } = await store.events(run.id);
    expect(events.map((event) => event.type)).toEqual(["requested", "started", "finished"]);
    expect(chain.status).toBe("verified");
  });

  it("records a failure with the pipeline's last log line", async () => {
    writeFileSync(join(root, "tunedtensor.json"), "{}");
    const run = await store.create(newRun());
    const entrypoint = fakeEntrypoint(`process.stderr.write("CUDA out of memory\\n"); process.exit(3);`);
    const finished = await superviseBackgroundRun({ store, id: run.id, entrypoint });
    expect(finished).toMatchObject({ status: "failed", outcome: { exit_code: 3, error: "CUDA out of memory" } });
  });

  it("records cancelled when cancellation arrives before the supervisor starts", async () => {
    const run = await store.create(newRun());
    // A launched supervisor that has not installed its handlers yet must not be signalled.
    await store.update(run.id, (current) => { current.process = { host: hostname(), supervisor_pid: process.pid }; });
    const kill = vi.spyOn(process, "kill");
    await store.requestCancel(run.id, { reason: "Changed my mind", requested_by: currentRequester("cli") });
    const entrypoint = fakeEntrypoint(`throw new Error("must not run");`);
    expect(await superviseBackgroundRun({ store, id: run.id, entrypoint })).toMatchObject({
      status: "cancelled",
      outcome: { error: "Cancelled before the pipeline started." },
    });
    expect(kill.mock.calls.filter(([, signal]) => signal === "SIGTERM")).toEqual([]);
    kill.mockRestore();
  });

  it("records a failure when the sealed snapshot cannot be written", async () => {
    const run = await store.create(newRun());
    const finished = await superviseBackgroundRun({
      store, id: run.id, entrypoint: fakeEntrypoint(""), snapshotPath: () => join(root, "missing-dir", "snapshot.json"),
    });
    expect(finished).toMatchObject({ status: "failed", outcome: { error: expect.stringMatching(/Could not prepare the run/) } });
    expect((await store.events(run.id)).events.map((event) => event.type)).toEqual(["requested", "finished"]);
  });

  it("stops the pipeline and records cleanup when cancellation is requested", async () => {
    writeFileSync(join(root, "tunedtensor.json"), "{}");
    const run = await store.create(newRun());
    const entrypoint = fakeEntrypoint(`
      process.stderr.write("started\\n");
      process.on("SIGTERM", () => { process.stderr.write("stopping cleanly\\n"); process.exit(143); });
      setInterval(() => {}, 1000);
    `);
    const supervising = superviseBackgroundRun({ store, id: run.id, entrypoint });
    // Wait until the child installed its handler, as a real pipeline does before GPU work.
    for (let attempt = 0; attempt < 250 && !(await store.logTail(run.id)).includes("started"); attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 20));
    }
    await store.requestCancel(run.id, { reason: "Wrong base model", requested_by: currentRequester("cli") });
    const finished = await supervising;
    expect(finished).toMatchObject({ status: "cancelled", cancel: { reason: "Wrong base model" } });
    expect(await store.logTail(run.id)).toContain("stopping cleanly");
    expect(existsSync(workspaceSnapshotPath(run))).toBe(false);
  });

  it.runIf(process.platform === "win32")("stops an unresponsive Windows pipeline after the cleanup grace period", async () => {
    writeFileSync(join(root, "tunedtensor.json"), "{}");
    const run = await store.create(newRun());
    const entrypoint = fakeEntrypoint(`
      process.stderr.write("started\\n");
      process.on("SIGTERM", () => {});
      setInterval(() => {}, 1000);
    `);
    const supervising = superviseBackgroundRun({ store, id: run.id, entrypoint });
    for (let attempt = 0; attempt < 250 && !(await store.logTail(run.id)).includes("started"); attempt += 1) {
      await new Promise(resolve => setTimeout(resolve, 20));
    }
    const childPid = (await store.get(run.id)).process!.pid!;
    await store.requestCancel(run.id, { reason: "Pipeline did not respond", requested_by: currentRequester("cli") });
    expect(await supervising).toMatchObject({ status: "cancelled" });
    expect(() => process.kill(childPid, 0)).toThrow();
    expect(existsSync(workspaceSnapshotPath(run))).toBe(false);
  });
});
