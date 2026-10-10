import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, stat, utimes, writeFile } from "node:fs/promises";
import { hostname, tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import {
  auditRun,
  buildTimeline,
  canonicalJson,
  chainEvent,
  verifyEventChain,
  withAppendLock,
} from "../../src/local-runtime/audit.js";
import { createRunConsoleReporter } from "../../src/local-runtime/console-reporter.js";
import { fineTuneRunRequestSchema, localRunnerConfigSchema } from "../../src/local-runtime/contracts.js";
import { canonicalLocalPipeline, runLocalFineTune, runLocalPipeline } from "../../src/local-runtime/orchestrator.js";
import { createLocalStore } from "../../src/local-runtime/store.js";

function chain(count: number) {
  const events: Array<Record<string, unknown>> = [];
  for (let index = 0; index < count; index += 1) {
    events.push(chainEvent({ stage: `stage-${index}`, message: "m", occurred_at: `2026-10-10T00:00:0${index}.000Z` }, events.at(-1)));
  }
  return events;
}

test("canonical JSON sorts keys at every level and drops undefined", () => {
  assert.equal(canonicalJson({ b: 1, a: { d: undefined, c: [{ z: 1, y: 2 }] } }), '{"a":{"c":[{"y":2,"z":1}]},"b":1}');
});

test("a chained log verifies and every kind of tampering is located", () => {
  const events = chain(4);
  assert.deepEqual(verifyEventChain(events), {
    status: "verified", events: 4, chained_events: 4, legacy_events: 0, unchained_after_chain: 0, head_hash: events[3]!.hash,
  });
  const edited = events.map((event, index) => index === 1 ? { ...event, message: "rewritten" } : event);
  assert.match(verifyEventChain(edited).break!.reason, /was edited/);
  assert.equal(verifyEventChain(edited).break!.index, 1);
  const removed = [events[0]!, events[2]!, events[3]!];
  assert.match(verifyEventChain(removed).break!.reason, /removed, reordered or inserted/);
  const reordered = [events[0]!, events[2]!, events[1]!, events[3]!];
  assert.equal(verifyEventChain(reordered).status, "broken");
  // Recomputing one event's own hash still breaks the next link.
  const rehashed = [...events];
  rehashed[1] = chainEvent({ stage: "stage-1", message: "forged", occurred_at: events[1]!.occurred_at as string }, events[0]);
  assert.match(verifyEventChain(rehashed).break!.reason, /previous-event hash/);
  const mixed = verifyEventChain([...events, { stage: "x", message: "unchained" }]);
  assert.equal(mixed.status, "mixed");
  assert.equal(mixed.unchained_after_chain, 1);
  // A chained event may follow an unchained one written by an older TT process.
  const resumed = [...events, { stage: "x", message: "old worker" }];
  resumed.push(chainEvent({ stage: "y", message: "new" }, events.at(-1)));
  assert.equal(verifyEventChain(resumed).status, "mixed");
  assert.equal(verifyEventChain([]).status, "empty");
});

test("events from older TT versions are reported as legacy, not as tampering", () => {
  const legacy = [{ stage: "queued", message: "old" }, { stage: "preparing", message: "old" }];
  const continued = [...legacy];
  continued.push(chainEvent({ stage: "training", message: "new" }, undefined));
  const result = verifyEventChain(continued);
  assert.equal(result.status, "legacy");
  assert.equal(result.legacy_events, 2);
  assert.equal(verifyEventChain(legacy).status, "legacy");
});

test("the timeline turns events into timed stages", () => {
  const timeline = buildTimeline([
    { stage: "queued", status: "queued", message: "q", occurred_at: "2026-10-10T00:00:00.000Z" },
    { stage: "training", status: "running", message: "t", occurred_at: "2026-10-10T00:00:05.000Z" },
    { stage: "training", status: "running", message: "t2", occurred_at: "2026-10-10T00:30:00.000Z" },
    { stage: "completed", status: "completed", message: "done", occurred_at: "2026-10-10T01:00:05.000Z" },
  ]);
  assert.deepEqual(timeline.map((entry) => [entry.stage, entry.duration_seconds, entry.message]), [
    ["queued", 5, "q"],
    ["training", 3600, "t2"],
    ["completed", 0, "done"],
  ]);
});

test("the append lock serializes writers and reclaims a crashed owner's lock at once", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-audit-lock-"));
  try {
    const lock = join(root, "events.lock");
    const order: string[] = [];
    await Promise.all([1, 2, 3].map((index) => withAppendLock(lock, async () => {
      order.push(`start-${index}`);
      await new Promise((resolve) => setTimeout(resolve, 15));
      order.push(`end-${index}`);
    })));
    for (let index = 0; index < order.length; index += 2) {
      assert.equal(order[index]!.replace("start", "end"), order[index + 1]);
    }
    // A lock whose owner process is gone is reclaimed without waiting.
    await mkdir(lock);
    await writeFile(join(lock, "owner.json"), JSON.stringify({ token: "dead", pid: 2 ** 22 + 12345, host: hostname() }));
    const started = Date.now();
    assert.equal(await withAppendLock(lock, async () => "recovered", { timeoutMs: 2_000 }), "recovered");
    assert.ok(Date.now() - started < 1_000);
    await assert.rejects(stat(lock), { code: "ENOENT" });
    // A live owner is waited for, never broken.
    await mkdir(lock);
    await writeFile(join(lock, "owner.json"), JSON.stringify({ token: "live", pid: process.pid, host: hostname() }));
    await assert.rejects(withAppendLock(lock, async () => "stolen", { timeoutMs: 200 }), /Timed out/);
    assert.equal(JSON.parse(await readFile(join(lock, "owner.json"), "utf8")).token, "live");
    // An owner record that never got written counts as abandoned after a grace period.
    await rm(lock, { recursive: true });
    await mkdir(lock);
    const old = new Date(Date.now() - 60_000);
    await utimes(lock, old, old);
    assert.equal(await withAppendLock(lock, async () => "ownerless", { timeoutMs: 2_000 }), "ownerless");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("an interrupted write does not block later events or hide them from the audit", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-audit-torn-"));
  try {
    const store = createLocalStore(join(root, "store"));
    const request = runFixture("77777777-7777-4777-8777-777777777777");
    await store.startRun({ request, artifactDir: join(root, "artifacts") });
    const path = join(root, "store", "runs", request.run_id, "progress.jsonl");
    await writeFile(path, `${await readFile(path, "utf8")}{"id":"torn","stag`);
    await store.updateRun({ runId: request.run_id, status: "training", stage: "training", message: "after crash" });
    const log = await store.getRunEventLog(request.run_id);
    assert.equal(log.unreadable_lines, 1);
    assert.deepEqual(log.events.map((event) => event.stage), ["queued", "training"]);
    assert.equal(verifyEventChain(log.events as unknown as Array<Record<string, unknown>>).status, "verified");
    const audit = await auditRun(store, request.run_id);
    assert.match(audit.findings.join(" "), /1 event line\(s\) are unreadable/);
    assert.notEqual(audit.verdict, "tampered");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("concurrent writers from separate store instances keep one valid chain", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-audit-concurrent-"));
  try {
    const request = fineTuneRunRequestSchema.parse({
      run_id: "33333333-3333-4333-8333-333333333333",
      user_id: "local-user",
      behavior_spec_id: "22222222-2222-4222-8222-222222222222",
      run_number: 1,
      spec_snapshot: {
        name: "Audit", description: "", system_prompt: "Label.", guidelines: [], constraints: [],
        base_model: "Qwen/Qwen3.5-2B", examples: [{ input: "a", output: "b" }, { input: "c", output: "d" }],
      },
    });
    const first = createLocalStore(join(root, "store"));
    await first.startRun({ request, artifactDir: join(root, "artifacts") });
    const writers = [first, createLocalStore(join(root, "store")), createLocalStore(join(root, "store"))];
    await Promise.all(Array.from({ length: 24 }, (_, index) => writers[index % 3]!.updateRun({
      runId: request.run_id, status: "training", stage: "training", message: `step ${index}`,
    })));
    const events = await first.getRunEvents(request.run_id);
    assert.equal(events.length, 25);
    assert.equal(verifyEventChain(events as unknown as Array<Record<string, unknown>>).status, "verified");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

function runFixture(runId: string) {
  return fineTuneRunRequestSchema.parse({
    run_id: runId,
    user_id: "local-user",
    behavior_spec_id: "22222222-2222-4222-8222-222222222222",
    run_number: 1,
    spec_snapshot: {
      name: "Audit", description: "", system_prompt: "Return labels.", guidelines: [], constraints: [],
      base_model: "Qwen/Qwen3.5-2B",
      examples: [
        { input: "Classify: good", output: "positive" },
        { input: "Classify: bad", output: "negative" },
        { input: "Classify: fine", output: "positive" },
      ],
    },
    hyperparameters: { n_epochs: 2, lora_rank: 8 },
  });
}

test("a completed run records provenance and its audit detects report edits", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-audit-run-"));
  try {
    const config = localRunnerConfigSchema.parse({ artifactRoot: join(root, "artifacts"), storeRoot: join(root, "store"), dryRun: true });
    const request = runFixture("44444444-4444-4444-8444-444444444444");
    const result = await runLocalPipeline({
      request,
      config,
      pipeline: canonicalLocalPipeline(),
      specFile: { path: "/project/tunedtensor.json", sha256: "a".repeat(64) },
    });
    const provenance = result.report!.provenance!;
    assert.equal(provenance.schema_version, 1);
    assert.equal(provenance.execution.target, "local");
    assert.equal(provenance.execution.training_backend, "local-uv");
    assert.equal(provenance.spec.file_sha256, "a".repeat(64));
    assert.match(provenance.spec.snapshot_sha256, /^[a-f0-9]{64}$/);
    assert.equal(provenance.base_model.id, "Qwen/Qwen3.5-2B");
    assert.match(provenance.base_model.revision ?? "", /^[a-f0-9]{40}$/);
    assert.deepEqual(provenance.training.hyperparameters, { n_epochs: 2, lora_rank: 8 });
    assert.match(provenance.data.compiled_training_sha256 ?? "", /^[a-f0-9]{64}$/);
    assert.equal(provenance.data.eval_split, "spec_holdout");
    assert.match(provenance.software.python_lock_sha256 ?? "", /^[a-f0-9]{64}$/);

    const store = createLocalStore(config.storeRoot);
    const audit = await auditRun(store, request.run_id);
    assert.equal(audit.verdict, "verified", audit.findings.join(" "));
    assert.equal(audit.report.files.length, 2);
    assert.ok(audit.report.files.every((file) => file.matches_recorded));
    assert.deepEqual(audit.timeline.map((entry) => entry.stage), [
      "queued", "preparing", "evaluating_baseline", "training", "evaluating_candidate", "reporting", "completed",
    ]);
    assert.deepEqual(audit.provenance, provenance);

    assert.equal(audit.artifact_manifest.file?.matches_recorded, true);
    const eventsPath = join(config.storeRoot, "runs", request.run_id, "progress.jsonl");
    const eventsText = await readFile(eventsPath, "utf8");
    // Dropping the completion event (and its digests) is truncation, not an old version.
    await writeFile(eventsPath, `${eventsText.trim().split("\n").slice(0, -1).join("\n")}\n`);
    const truncated = await auditRun(store, request.run_id);
    assert.equal(truncated.verdict, "tampered");
    assert.match(truncated.findings.join(" "), /log was truncated/);
    await writeFile(eventsPath, "");
    assert.equal((await auditRun(store, request.run_id)).verdict, "tampered");
    await writeFile(eventsPath, eventsText);
    const manifestPath = audit.artifact_manifest.file!.path;
    const manifest = await readFile(manifestPath, "utf8");
    await writeFile(manifestPath, `${manifest} `);
    assert.match((await auditRun(store, request.run_id)).findings.join(" "), /Artifact manifest .* changed/);
    await writeFile(manifestPath, manifest);
    assert.equal((await auditRun(store, request.run_id)).verdict, "verified");

    const reportPath = audit.report.files[0]!.path;
    const original = await readFile(reportPath, "utf8");
    await writeFile(reportPath, original.replace('"status": "completed"', '"status": "completed" '));
    const tampered = await auditRun(store, request.run_id);
    assert.equal(tampered.verdict, "tampered");
    assert.match(tampered.findings.join(" "), /changed after the run completed/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("an unfinished or legacy run is incomplete rather than verified", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-audit-incomplete-"));
  try {
    const store = createLocalStore(join(root, "store"));
    const request = runFixture("55555555-5555-4555-8555-555555555555");
    await store.startRun({ request, artifactDir: join(root, "artifacts") });
    const audit = await auditRun(store, request.run_id);
    assert.equal(audit.verdict, "incomplete");
    assert.match(audit.findings.join(" "), /only completed runs have recorded report digests/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("live pipeline output shows elapsed time, hides process logs unless verbose", () => {
  let clock = 0;
  const lines: string[] = [];
  const quiet = createRunConsoleReporter({ write: (text) => lines.push(text), now: () => clock });
  clock = 3_725_000;
  void quiet.onEvent?.({ stage: "training", status: "running", message: "Training adapter.", details: { log_path: "/x/train.log", metrics: 1, command: ["uv", "run"], nested: { a: 1 } } });
  assert.equal(quiet.verbose, false);
  assert.deepEqual(lines, ["[tt] +01:02:05 training: Training adapter. (log_path=/x/train.log command=uv run)\n"]);
  const verbose = createRunConsoleReporter({ write: (text) => lines.push(text), now: () => clock, verbose: true });
  assert.equal(verbose.verbose, true);
  void verbose.onLog?.({ stage: "training", stream: "stderr", message: "step 1 token=abc secret=hunter2" });
  void verbose.onLog?.({ stage: "training", stream: "stderr", message: "step 1 token=abc secret=hunter2" });
  assert.equal(lines.length, 2);
  assert.match(lines[1]!, /secret=\[redacted\]/);
});

test("the legacy one-command run is also chained and verifiable", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-audit-legacy-command-"));
  try {
    const config = localRunnerConfigSchema.parse({ artifactRoot: join(root, "artifacts"), storeRoot: join(root, "store"), dryRun: true });
    const request = runFixture("66666666-6666-4666-8666-666666666666");
    await runLocalFineTune({ request, config });
    assert.equal((await auditRun(createLocalStore(config.storeRoot), request.run_id)).verdict, "verified");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
