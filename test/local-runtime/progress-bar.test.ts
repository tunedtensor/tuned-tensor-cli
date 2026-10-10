import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import {
  createProgressRenderer,
  formatBytes,
  formatDuration,
  formatProgressLine,
  parseProgressLine,
  progressTty,
  RateTracker,
  type TransferProgress,
} from "../../src/local-runtime/progress-bar.js";
import { runLoggedProcess } from "../../src/local-runtime/process-runner.js";

const progress = (completed: number, total = 1_000, phase = "download"): TransferProgress => ({
  label: "model_prefetch",
  phase,
  completed_bytes: completed,
  total_bytes: total,
  files_completed: completed >= total ? 2 : 1,
  files_total: 2,
});

test("progress lines parse strictly", () => {
  assert.deepEqual(
    parseProgressLine('@@tt-progress {"label":"dataset_prefetch","phase":"download","completed_bytes":5,"total_bytes":10,"files_completed":0,"files_total":1}'),
    { label: "dataset_prefetch", phase: "download", completed_bytes: 5, total_bytes: 10, files_completed: 0, files_total: 1 },
  );
  for (const line of ["plain output", "@@tt-progress not json", '@@tt-progress {"completed_bytes":-1,"total_bytes":2}', ' @@tt-progress {"completed_bytes":1,"total_bytes":2}']) {
    assert.equal(parseProgressLine(line), null, line);
  }
});

test("bytes, durations and rates format for long downloads", () => {
  assert.equal(formatBytes(512), "512 B");
  assert.equal(formatBytes(1536), "1.50 KiB");
  assert.equal(formatBytes(64 * 1024 ** 3), "64.0 GiB");
  assert.equal(formatDuration(42), "42s");
  assert.equal(formatDuration(125), "2m 05s");
  assert.equal(formatDuration(3 * 3600 + 120), "3h 02m");
  assert.equal(formatDuration(Number.POSITIVE_INFINITY), "--");
  const rate = new RateTracker(10_000);
  rate.add(0, 0);
  assert.equal(rate.bytesPerSecond(), null);
  rate.add(2_000, 2_000);
  assert.equal(rate.bytesPerSecond(), 1_000);
  rate.add(3_000, 500); // a restarted file resets the window instead of going negative
  assert.equal(rate.bytesPerSecond(), null);
});

test("a progress line shows percent, size, rate, ETA and a width-fitted bar", () => {
  const line = formatProgressLine(progress(250), 50, 120);
  assert.match(line, /^\[tt\] model_prefetch: \[#+-+\]  25\.0%  250 B \/ 1000 B  50 B\/s  ETA 15s  1\/2 files$/);
  assert.ok(line.length < 120);
  const narrow = formatProgressLine(progress(250), 50, 60);
  assert.doesNotMatch(narrow, /\[[#-]+\]/, "narrow terminals drop the bar");
  assert.ok(narrow.length <= 59);
  assert.doesNotMatch(formatProgressLine(progress(1_000), 50), /ETA/);
});

test("the terminal renderer redraws one line and finishes with a newline", () => {
  let clock = 0;
  const output: string[] = [];
  const renderer = createProgressRenderer({ write: (text) => output.push(text), tty: true, columns: 100, now: () => clock });
  renderer.update(progress(0));
  clock = 50;
  renderer.update(progress(100)); // throttled
  clock = 1_000;
  renderer.update(progress(500));
  renderer.interrupt();
  clock = 2_000;
  renderer.update(progress(1_000, 1_000, "downloaded"));
  assert.equal(output.length, 4);
  assert.ok(output.slice(0, 2).every((text) => text.startsWith("\r\x1b[2K") && !text.endsWith("\n")));
  assert.equal(output[2], "\r\x1b[2K");
  assert.ok(output[3]!.endsWith("100.0%  1000 B / 1000 B  2/2 files\n"));
});

test("captured output gets a line per 10% or 30 seconds, never a flood", () => {
  let clock = 0;
  const output: string[] = [];
  const renderer = createProgressRenderer({ write: (text) => output.push(text), tty: false, now: () => clock });
  for (let completed = 0; completed <= 1_000; completed += 10) {
    clock += 100;
    renderer.update(progress(completed));
  }
  renderer.update(progress(1_000, 1_000, "downloaded"));
  assert.equal(output.length, 11);
  assert.ok(output.every((text) => text.endsWith("\n") && !text.includes("\r")));
  const slow: string[] = [];
  clock = 0;
  const heartbeat = createProgressRenderer({ write: (text) => slow.push(text), tty: false, now: () => clock });
  heartbeat.update(progress(1));
  clock = 31_000;
  heartbeat.update(progress(2));
  assert.equal(slow.length, 2, "a stalled download still reports every 30 seconds");
});

test("the TTY hint comes from the parent process", () => {
  assert.equal(progressTty({ TT_PROGRESS_TTY: "1" }, { isTTY: false }), true);
  assert.equal(progressTty({ TT_PROGRESS_TTY: "0" }, { isTTY: true }), false);
  assert.equal(progressTty({}, { isTTY: true }), true);
  assert.equal(progressTty({}, {}), false);
});

test("the process runner routes progress lines to the reporter, not the log stream", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-progress-runner-"));
  try {
    const updates: TransferProgress[] = [];
    const logs: string[] = [];
    const logPath = join(root, "process.log");
    const script = [
      "console.log('starting');",
      "console.log('@@tt-progress ' + JSON.stringify({label:'x',phase:'download',completed_bytes:1,total_bytes:2,files_completed:0,files_total:1}));",
      "console.log('done');",
    ].join("");
    const { exitCode } = await runLoggedProcess({
      command: process.execPath,
      commandArgs: ["-e", script],
      logPath,
      stage: "model_prefetch",
      reporter: { verbose: true, onProgress: (update) => { updates.push(update); }, onLog: (log) => { logs.push(log.message); } },
    });
    assert.equal(exitCode, 0);
    await new Promise((resolve) => setImmediate(resolve));
    assert.deepEqual(updates.map((update) => update.completed_bytes), [1]);
    assert.deepEqual(logs, ["starting", "done"]);
    assert.match(await readFile(logPath, "utf8"), /@@tt-progress/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
