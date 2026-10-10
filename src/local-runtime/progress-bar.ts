/**
 * Byte-level progress for long Hugging Face downloads.
 *
 * Python downloaders print `@@tt-progress {json}` lines; the process runner
 * turns them into {@link TransferProgress} events and a reporter renders them.
 * On a terminal the bar redraws in place; otherwise (CI, logs, the TT shell) it
 * prints a line every 10% or 30 seconds so captured output stays readable.
 */

import type { LocalRunReporter } from "./run-reporter.js";

export const PROGRESS_PREFIX = "@@tt-progress ";

export interface TransferProgress {
  label: string;
  phase: string;
  completed_bytes: number;
  total_bytes: number;
  files_completed: number;
  files_total: number;
}

function nonNegativeInteger(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? Math.floor(value) : null;
}

export function parseProgressLine(line: string): TransferProgress | null {
  const start = line.indexOf(PROGRESS_PREFIX);
  if (start !== 0) return null;
  let value: unknown;
  try {
    value = JSON.parse(line.slice(PROGRESS_PREFIX.length));
  } catch {
    return null;
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const record = value as Record<string, unknown>;
  const completed = nonNegativeInteger(record.completed_bytes);
  const total = nonNegativeInteger(record.total_bytes);
  if (completed === null || total === null) return null;
  return {
    label: typeof record.label === "string" ? record.label : "download",
    phase: typeof record.phase === "string" ? record.phase : "download",
    completed_bytes: completed,
    total_bytes: total,
    files_completed: nonNegativeInteger(record.files_completed) ?? 0,
    files_total: nonNegativeInteger(record.files_total) ?? 0,
  };
}

export function formatBytes(bytes: number): string {
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${unit === 0 ? value.toFixed(0) : value.toFixed(value >= 100 ? 0 : value >= 10 ? 1 : 2)} ${units[unit]}`;
}

export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "--";
  const whole = Math.round(seconds);
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  const rest = whole % 60;
  if (hours > 0) return `${hours}h ${String(minutes).padStart(2, "0")}m`;
  if (minutes > 0) return `${minutes}m ${String(rest).padStart(2, "0")}s`;
  return `${rest}s`;
}

interface Sample {
  at: number;
  bytes: number;
}

/** Throughput over a sliding window so the ETA follows Wi-Fi speed changes. */
export class RateTracker {
  private samples: Sample[] = [];

  constructor(private readonly windowMs = 15_000) {}

  add(at: number, bytes: number): void {
    const last = this.samples.at(-1);
    // A retry can restart a partially written file; restart the window.
    if (last && bytes < last.bytes) this.samples = [];
    this.samples.push({ at, bytes });
    while (this.samples.length > 2 && at - this.samples[0]!.at > this.windowMs) this.samples.shift();
  }

  bytesPerSecond(): number | null {
    const first = this.samples[0];
    const last = this.samples.at(-1);
    if (!first || !last || last.at - first.at < 500) return null;
    return (last.bytes - first.bytes) / ((last.at - first.at) / 1000);
  }
}

/**
 * One progress line. With `columns`, a bar fills the space the text leaves
 * (10 to 30 cells) and is dropped when the terminal is too narrow.
 */
export function formatProgressLine(
  progress: TransferProgress,
  rate: number | null,
  columns?: number,
): string {
  const total = progress.total_bytes;
  const fraction = total > 0 ? Math.min(1, progress.completed_bytes / total) : 1;
  // `drop` orders what a narrow terminal loses first; percent and size stay.
  const fields: Array<{ text: string; drop?: number }> = [
    { text: `${(fraction * 100).toFixed(1).padStart(5)}%` },
    { text: `${formatBytes(progress.completed_bytes)} / ${formatBytes(total)}` },
  ];
  const done = progress.completed_bytes >= total;
  if (!done && rate !== null && rate > 0) {
    fields.push(
      { text: `${formatBytes(rate)}/s`, drop: 2 },
      { text: `ETA ${formatDuration((total - progress.completed_bytes) / rate)}`, drop: 3 },
    );
  }
  if (progress.files_total > 0) fields.push({ text: `${progress.files_completed}/${progress.files_total} files`, drop: 1 });
  const prefix = `[tt] ${progress.label}: `;
  const join = () => fields.map((field) => field.text).join("  ");
  if (columns === undefined) return `${prefix}${join()}`;
  for (const rank of [1, 2, 3]) {
    if (prefix.length + join().length <= columns - 1) break;
    const index = fields.findIndex((field) => field.drop === rank);
    if (index !== -1) fields.splice(index, 1);
  }
  const text = join();
  const barWidth = Math.min(30, columns - 1 - prefix.length - text.length - 3);
  if (barWidth < 10) return `${prefix}${text}`.slice(0, Math.max(0, columns - 1));
  const filled = Math.round(fraction * barWidth);
  return `${prefix}[${"#".repeat(filled)}${"-".repeat(barWidth - filled)}] ${text}`;
}

export interface ProgressRenderer {
  update(progress: TransferProgress): void;
  /** Clear an in-place bar before other output is written. */
  interrupt(): void;
}

export function createProgressRenderer(options: {
  write: (text: string) => void;
  tty: boolean;
  columns?: number;
  now?: () => number;
  lineIntervalMs?: number;
}): ProgressRenderer {
  const now = options.now ?? (() => Date.now());
  const rates = new Map<string, RateTracker>();
  const lastLine = new Map<string, { at: number; decile: number }>();
  const completed = new Set<string>();
  let drawn = false;
  let lastDraw = Number.NEGATIVE_INFINITY;

  const finished = (progress: TransferProgress) =>
    progress.phase === "downloaded" || (progress.total_bytes > 0 && progress.completed_bytes >= progress.total_bytes);

  return {
    update(progress) {
      const at = now();
      const tracker = rates.get(progress.label) ?? new RateTracker();
      rates.set(progress.label, tracker);
      tracker.add(at, progress.completed_bytes);
      const done = finished(progress);
      if (options.tty) {
        // The downloader reports completion more than once; print it once.
        if (done && completed.has(progress.label)) return;
        if (!done) completed.delete(progress.label);
        if (!done && at - lastDraw < 100) return;
        lastDraw = at;
        const line = formatProgressLine(progress, tracker.bytesPerSecond(), Math.max(40, options.columns ?? 80));
        options.write(`\r\x1b[2K${line}${done ? "\n" : ""}`);
        drawn = !done;
        if (done) {
          rates.delete(progress.label);
          completed.add(progress.label);
        }
        return;
      }
      const fraction = progress.total_bytes > 0 ? progress.completed_bytes / progress.total_bytes : 1;
      const decile = Math.floor(fraction * 10);
      const previous = lastLine.get(progress.label);
      const interval = options.lineIntervalMs ?? 30_000;
      if (previous && !done && decile <= previous.decile && at - previous.at < interval) return;
      if (previous && done && previous.decile === 10) return;
      lastLine.set(progress.label, { at, decile: done ? 10 : decile });
      options.write(`${formatProgressLine(progress, tracker.bytesPerSecond())}\n`);
    },
    interrupt() {
      if (!drawn) return;
      options.write("\r\x1b[2K");
      drawn = false;
    },
  };
}

/** Whether stderr should get an in-place bar; the parent `tt` sets the hint for its child. */
export function progressTty(env: NodeJS.ProcessEnv = process.env, stream: { isTTY?: boolean } = process.stderr): boolean {
  if (env.TT_PROGRESS_TTY === "0") return false;
  return env.TT_PROGRESS_TTY === "1" || Boolean(stream.isTTY);
}

export function progressColumns(env: NodeJS.ProcessEnv = process.env, stream: { columns?: number } = process.stderr): number | undefined {
  const hinted = Number(env.TT_PROGRESS_COLUMNS);
  if (Number.isInteger(hinted) && hinted > 0) return hinted;
  return stream.columns;
}

/** Minimal stderr reporter for download steps run from the main `tt` process. */
export function createTransferReporter(stream: NodeJS.WriteStream = process.stderr): LocalRunReporter {
  const progress = createProgressRenderer({
    write: (text) => stream.write(text),
    tty: progressTty(process.env, stream),
    columns: progressColumns(process.env, stream),
  });
  return {
    onEvent(event) {
      progress.interrupt();
      stream.write(`[tt] ${event.stage}: ${event.message}\n`);
    },
    onLog(log) {
      progress.interrupt();
      stream.write(`[tt] ${log.stage}: ${log.message}\n`);
    },
    onProgress(update) {
      progress.update(update);
    },
  };
}
