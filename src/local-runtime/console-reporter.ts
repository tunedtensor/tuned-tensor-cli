import { createProgressRenderer, progressTty, progressColumns } from "./progress-bar.js";
import { sanitizeLogLine, type LocalRunProgressEvent, type LocalRunReporter } from "./run-reporter.js";

function shortValue(value: unknown): string | null {
  if (value === undefined || value === null) return null;
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (Array.isArray(value)) return `[${value.length} items]`;
  return null;
}

/** One line for a run event: stage, message and up to five scalar details. */
export function formatRunEvent(event: LocalRunProgressEvent): string {
  const detailText = Object.entries(event.details ?? {})
    .filter(([key]) => key !== "metrics")
    .map(([key, value]) => {
      const formatted = key === "command" && Array.isArray(value)
        ? value.join(" ")
        : shortValue(value);
      return formatted ? `${key}=${formatted}` : null;
    })
    .filter((value): value is string => Boolean(value))
    .slice(0, 5)
    .join(" ");
  return sanitizeLogLine(`[tt] ${event.stage}: ${event.message}${detailText ? ` (${detailText})` : ""}`);
}

function elapsedLabel(milliseconds: number): string {
  const total = Math.max(0, Math.floor(milliseconds / 1000));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  return `+${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

/**
 * Live run output on stderr for `tt run` and `tt pipeline run`. Each line is
 * stamped with the time since the run started, so a long run shows where its
 * hours went. Process logs appear only with `verbose`; full logs and the event
 * record stay in the run directory either way.
 */
export function createRunConsoleReporter(options: {
  write?: (text: string) => void;
  verbose?: boolean;
  now?: () => number;
} = {}): LocalRunReporter {
  const write = options.write ?? ((text: string) => { process.stderr.write(text); });
  const now = options.now ?? (() => Date.now());
  const started = now();
  const stamp = (line: string) => line.replace(/^\[tt\] /, `[tt] ${elapsedLabel(now() - started)} `);
  const progress = createProgressRenderer({ write, tty: progressTty(), columns: progressColumns() });
  let lastLog = "";
  return {
    verbose: options.verbose ?? false,
    onProgress(update) { progress.update(update); },
    onEvent(event) {
      progress.interrupt();
      write(`${stamp(formatRunEvent(event))}\n`);
    },
    onLog(log) {
      const line = sanitizeLogLine(`[tt] ${log.stage}${log.stream ? ` ${log.stream}` : ""}: ${log.message}`);
      // tqdm redraws one line many times; keep consecutive duplicates out.
      if (line === lastLog) return;
      lastLog = line;
      progress.interrupt();
      write(`${stamp(line)}\n`);
    },
  };
}
