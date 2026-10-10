import { sanitizeLogLine, type LocalRunProgressEvent, type LocalRunReporter } from "./run-reporter.js";

function elapsedLabel(milliseconds: number): string {
  const total = Math.max(0, Math.floor(milliseconds / 1000));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  return `+${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function detailText(details: LocalRunProgressEvent["details"]): string {
  if (!details) return "";
  return Object.entries(details)
    .map(([key, value]) => {
      if (value === null || value === undefined) return null;
      if (typeof value === "object") return null;
      const text = String(value);
      return text.length > 120 ? null : `${key}=${text}`;
    })
    .filter((value): value is string => Boolean(value))
    .slice(0, 5)
    .join(" ");
}

/**
 * Live stage output for a pipeline run on stderr, stamped with time since the
 * run started so a long run shows where its hours went. Process logs appear
 * only with `verbose`; the full logs and the event record stay in the run
 * directory either way.
 */
export function createRunConsoleReporter(options: {
  write?: (text: string) => void;
  verbose?: boolean;
  now?: () => number;
} = {}): LocalRunReporter {
  const write = options.write ?? ((text: string) => { process.stderr.write(text); });
  const now = options.now ?? (() => Date.now());
  const started = now();
  let lastLog = "";
  return {
    verbose: options.verbose ?? false,
    onEvent(event) {
      const details = detailText(event.details);
      write(`${sanitizeLogLine(`[tt] ${elapsedLabel(now() - started)} ${event.stage}: ${event.message}${details ? ` (${details})` : ""}`)}\n`);
    },
    onLog(log) {
      const line = sanitizeLogLine(`[tt] ${elapsedLabel(now() - started)} ${log.stage}${log.stream ? ` ${log.stream}` : ""}: ${log.message}`);
      // tqdm redraws one line many times; keep consecutive duplicates out.
      const key = line.slice(line.indexOf(" ", 5));
      if (key === lastLog) return;
      lastLog = key;
      write(`${line}\n`);
    },
  };
}
