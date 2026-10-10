import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { mkdir, rm, stat } from "node:fs/promises";

/**
 * Tamper-evident run event log.
 *
 * Each event stores its sequence number, the previous event's hash and its own
 * hash over a canonical JSON form. Editing, reordering or deleting an event
 * (other than truncating the newest ones) breaks the chain, and
 * `tt runs audit` reports where. The chain makes accidental or casual edits
 * evident; it is not a signature, since anyone who can write the store can
 * rewrite the whole log.
 */

export const AUDIT_CHAIN_VERSION = 1;

export interface ChainedFields {
  seq?: number;
  prev_hash?: string | null;
  hash?: string;
}

/** JSON with object keys sorted at every level, after dropping undefined values. */
export function canonicalJson(value: unknown): string {
  const normalized = JSON.parse(JSON.stringify(value ?? null)) as unknown;
  const encode = (entry: unknown): string => {
    if (entry === null || typeof entry !== "object") return JSON.stringify(entry);
    if (Array.isArray(entry)) return `[${entry.map(encode).join(",")}]`;
    const object = entry as Record<string, unknown>;
    return `{${Object.keys(object).sort().map((key) => `${JSON.stringify(key)}:${encode(object[key])}`).join(",")}}`;
  };
  return encode(normalized);
}

export function eventHash(event: Record<string, unknown>): string {
  const { hash: _hash, ...body } = event;
  return createHash("sha256").update(canonicalJson(body)).digest("hex");
}

/** Add seq, prev_hash and hash to a new event that follows `previous`. */
export function chainEvent<T extends Record<string, unknown>>(
  event: T,
  previous: (ChainedFields & Record<string, unknown>) | undefined,
): T & Required<ChainedFields> {
  const body = JSON.parse(JSON.stringify({
    ...event,
    seq: typeof previous?.seq === "number" ? previous.seq + 1 : 0,
    prev_hash: previous?.hash ?? null,
  })) as T & { seq: number; prev_hash: string | null };
  return { ...body, hash: eventHash(body) };
}

export type ChainStatus = "verified" | "legacy" | "broken" | "empty";

export interface ChainVerification {
  status: ChainStatus;
  events: number;
  chained_events: number;
  /** Events recorded before the chain existed (older TT versions). */
  legacy_events: number;
  head_hash: string | null;
  break?: { index: number; seq: number | null; reason: string };
}

/**
 * Verify an event list. Leading unchained events from older TT versions are
 * reported as legacy; once chaining starts, every later event must link.
 */
export function verifyEventChain(events: ReadonlyArray<Record<string, unknown>>): ChainVerification {
  let legacy = 0;
  let previous: (ChainedFields & Record<string, unknown>) | undefined;
  for (const [index, event] of events.entries()) {
    const chained = event as ChainedFields & Record<string, unknown>;
    if (chained.hash === undefined) {
      if (previous) {
        return result("broken", { index, seq: null, reason: "An unchained event follows chained events." });
      }
      legacy += 1;
      continue;
    }
    const expectedSeq = previous ? (previous.seq ?? -1) + 1 : 0;
    if (chained.seq !== expectedSeq) {
      return result("broken", { index, seq: chained.seq ?? null, reason: `Expected sequence ${expectedSeq}; an event was removed, reordered or inserted.` });
    }
    if ((chained.prev_hash ?? null) !== (previous?.hash ?? null)) {
      return result("broken", { index, seq: chained.seq, reason: "The previous-event hash does not match; an earlier event was changed or removed." });
    }
    if (eventHash(chained) !== chained.hash) {
      return result("broken", { index, seq: chained.seq, reason: "The event's contents do not match its hash; it was edited." });
    }
    previous = chained;
  }
  if (events.length === 0) return result("empty");
  return result(previous ? (legacy > 0 ? "legacy" : "verified") : "legacy");

  function result(status: ChainStatus, brk?: ChainVerification["break"]): ChainVerification {
    const chainedCount = events.filter((event) => (event as ChainedFields).hash !== undefined).length;
    return {
      status,
      events: events.length,
      chained_events: chainedCount,
      legacy_events: legacy,
      head_hash: (previous?.hash as string | undefined) ?? null,
      ...(brk ? { break: brk } : {}),
    };
  }
}

export async function sha256File(path: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}

/**
 * Cross-process mutual exclusion for appending to one event log. `mkdir` is
 * atomic; a lock older than `staleMs` is assumed to belong to a crashed writer.
 */
export async function withAppendLock<T>(
  lockPath: string,
  action: () => Promise<T>,
  options: { timeoutMs?: number; staleMs?: number } = {},
): Promise<T> {
  const timeoutMs = options.timeoutMs ?? 10_000;
  const staleMs = options.staleMs ?? 30_000;
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    try {
      await mkdir(lockPath, { mode: 0o700 });
      break;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
      const metadata = await stat(lockPath).catch(() => null);
      if (metadata && Date.now() - metadata.mtimeMs > staleMs) {
        await rm(lockPath, { recursive: true, force: true });
        continue;
      }
      if (Date.now() > deadline) throw new Error(`Timed out waiting for the run event log lock: ${lockPath}`);
      await new Promise((resolve) => setTimeout(resolve, 20));
    }
  }
  try {
    return await action();
  } finally {
    await rm(lockPath, { recursive: true, force: true });
  }
}

export interface TimelineEntry {
  stage: string;
  status: string;
  message: string;
  started_at: string;
  ended_at: string | null;
  duration_seconds: number | null;
}

/** Collapse consecutive events of the same stage into timed segments. */
export function buildTimeline(events: ReadonlyArray<{ stage: string; status: string; message: string; occurred_at: string }>): TimelineEntry[] {
  const timeline: TimelineEntry[] = [];
  for (const event of events) {
    const last = timeline.at(-1);
    if (last && last.stage === event.stage) {
      last.status = event.status;
      last.message = event.message;
      continue;
    }
    if (last) {
      last.ended_at = event.occurred_at;
      last.duration_seconds = Math.max(0, (Date.parse(event.occurred_at) - Date.parse(last.started_at)) / 1000);
    }
    timeline.push({
      stage: event.stage,
      status: event.status,
      message: event.message,
      started_at: event.occurred_at,
      ended_at: null,
      duration_seconds: null,
    });
  }
  const last = timeline.at(-1);
  if (last && ["completed", "failed", "cancelled", "stage_completed"].includes(last.stage)) {
    last.ended_at = last.started_at;
    last.duration_seconds = 0;
  }
  return timeline;
}

export type AuditVerdict = "verified" | "incomplete" | "tampered";

export interface RunAudit {
  run_id: string;
  status: string;
  verdict: AuditVerdict;
  findings: string[];
  chain: ChainVerification;
  report: {
    recorded_sha256: string | null;
    files: Array<{ path: string; sha256: string; matches_recorded: boolean | null }>;
  };
  timeline: TimelineEntry[];
  provenance: unknown;
}

/**
 * Check a run's evidence: the event chain, and that report files still match
 * the digest recorded when the run completed. `verified` needs both; older or
 * unfinished runs are `incomplete`; any mismatch is `tampered`.
 */
export async function auditRun(store: {
  getRun(id: string): Promise<{ id: string; status: string }>;
  getRunEvents(id: string): Promise<Array<{ stage: string; status: string; message: string; occurred_at: string; details?: Record<string, unknown> }>>;
  getRunReportPaths(id: string): Promise<string[]>;
  getRunReport(id: string): Promise<unknown>;
}, runId: string): Promise<RunAudit> {
  const state = await store.getRun(runId);
  const events = await store.getRunEvents(state.id);
  const chain = verifyEventChain(events as unknown as Array<Record<string, unknown>>);
  const completion = [...events].reverse().find((event) => event.stage === "completed");
  const recorded = typeof completion?.details?.report_sha256 === "string" ? completion.details.report_sha256 : null;
  const files = await Promise.all((await store.getRunReportPaths(state.id)).map(async (path) => {
    const sha256 = await sha256File(path);
    return { path, sha256, matches_recorded: recorded ? sha256 === recorded : null };
  }));
  const report = await store.getRunReport(state.id).catch(() => null) as { provenance?: unknown } | null;
  const findings: string[] = [];
  let verdict: AuditVerdict = "verified";
  const downgrade = (finding: string) => {
    findings.push(finding);
    if (verdict === "verified") verdict = "incomplete";
  };
  if (chain.status === "broken") {
    verdict = "tampered";
    findings.push(`Event log chain is broken at event ${chain.break!.index}: ${chain.break!.reason}`);
  } else if (chain.status === "legacy") {
    downgrade(`${chain.legacy_events} event(s) were recorded before event chaining; they cannot be verified.`);
  } else if (chain.status === "empty") {
    downgrade("The run has no events.");
  }
  for (const file of files) {
    if (file.matches_recorded === false) {
      verdict = "tampered";
      findings.push(`Report ${file.path} changed after the run completed (sha256 ${file.sha256.slice(0, 12)}, recorded ${recorded!.slice(0, 12)}).`);
    }
  }
  if (state.status !== "completed") downgrade(`The run is ${state.status}; only completed runs have a recorded report digest.`);
  else if (!recorded) downgrade("No report digest was recorded at completion (run made by an older TT version).");
  else if (files.length === 0) downgrade("The recorded report file is missing.");
  if (state.status === "completed" && report && !report.provenance) downgrade("The report has no provenance block (run made by an older TT version).");
  return {
    run_id: state.id,
    status: state.status,
    verdict,
    findings,
    chain,
    report: { recorded_sha256: recorded, files },
    timeline: buildTimeline(events),
    provenance: report?.provenance ?? null,
  };
}
