import { createHash, randomUUID } from "node:crypto";
import { createReadStream } from "node:fs";
import { mkdir, open, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { hostname } from "node:os";
import { join } from "node:path";

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

export type ChainStatus = "verified" | "legacy" | "mixed" | "broken" | "empty";

export interface ChainVerification {
  status: ChainStatus;
  events: number;
  chained_events: number;
  /** Events recorded before the chain existed (older TT versions). */
  legacy_events: number;
  /** Unchained events written after chaining began, e.g. by an older TT process still running. */
  unchained_after_chain: number;
  head_hash: string | null;
  break?: { index: number; seq: number | null; reason: string };
}

/**
 * Verify an event list. Chained events must form one unbroken chain; events
 * without hashes (written by older TT versions) are skipped and counted, so
 * they make a log unverifiable rather than tampered.
 */
export function verifyEventChain(events: ReadonlyArray<Record<string, unknown>>): ChainVerification {
  let legacy = 0;
  let unchainedAfter = 0;
  let previous: (ChainedFields & Record<string, unknown>) | undefined;
  const result = (status: ChainStatus, brk?: ChainVerification["break"]): ChainVerification => ({
    status,
    events: events.length,
    chained_events: events.filter((event) => (event as ChainedFields).hash !== undefined).length,
    legacy_events: legacy,
    unchained_after_chain: unchainedAfter,
    head_hash: (previous?.hash as string | undefined) ?? null,
    ...(brk ? { break: brk } : {}),
  });
  for (const [index, event] of events.entries()) {
    const chained = event as ChainedFields & Record<string, unknown>;
    if (chained.hash === undefined) {
      if (previous) unchainedAfter += 1;
      else legacy += 1;
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
  if (!previous) return result("legacy");
  if (unchainedAfter > 0) return result("mixed");
  return result(legacy > 0 ? "legacy" : "verified");
}

export async function sha256File(path: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}

interface LockOwner {
  token: string;
  pid: number;
  host: string;
}

function processAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === "EPERM";
  }
}

async function readLockOwner(lockPath: string): Promise<LockOwner | null> {
  try {
    return JSON.parse(await readFile(join(lockPath, "owner.json"), "utf8")) as LockOwner;
  } catch {
    return null;
  }
}

/**
 * A lock is abandoned when its owner process is gone. Owners hold the lock
 * only for one read and append, so a live owner (even a suspended laptop
 * process) is always waited for rather than broken.
 */
async function abandonedLockOwner(lockPath: string, orphanMs: number, remoteMs: number): Promise<LockOwner | "ownerless" | null> {
  const metadata = await stat(lockPath).catch(() => null);
  if (!metadata) return null;
  const owner = await readLockOwner(lockPath);
  const age = Date.now() - metadata.mtimeMs;
  // Created but the owner record never written: the writer died in between.
  if (!owner) return age > orphanMs ? "ownerless" : null;
  if (owner.host === hostname()) return processAlive(owner.pid) ? null : owner;
  return age > remoteMs ? owner : null;
}

async function breakAbandonedLock(lockPath: string, abandoned: LockOwner | "ownerless"): Promise<void> {
  const moved = `${lockPath}.abandoned-${randomUUID()}`;
  try {
    await rename(lockPath, moved);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return; // another waiter broke it
    throw error;
  }
  const owner = await readLockOwner(moved);
  const expected = abandoned === "ownerless" ? undefined : abandoned.token;
  if (owner?.token !== expected) {
    // A new owner took the lock between our check and the rename: give it back.
    await rename(moved, lockPath).catch(() => undefined);
  }
  await rm(moved, { recursive: true, force: true });
}

/**
 * Cross-process mutual exclusion for appending to one event log. `mkdir` is
 * atomic; the owner record names the holder so a crashed holder's lock is
 * reclaimed as soon as its process is gone, and release only removes a lock
 * this holder still owns.
 */
export async function withAppendLock<T>(
  lockPath: string,
  action: () => Promise<T>,
  options: { timeoutMs?: number; orphanMs?: number; remoteMs?: number } = {},
): Promise<T> {
  const timeoutMs = options.timeoutMs ?? 60_000;
  const token = randomUUID();
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    try {
      await mkdir(lockPath, { mode: 0o700 });
      await writeFile(join(lockPath, "owner.json"), JSON.stringify({ token, pid: process.pid, host: hostname() }), { mode: 0o600 });
      break;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
      const abandoned = await abandonedLockOwner(lockPath, options.orphanMs ?? 5_000, options.remoteMs ?? 600_000);
      if (abandoned) {
        await breakAbandonedLock(lockPath, abandoned);
        continue;
      }
      if (Date.now() > deadline) throw new Error(`Timed out waiting for the run event log lock: ${lockPath}`);
      await new Promise((resolve) => setTimeout(resolve, 20));
    }
  }
  try {
    return await action();
  } finally {
    if ((await readLockOwner(lockPath))?.token === token) await rm(lockPath, { recursive: true, force: true });
  }
}

/**
 * Read the last parsable JSON line of a JSONL file without reading it all,
 * and whether the file ends mid-line (an interrupted write).
 */
export async function readJsonlTail<T>(
  path: string,
  accept: (row: T) => boolean = () => true,
): Promise<{ last: T | undefined; partial: boolean }> {
  let handle;
  try {
    handle = await open(path, "r");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return { last: undefined, partial: false };
    throw error;
  }
  try {
    const size = (await handle.stat()).size;
    if (size === 0) return { last: undefined, partial: false };
    let window = Math.min(size, 64 * 1024);
    for (;;) {
      const buffer = Buffer.alloc(window);
      await handle.read(buffer, 0, window, size - window);
      const text = buffer.toString("utf8");
      const partial = !text.endsWith("\n");
      const lines = text.split("\n");
      // The first line may be cut by the window unless it starts the file.
      const complete = window === size ? lines : lines.slice(1);
      for (const line of complete.reverse()) {
        if (!line.trim()) continue;
        let row: T;
        try {
          row = JSON.parse(line) as T;
        } catch {
          continue; // a torn line from an interrupted write
        }
        if (accept(row)) return { last: row, partial };
      }
      if (window === size) return { last: undefined, partial };
      window = Math.min(size, window * 4);
    }
  } finally {
    await handle.close();
  }
}

/** Parse JSONL, skipping and counting lines an interrupted write left unreadable. */
export function parseJsonlTolerant<T>(text: string): { rows: T[]; unreadable: number } {
  const rows: T[] = [];
  let unreadable = 0;
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim()) continue;
    try {
      rows.push(JSON.parse(line) as T);
    } catch {
      unreadable += 1;
    }
  }
  return { rows, unreadable };
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

interface DigestCheck {
  path: string;
  sha256: string;
  matches_recorded: boolean | null;
}

export interface RunAudit {
  run_id: string;
  status: string;
  verdict: AuditVerdict;
  findings: string[];
  chain: ChainVerification & { unreadable_lines: number };
  report: { recorded_sha256: string | null; files: DigestCheck[] };
  artifact_manifest: { recorded_sha256: string | null; file: DigestCheck | null };
  timeline: TimelineEntry[];
  provenance: unknown;
}

type AuditEvent = { stage: string; status: string; message: string; occurred_at: string; details?: Record<string, unknown> };

/**
 * Check a run's evidence: the event chain, and that the report and artifact
 * manifest still match the digests recorded when the run completed.
 * `verified` needs all of it; records from older TT versions or unfinished
 * runs are `incomplete`; any mismatch or truncation is `tampered`.
 */
export async function auditRun(store: {
  getRun(id: string): Promise<{ id: string; status: string; artifact_dir?: string }>;
  getRunEventLog(id: string): Promise<{ events: AuditEvent[]; unreadable_lines: number }>;
  getRunReportPaths(id: string): Promise<string[]>;
  getRunReport(id: string): Promise<unknown>;
}, runId: string): Promise<RunAudit> {
  const state = await store.getRun(runId);
  const log = await store.getRunEventLog(state.id);
  const events = log.events;
  const chain = verifyEventChain(events as unknown as Array<Record<string, unknown>>);
  const completion = [...events].reverse().find((event) => event.stage === "completed");
  const recorded = typeof completion?.details?.report_sha256 === "string" ? completion.details.report_sha256 : null;
  const recordedManifest = typeof completion?.details?.artifact_manifest_sha256 === "string"
    ? completion.details.artifact_manifest_sha256
    : null;
  const files = await Promise.all((await store.getRunReportPaths(state.id)).map(async (path) => {
    const sha256 = await sha256File(path);
    return { path, sha256, matches_recorded: recorded ? sha256 === recorded : null };
  }));
  const manifestPath = state.artifact_dir ? join(state.artifact_dir, "artifact-manifest.json") : null;
  const manifestFile = manifestPath && (await stat(manifestPath).catch(() => null))?.isFile()
    ? await sha256File(manifestPath).then((sha256) => ({
      path: manifestPath,
      sha256,
      matches_recorded: recordedManifest ? sha256 === recordedManifest : null,
    }))
    : null;
  const report = await store.getRunReport(state.id).catch(() => null) as { provenance?: unknown } | null;

  const findings: string[] = [];
  let verdict: AuditVerdict = "verified";
  const tampered = (finding: string) => {
    findings.push(finding);
    verdict = "tampered";
  };
  const incomplete = (finding: string) => {
    findings.push(finding);
    if (verdict === "verified") verdict = "incomplete";
  };
  const completed = state.status === "completed";
  if (chain.status === "broken") {
    tampered(`Event log chain is broken at event ${chain.break!.index}: ${chain.break!.reason}`);
  } else if (chain.status === "empty") {
    if (completed) tampered("The run is completed but its event log is missing or empty.");
    else incomplete("The run has no events.");
  } else if (chain.status === "legacy") {
    incomplete(`${chain.legacy_events} event(s) were recorded before event chaining (older TT version) and cannot be verified.`);
  } else if (chain.status === "mixed") {
    incomplete(`${chain.unchained_after_chain} event(s) were written without hashes after chaining began, e.g. by an older TT process.`);
  }
  if (log.unreadable_lines > 0) incomplete(`${log.unreadable_lines} event line(s) are unreadable, likely from an interrupted write.`);
  for (const file of files) {
    if (file.matches_recorded === false) {
      tampered(`Report ${file.path} changed after the run completed (sha256 ${file.sha256.slice(0, 12)}, recorded ${recorded!.slice(0, 12)}).`);
    }
  }
  if (manifestFile?.matches_recorded === false) {
    tampered(`Artifact manifest ${manifestFile.path} changed after the run completed.`);
  }
  if (!completed) {
    incomplete(`The run is ${state.status}; only completed runs have recorded report digests.`);
  } else if (!recorded) {
    // A fully chained log always ends with a completion event carrying the digest.
    if (chain.status === "verified") tampered("The completion event is missing from a chained event log; the log was truncated.");
    else if (chain.status !== "broken" && chain.status !== "empty") incomplete("No report digest was recorded at completion (older TT version).");
  } else if (files.length === 0) {
    incomplete("The recorded report file is missing.");
  }
  if (completed && recordedManifest && !manifestFile) incomplete("The artifact manifest recorded at completion is missing.");
  if (completed && report && !report.provenance) incomplete("The report has no provenance block (older TT version).");
  return {
    run_id: state.id,
    status: state.status,
    verdict,
    findings,
    chain: { ...chain, unreadable_lines: log.unreadable_lines },
    report: { recorded_sha256: recorded, files },
    artifact_manifest: { recorded_sha256: recordedManifest, file: manifestFile },
    timeline: buildTimeline(events),
    provenance: report?.provenance ?? null,
  };
}
