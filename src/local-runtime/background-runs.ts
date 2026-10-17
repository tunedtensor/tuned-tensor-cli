import { spawn as nodeSpawn, type ChildProcess, type SpawnOptions } from "node:child_process";
import { randomUUID } from "node:crypto";
import { closeSync, openSync } from "node:fs";
import { mkdir, open, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { hostname, userInfo } from "node:os";
import { basename, dirname, join } from "node:path";
import { getTunedTensorHome } from "../paths.js";
import { chainEvent, parseJsonlTolerant, readJsonlTail, verifyEventChain, withAppendLock, type ChainVerification } from "./audit.js";
import type { LiveUsage } from "./live-usage.js";
import { writePrivateJsonAtomic } from "./store.js";

/**
 * Real pipeline runs started in the background, by the agent after approval or
 * by `tt pipeline start`. Each run keeps a durable record so it can be listed,
 * inspected, cancelled and audited after the shell that started it exits:
 *
 *   record.json  why it ran, who approved it, the reviewed choices, and the outcome
 *   spec.json    the sealed spec bytes the run consumed
 *   events.jsonl hash-chained lifecycle journal (see audit.ts)
 *   output.log   the pipeline's progress and process logs
 *   result.json  the pipeline's final JSON result
 *
 * A detached supervisor (`tt pipeline supervise <id>`) owns the pipeline
 * process and writes the outcome, so the conversation never waits on it.
 */

export type BackgroundRunStatus =
  | "starting"
  | "running"
  | "cancel_requested"
  | "succeeded"
  | "failed"
  | "cancelled"
  | "lost";

const ACTIVE: ReadonlySet<BackgroundRunStatus> = new Set(["starting", "running", "cancel_requested"]);
export const DEFAULT_MAX_PARALLEL_RUNS = 1;
const MAX_PARALLEL_RUNS_LIMIT = 8;
/** A launcher records the supervisor PID immediately; this only covers a crash in between. */
const STARTING_GRACE_MS = 60_000;
const MAX_LOG_TAIL_BYTES = 64 * 1024;
const MAX_REASON_LENGTH = 1_000;

export interface BackgroundRunRequester {
  user: string;
  host: string;
  via: "agent" | "cli";
  thread_id?: string;
  action_id?: string;
  dry_run_action_id?: string;
}

export interface BackgroundRunChoices {
  engine: "adapter" | "foundation" | "decision";
  spec_path: string;
  spec_sha256: string;
  config_path: string | null;
  config_sha256: string | null;
  steps: Array<{ id: string; uses: string; target: string }>;
  /** "local" or "aws:<instance-id>"; runs sharing a target compete for the same GPU. */
  target: string;
  workspace_root: string;
  base_model?: string;
}

export interface BackgroundRunOutcome {
  exit_code?: number | null;
  signal?: string | null;
  pipeline_status?: string;
  local_run_id?: string;
  model_id?: string;
  report_path?: string;
  output_dir?: string;
  metrics?: Record<string, unknown>;
  error?: string;
}

export interface BackgroundRunRecord {
  version: 1;
  id: string;
  status: BackgroundRunStatus;
  reason: string;
  requested_by: BackgroundRunRequester;
  approved_at: string;
  choices: BackgroundRunChoices;
  resources_at_launch?: ResourceSummary;
  process?: { host: string; supervisor_pid?: number; pid?: number };
  cancel?: { requested_at: string; reason: string; requested_by: BackgroundRunRequester };
  outcome?: BackgroundRunOutcome;
  /** When the agent was told about the terminal outcome in its conversation. */
  reported_at?: string;
  created_at: string;
  updated_at: string;
  started_at?: string;
  finished_at?: string;
}

export interface BackgroundRunEvent {
  run_id: string;
  type: "requested" | "started" | "cancel_requested" | "finished" | "lost";
  status: BackgroundRunStatus;
  message: string;
  details?: Record<string, unknown>;
  occurred_at: string;
  seq?: number;
  prev_hash?: string | null;
  hash?: string;
}

export interface ResourceSummary {
  sampled_at: string;
  cpu_utilization_percent?: number;
  cpu_cores: number;
  memory_used_bytes: number;
  memory_total_bytes: number;
  gpus: Array<{
    index: number;
    name: string;
    utilization_percent?: number;
    memory_used_bytes?: number;
    memory_total_bytes?: number;
  }>;
  gpu_note?: string;
}

export class BackgroundRunError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "BackgroundRunError";
  }
}

export function backgroundRunsRoot(env: NodeJS.ProcessEnv = process.env): string {
  return join(getTunedTensorHome(env), "background-runs");
}

export function maxParallelRuns(env: NodeJS.ProcessEnv = process.env): number {
  const raw = env.TT_MAX_PARALLEL_RUNS?.trim();
  if (!raw) return DEFAULT_MAX_PARALLEL_RUNS;
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 1 || value > MAX_PARALLEL_RUNS_LIMIT) {
    throw new BackgroundRunError(`TT_MAX_PARALLEL_RUNS must be an integer from 1 to ${MAX_PARALLEL_RUNS_LIMIT}, got: ${raw}`);
  }
  return value;
}

export function isActiveBackgroundRun(record: Pick<BackgroundRunRecord, "status">): boolean {
  return ACTIVE.has(record.status);
}

export function currentRequester(
  via: BackgroundRunRequester["via"],
  extra: Omit<BackgroundRunRequester, "user" | "host" | "via"> = {},
): BackgroundRunRequester {
  let user = "unknown";
  try { user = userInfo().username || user; } catch { /* Containers may lack a passwd entry. */ }
  return { user, host: hostname(), via, ...extra };
}

export function normalizeReason(reason: unknown): string {
  const text = typeof reason === "string" ? reason.replace(/\s+/g, " ").trim() : "";
  if (!text) throw new BackgroundRunError("A real run needs a reason; it is kept in the run's audit record.");
  if (text.length > MAX_REASON_LENGTH) throw new BackgroundRunError(`The run reason must be at most ${MAX_REASON_LENGTH} characters.`);
  return text;
}

export function summarizeResources(usage: LiveUsage): ResourceSummary {
  return {
    sampled_at: usage.sampled_at,
    ...(usage.cpu.utilization_percent !== undefined ? { cpu_utilization_percent: usage.cpu.utilization_percent } : {}),
    cpu_cores: usage.cpu.cores,
    memory_used_bytes: usage.memory.used_bytes,
    memory_total_bytes: usage.memory.total_bytes,
    gpus: usage.gpus.map((gpu) => ({
      index: gpu.index,
      name: gpu.name,
      ...(gpu.utilization_percent !== undefined ? { utilization_percent: gpu.utilization_percent } : {}),
      ...(gpu.memory_used_bytes !== undefined ? { memory_used_bytes: gpu.memory_used_bytes } : {}),
      ...(gpu.memory_total_bytes !== undefined ? { memory_total_bytes: gpu.memory_total_bytes } : {}),
    })),
    ...(usage.gpu_note ? { gpu_note: usage.gpu_note } : {}),
  };
}

function processAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === "EPERM";
  }
}

function errorCode(error: unknown): string | undefined {
  return (error as NodeJS.ErrnoException | undefined)?.code;
}

/** Strip terminal control sequences so log text is safe to print or hand to a model. */
export function plainLogText(text: string): string {
  return text
    // eslint-disable-next-line no-control-regex
    .replace(/\u001b\[[0-9;?]*[ -/]*[@-~]/g, "")
    // eslint-disable-next-line no-control-regex
    .replace(/\u001b\][^\u0007]*(?:\u0007|\u001b\\)/g, "")
    // eslint-disable-next-line no-control-regex
    .replace(/[\u0000-\u0008\u000b-\u001f\u007f]/g, "");
}

export interface NewBackgroundRun {
  reason: string;
  requested_by: BackgroundRunRequester;
  choices: BackgroundRunChoices;
  sealed_spec: unknown;
  resources_at_launch?: ResourceSummary;
}

export interface BackgroundRunStore {
  root: string;
  paths(id: string): { dir: string; record: string; spec: string; events: string; log: string; result: string };
  create(input: NewBackgroundRun, options?: { maxParallel?: number }): Promise<BackgroundRunRecord>;
  list(): Promise<BackgroundRunRecord[]>;
  get(idOrPrefix: string): Promise<BackgroundRunRecord>;
  active(): Promise<BackgroundRunRecord[]>;
  update(
    id: string,
    mutate: (record: BackgroundRunRecord) => void,
    event?: Omit<BackgroundRunEvent, "run_id" | "status" | "occurred_at">,
  ): Promise<BackgroundRunRecord>;
  requestCancel(idOrPrefix: string, args: { reason: string; requested_by: BackgroundRunRequester }): Promise<BackgroundRunRecord>;
  events(idOrPrefix: string): Promise<{ events: BackgroundRunEvent[]; unreadable_lines: number; chain: ChainVerification }>;
  logTail(idOrPrefix: string, maxLines?: number): Promise<string>;
  markReported(ids: string[]): Promise<void>;
}

export function createBackgroundRunStore(root = backgroundRunsRoot()): BackgroundRunStore {
  const paths = (id: string) => {
    const dir = join(root, id);
    return {
      dir,
      record: join(dir, "record.json"),
      spec: join(dir, "spec.json"),
      events: join(dir, "events.jsonl"),
      log: join(dir, "output.log"),
      result: join(dir, "result.json"),
    };
  };
  const lockPath = (id: string) => join(paths(id).dir, ".lock");

  async function readRecord(id: string): Promise<BackgroundRunRecord> {
    return JSON.parse(await readFile(paths(id).record, "utf8")) as BackgroundRunRecord;
  }

  async function appendEvent(record: BackgroundRunRecord, event: Omit<BackgroundRunEvent, "run_id" | "status" | "occurred_at">): Promise<void> {
    const file = paths(record.id).events;
    const { last } = await readJsonlTail<BackgroundRunEvent>(file);
    const chained = chainEvent({
      run_id: record.id,
      status: record.status,
      occurred_at: record.updated_at,
      ...event,
    } as BackgroundRunEvent & Record<string, unknown>, last as (BackgroundRunEvent & Record<string, unknown>) | undefined);
    const handle = await open(file, "a", 0o600);
    try {
      await handle.write(`${JSON.stringify(chained)}\n`);
    } finally {
      await handle.close();
    }
  }

  async function update(
    id: string,
    mutate: (record: BackgroundRunRecord) => void,
    event?: Omit<BackgroundRunEvent, "run_id" | "status" | "occurred_at">,
  ): Promise<BackgroundRunRecord> {
    return await withAppendLock(lockPath(id), async () => {
      const record = await readRecord(id);
      mutate(record);
      record.updated_at = new Date().toISOString();
      await writePrivateJsonAtomic(paths(id).record, record);
      if (event) await appendEvent(record, event);
      return record;
    });
  }

  /** Mark a run whose supervisor disappeared (crash, reboot) instead of reporting it as running forever. */
  async function reconcile(record: BackgroundRunRecord): Promise<BackgroundRunRecord> {
    const missingSupervisor = (current: BackgroundRunRecord) => {
      if (!isActiveBackgroundRun(current)) return false;
      const owner = current.process;
      if (owner?.host && owner.host !== hostname()) return false;
      const pid = owner?.supervisor_pid;
      return pid === undefined
        ? Date.now() - Date.parse(current.created_at) >= STARTING_GRACE_MS
        : !processAlive(pid);
    };
    if (!missingSupervisor(record)) return record;
    return await withAppendLock(lockPath(record.id), async () => {
      // A launcher or supervisor may have saved its PID or final outcome since
      // the read above. Recheck under the same lock before declaring it lost.
      const current = await readRecord(record.id);
      if (!missingSupervisor(current)) return current;
      current.status = "lost";
      current.finished_at = new Date().toISOString();
      current.updated_at = current.finished_at;
      current.outcome = {
        ...current.outcome,
        error: "The background supervisor stopped without recording an outcome (for example after a crash or reboot). Inspect the log and the local run store before starting again.",
      };
      await writePrivateJsonAtomic(paths(current.id).record, current);
      await appendEvent(current, { type: "lost", message: "Supervisor process is gone; outcome unknown." });
      return current;
    });
  }

  async function listRaw(): Promise<BackgroundRunRecord[]> {
    let entries: string[];
    try {
      entries = await readdir(root);
    } catch (error) {
      if (errorCode(error) === "ENOENT") return [];
      throw error;
    }
    const records: BackgroundRunRecord[] = [];
    for (const entry of entries) {
      if (!/^[0-9a-f-]{36}$/.test(entry)) continue;
      try {
        records.push(await readRecord(entry));
      } catch {
        // A directory created by an interrupted launch has no record yet.
      }
    }
    return records.sort((a, b) => b.created_at.localeCompare(a.created_at));
  }

  async function list(): Promise<BackgroundRunRecord[]> {
    return await Promise.all((await listRaw()).map(reconcile));
  }

  async function resolveId(idOrPrefix: string): Promise<string> {
    const id = idOrPrefix.trim().toLowerCase();
    if (!/^[0-9a-f-]{4,36}$/.test(id)) throw new BackgroundRunError(`Invalid background run ID: ${idOrPrefix}`);
    const matches = (await listRaw()).filter((record) => record.id.startsWith(id));
    if (matches.length === 0) throw new BackgroundRunError(`No background run matches ${idOrPrefix}.`);
    if (matches.length > 1) throw new BackgroundRunError(`Background run prefix ${idOrPrefix} is ambiguous.`);
    return matches[0]!.id;
  }

  async function get(idOrPrefix: string): Promise<BackgroundRunRecord> {
    return await reconcile(await readRecord(await resolveId(idOrPrefix)));
  }

  return {
    root,
    paths,
    update,
    list,
    get,

    async active() {
      return (await list()).filter(isActiveBackgroundRun);
    },

    async create(input, options = {}) {
      const limit = options.maxParallel ?? DEFAULT_MAX_PARALLEL_RUNS;
      await mkdir(root, { recursive: true, mode: 0o700 });
      // One launch at a time so two approvals cannot both pass the limit check.
      return await withAppendLock(join(root, ".launch.lock"), async () => {
        const active = (await list()).filter(isActiveBackgroundRun);
        if (active.length >= limit) {
          throw new BackgroundRunError(
            `${active.length} background run${active.length === 1 ? " is" : "s are"} already active (limit ${limit}: ${active.map((run) => run.id.slice(0, 8)).join(", ")}). `
            + "Wait for it to finish, cancel it, or raise TT_MAX_PARALLEL_RUNS when the hardware can hold both.",
          );
        }
        const now = new Date().toISOString();
        const record: BackgroundRunRecord = {
          version: 1,
          id: randomUUID(),
          status: "starting",
          reason: normalizeReason(input.reason),
          requested_by: input.requested_by,
          approved_at: now,
          choices: input.choices,
          ...(input.resources_at_launch ? { resources_at_launch: input.resources_at_launch } : {}),
          process: { host: hostname() },
          created_at: now,
          updated_at: now,
        };
        const files = paths(record.id);
        await mkdir(files.dir, { mode: 0o700 });
        await writeFile(files.spec, `${JSON.stringify(input.sealed_spec, null, 2)}\n`, { mode: 0o600, flag: "wx" });
        await writePrivateJsonAtomic(files.record, record);
        await appendEvent(record, {
          type: "requested",
          message: `Approved by ${record.requested_by.user}@${record.requested_by.host} via ${record.requested_by.via}.`,
          details: { reason: record.reason, spec_sha256: record.choices.spec_sha256, target: record.choices.target },
        });
        return record;
      });
    },

    async requestCancel(idOrPrefix, args) {
      const id = await resolveId(idOrPrefix);
      const reason = normalizeReason(args.reason);
      let wasRunning = false;
      const record = await update(id, (current) => {
        wasRunning = current.status === "running";
        if (!isActiveBackgroundRun(current)) {
          throw new BackgroundRunError(`Background run ${id.slice(0, 8)} is already ${current.status}; nothing to cancel.`);
        }
        current.status = "cancel_requested";
        current.cancel = { requested_at: new Date().toISOString(), reason, requested_by: args.requested_by };
      }, {
        type: "cancel_requested",
        message: `Cancellation requested by ${args.requested_by.user}@${args.requested_by.host} via ${args.requested_by.via}.`,
        details: { reason },
      });
      // The supervisor forwards SIGTERM to the pipeline's process group, which
      // stops remote GPU work and records the cancellation in the run store.
      // Only a running supervisor has its handlers installed; while starting it
      // checks for this request itself before and after spawning the pipeline.
      const pid = record.process?.supervisor_pid;
      if (wasRunning && pid !== undefined && record.process?.host === hostname()) {
        try { process.kill(pid, "SIGTERM"); } catch { /* Already exited; its outcome is recorded. */ }
      }
      return record;
    },

    async events(idOrPrefix) {
      const id = await resolveId(idOrPrefix);
      let text = "";
      try {
        text = await readFile(paths(id).events, "utf8");
      } catch (error) {
        if (errorCode(error) !== "ENOENT") throw error;
      }
      const { rows, unreadable } = parseJsonlTolerant<BackgroundRunEvent>(text);
      return {
        events: rows,
        unreadable_lines: unreadable,
        chain: verifyEventChain(rows as unknown as Array<Record<string, unknown>>),
      };
    },

    async logTail(idOrPrefix, maxLines = 40) {
      const id = await resolveId(idOrPrefix);
      let handle;
      try {
        handle = await open(paths(id).log, "r");
      } catch (error) {
        if (errorCode(error) === "ENOENT") return "";
        throw error;
      }
      try {
        const size = (await handle.stat()).size;
        const length = Math.min(size, MAX_LOG_TAIL_BYTES);
        const buffer = Buffer.alloc(length);
        await handle.read(buffer, 0, length, size - length);
        const lines = plainLogText(buffer.toString("utf8").replace(/\r(?!\n)/g, "\n")).split(/\r?\n/);
        if (length < size) lines.shift();
        return lines.filter((line) => line.trim()).slice(-Math.max(1, maxLines)).join("\n");
      } finally {
        await handle.close();
      }
    },

    async markReported(ids) {
      const reportedAt = new Date().toISOString();
      for (const id of ids) {
        await update(id, (record) => { record.reported_at = reportedAt; });
      }
    },
  };
}

// ---------------------------------------------------------------------------
// Outcome summaries

function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

function text(value: unknown, max = 500): string | undefined {
  return typeof value === "string" && value ? value.slice(0, max) : undefined;
}

/** Finite numbers and booleans from one level of an object; keeps model-facing metrics small. */
function scalarMetrics(value: unknown, maxKeys = 16): Record<string, number | boolean> | undefined {
  const source = record(value);
  if (!source) return undefined;
  const result: Record<string, number | boolean> = {};
  for (const [key, entry] of Object.entries(source)) {
    if (Object.keys(result).length >= maxKeys) break;
    if ((typeof entry === "number" && Number.isFinite(entry)) || typeof entry === "boolean") result[key] = entry;
  }
  return Object.keys(result).length ? result : undefined;
}

function compact<T extends Record<string, unknown>>(value: T): T {
  return Object.fromEntries(Object.entries(value).filter(([, entry]) => entry !== undefined)) as T;
}

/** Reduce a `tt --json pipeline run` result to the outcome kept in the record. */
export function summarizePipelineResult(result: unknown): BackgroundRunOutcome {
  const value = record(result);
  if (!value) return {};
  const error = text(value.error) ?? text(record(value.error)?.message);
  const request = record(value.request);
  if (request) {
    const report = record(value.report);
    const runId = text(request.run_id, 64);
    const outputs = Object.values(record(value.outputs) ?? {}).map(record);
    const trained = Boolean(text(report?.fine_tuned_model_id)
      || outputs.some((output) => text(output?.modelArtifactUri)))
      && record(record(report?.training)?.metrics)?.dry_run !== true;
    return compact({
      pipeline_status: text(value.status, 40),
      local_run_id: runId,
      // Reports identify the artifact URI; CLI consumers need the store ID.
      // Evaluation-only recipes have a run ID but never register a model.
      model_id: trained && runId ? `local-${runId}` : undefined,
      report_path: text(value.reportPath),
      output_dir: text(value.artifactDir),
      metrics: report ? compact({
        baseline: scalarMetrics(report.baseline),
        candidate: scalarMetrics(report.candidate),
        comparison: scalarMetrics(report.comparison),
        general_regression_passed: typeof record(report.general_regression)?.passed === "boolean"
          ? record(report.general_regression)!.passed
          : undefined,
      }) : undefined,
      error,
    });
  }
  const steps = Array.isArray(value.steps) ? value.steps.slice(0, 16) : [];
  const metrics: Record<string, unknown> = {};
  for (const step of steps) {
    const entry = record(step);
    const id = text(entry?.id, 64);
    if (!entry || !id) continue;
    const stepMetrics = compact({
      ...scalarMetrics(entry.metrics),
      ...scalarMetrics(entry.decision_metrics),
      ...(entry.comparison ? { comparison: scalarMetrics(entry.comparison) } : {}),
    });
    if (Object.keys(stepMetrics).length) metrics[id] = stepMetrics;
  }
  return compact({
    pipeline_status: text(value.status, 40),
    report_path: text(value.report_path),
    output_dir: text(value.run_dir) ?? text(value.model_dir),
    model_id: text(value.model_dir),
    metrics: Object.keys(metrics).length ? metrics : undefined,
    error,
  });
}

/** Compact view for agent tools and `tt pipeline list`: no sealed spec, no full log. */
export function describeBackgroundRun(run: BackgroundRunRecord): Record<string, unknown> {
  const end = run.finished_at ? Date.parse(run.finished_at) : Date.now();
  const start = run.started_at ? Date.parse(run.started_at) : undefined;
  return compact({
    id: run.id,
    status: run.status,
    reason: run.reason,
    requested_by: `${run.requested_by.user}@${run.requested_by.host} via ${run.requested_by.via}`,
    engine: run.choices.engine,
    spec_path: run.choices.spec_path,
    spec_sha256: run.choices.spec_sha256,
    target: run.choices.target,
    steps: run.choices.steps.map((step) => step.id),
    created_at: run.created_at,
    started_at: run.started_at,
    finished_at: run.finished_at,
    elapsed_seconds: start !== undefined ? Math.max(0, Math.round((end - start) / 1000)) : undefined,
    cancel_reason: run.cancel?.reason,
    outcome: run.outcome,
  });
}

// ---------------------------------------------------------------------------
// Launch and supervision

export type SpawnLike = (command: string, args: string[], options: SpawnOptions) => ChildProcess;

/**
 * Start the detached supervisor and return as soon as its PID is recorded.
 * The supervisor outlives the shell, so closing the conversation never stops
 * or changes a run.
 */
export async function launchBackgroundSupervisor(args: {
  store: BackgroundRunStore;
  run: BackgroundRunRecord;
  entrypoint: string;
  env?: NodeJS.ProcessEnv;
  execPath?: string;
  spawn?: SpawnLike;
}): Promise<BackgroundRunRecord> {
  const spawnImpl = args.spawn ?? (nodeSpawn as SpawnLike);
  let child: ChildProcess;
  try {
    child = spawnImpl(args.execPath ?? process.execPath, [args.entrypoint, "pipeline", "supervise", args.run.id], {
      cwd: args.run.choices.workspace_root,
      env: args.env ?? process.env,
      detached: process.platform !== "win32",
      stdio: "ignore",
    });
    if (child.pid === undefined) {
      await new Promise<void>((resolve, reject) => {
        child.once("error", reject);
        child.once("spawn", () => resolve());
      });
    }
  } catch (error) {
    await args.store.update(args.run.id, (current) => {
      current.status = "failed";
      current.finished_at = new Date().toISOString();
      current.outcome = { error: `Could not start the background supervisor: ${(error as Error).message}` };
    }, { type: "finished", message: "Supervisor failed to start." });
    throw new BackgroundRunError(`Could not start the background run: ${(error as Error).message}`);
  }
  child.unref();
  return await args.store.update(args.run.id, (current) => {
    current.process = { ...current.process, host: hostname(), supervisor_pid: child.pid };
  });
}

export interface SuperviseOptions {
  store: BackgroundRunStore;
  id: string;
  entrypoint: string;
  env?: NodeJS.ProcessEnv;
  execPath?: string;
  spawn?: SpawnLike;
  /** The workspace-adjacent sealed spec; defaults to a hidden snapshot beside the reviewed spec. */
  snapshotPath?: (run: BackgroundRunRecord) => string;
}

export function workspaceSnapshotPath(run: BackgroundRunRecord): string {
  // Beside the reviewed spec so relative paths, output directories and the
  // spec's tunedtensor.lock.json resolve exactly as for a direct run (see
  // specLockPath and resolveProjectConfig).
  const specFile = join(run.choices.workspace_root, run.choices.spec_path);
  return join(dirname(specFile), `.${basename(specFile)}.tt-agent-${run.id}.json`);
}

/** Body of `tt pipeline supervise <id>`: run the sealed pipeline and record its outcome. */
export async function superviseBackgroundRun(options: SuperviseOptions): Promise<BackgroundRunRecord> {
  const { store } = options;
  const run = await store.get(options.id);
  const finishEarly = async (status: "cancelled" | "failed", error: string) => await store.update(run.id, (current) => {
    current.status = status;
    current.finished_at = new Date().toISOString();
    current.outcome = { ...current.outcome, error };
  }, { type: "finished", message: error });
  // Cancelled before the supervisor could record itself: nothing started.
  if (run.status === "cancel_requested") return await finishEarly("cancelled", "Cancelled before the pipeline started.");
  if (run.status !== "starting") throw new BackgroundRunError(`Background run ${run.id.slice(0, 8)} is ${run.status}, not starting.`);
  const files = store.paths(run.id);
  const snapshot = (options.snapshotPath ?? workspaceSnapshotPath)(run);
  let logFd: number;
  let resultFd: number;
  try {
    await writeFile(snapshot, await readFile(files.spec), { mode: 0o600, flag: "wx" });
    logFd = openSync(files.log, "a", 0o600);
    resultFd = openSync(files.result, "w", 0o600);
  } catch (error) {
    await rm(snapshot, { force: true }).catch(() => undefined);
    return await finishEarly("failed", `Could not prepare the run: ${(error as Error).message}`);
  }
  let child: ChildProcess | undefined;
  // Mutated from event callbacks; an object keeps TypeScript from narrowing them away.
  const state: {
    stopSignal?: NodeJS.Signals;
    exitCode: number | null;
    exitSignal: NodeJS.Signals | null;
    spawnError?: Error;
  } = { exitCode: null, exitSignal: null };
  const stop = (signal: NodeJS.Signals) => {
    state.stopSignal ??= signal;
    if (!child?.pid) return;
    try {
      if (process.platform !== "win32") process.kill(-child.pid, "SIGTERM");
      else child.kill("SIGTERM");
    } catch { /* Already exited. */ }
  };
  const onSignal = (signal: NodeJS.Signals) => stop(signal);
  process.on("SIGTERM", onSignal);
  process.on("SIGINT", onSignal);
  if (process.platform !== "win32") process.on("SIGHUP", onSignal);
  try {
    const spawnImpl = options.spawn ?? (nodeSpawn as SpawnLike);
    child = spawnImpl(options.execPath ?? process.execPath, [options.entrypoint, "--json", "pipeline", "run", "--spec", snapshot], {
      cwd: run.choices.workspace_root,
      env: options.env ?? process.env,
      detached: process.platform !== "win32",
      stdio: ["ignore", resultFd, logFd],
    });
    const closed = new Promise<void>((resolve) => {
      child!.once("error", (error) => { state.spawnError = error; resolve(); });
      child!.once("close", (code, signal) => { state.exitCode = code; state.exitSignal = signal; resolve(); });
    });
    if (child.pid !== undefined) {
      await store.update(run.id, (current) => {
        if (current.status === "starting") current.status = "running";
        current.started_at = new Date().toISOString();
        current.process = { ...current.process, host: hostname(), supervisor_pid: process.pid, pid: child!.pid };
      }, { type: "started", message: `Pipeline process ${child.pid} started.` });
      // A cancellation that arrived while starting has no PID to signal yet.
      if ((await store.get(run.id)).status === "cancel_requested") stop("SIGTERM");
    }
    await closed;
  } finally {
    process.off("SIGTERM", onSignal);
    process.off("SIGINT", onSignal);
    if (process.platform !== "win32") process.off("SIGHUP", onSignal);
    closeSync(logFd);
    closeSync(resultFd);
    await rm(snapshot, { force: true });
  }

  let parsed: unknown;
  try {
    const raw = (await readFile(files.result, "utf8")).trim();
    if (raw) parsed = JSON.parse(raw);
  } catch { /* A crashed pipeline can leave partial output; the log keeps the detail. */ }
  const summary = summarizePipelineResult(parsed);
  const { exitCode, exitSignal, spawnError, stopSignal } = state;
  const lastLogLine = spawnError ? undefined : (await store.logTail(run.id, 1).catch(() => "")) || undefined;
  return await store.update(run.id, (current) => {
    const cancelled = current.status === "cancel_requested" || stopSignal !== undefined;
    const succeeded = !cancelled && !spawnError && exitCode === 0;
    current.status = cancelled ? "cancelled" : succeeded ? "succeeded" : "failed";
    current.finished_at = new Date().toISOString();
    current.outcome = compact({
      ...summary,
      exit_code: exitCode,
      signal: exitSignal,
      error: succeeded ? undefined : summary.error ?? spawnError?.message ?? (cancelled ? "Cancelled." : lastLogLine ?? `Pipeline exited with code ${exitCode ?? "unknown"}.`),
    });
  }, {
    type: "finished",
    message: spawnError ? `Pipeline failed to start: ${spawnError.message}` : `Pipeline exited with code ${exitCode ?? "none"}${exitSignal ? ` (${exitSignal})` : ""}.`,
  });
}
