import { createHash } from "node:crypto";
import { existsSync, lstatSync, readFileSync } from "node:fs";
import { isAbsolute, join, relative } from "node:path";
import { Type } from "typebox";
import type { AgentAction } from "./agent-client.js";
import {
  labelingDownloadDir,
  labelingDownloadFiles,
  assertLabelDownloadPaths,
  readLocalAdapterSpec,
  validateUnlabeledFile,
} from "./commands/label.js";
import { canonicalWorkspace, fingerprintWorkspace } from "./local-spec-workspace.js";
import { inspectLocalSpec, SpecWorkspaceMismatchError } from "./spec-workspace.js";

/**
 * Agent workflow for hosted teacher labeling. The model can only inspect jobs
 * and prepare two proposals; approval runs the same code as the CLI:
 *
 *   start_label_job     push the local spec (as `tt cloud push`), upload the
 *                       raw file and start the labeling workflow.
 *   download_label_job  write a finished job's rows as local training and
 *                       validation chat JSONL (as `tt cloud label download`).
 */
export const LABELING_OPERATIONS = ["start_label_job", "download_label_job"] as const;

export interface StartLabelJobArgs {
  workspaceRoot: string;
  file: string;
  fileBytes: Buffer<ArrayBuffer>;
  format: "jsonl" | "csv";
  inputColumn: string | null;
  name: string | null;
  specPath: string;
}

export interface DownloadLabelJobArgs {
  workspaceRoot: string;
  jobId: string;
  specPath: string;
  includeUnreviewed: boolean;
  outDir: string;
  validationRatio: number;
  overwrite: boolean;
}

/** Account operations an approved labeling action performs; built from the CLI's TT credentials. */
export interface AgentLabelingExecutor {
  start(args: StartLabelJobArgs): Promise<unknown>;
  download(args: DownloadLabelJobArgs): Promise<unknown>;
}

export const LabelJobInput = Type.Object({
  file: Type.String({
    minLength: 1,
    maxLength: 500,
    description: "Workspace-relative .jsonl ({\"input\": ...} per line) or .csv file of raw inputs",
  }),
  input_column: Type.Optional(Type.String({ minLength: 1, maxLength: 200, description: "CSV column holding the input text" })),
  name: Type.Optional(Type.String({ minLength: 1, maxLength: 120 })),
  spec_path: Type.Optional(Type.String({ minLength: 1, maxLength: 500, description: "Workspace-relative tunedtensor.json; defaults to tunedtensor.json" })),
}, { additionalProperties: false });

export const LabelDownloadInput = Type.Object({
  job_id: Type.String({ pattern: "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-8][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$", description: "Full labeling job UUID" }),
  include_unreviewed: Type.Optional(Type.Boolean({ description: "Also use teacher drafts nobody accepted or edited. Ask the user." })),
  validation_ratio: Type.Optional(Type.Number({ minimum: 0.05, maximum: 0.5, default: 0.1 })),
  overwrite: Type.Optional(Type.Boolean({ description: "Replace files from an earlier download of this job" })),
  spec_path: Type.Optional(Type.String({ minLength: 1, maxLength: 500 })),
}, { additionalProperties: false });

function sha256(bytes: Buffer | string): string {
  return createHash("sha256").update(bytes).digest("hex");
}

/** Resolve a workspace-relative data file without traversal or symlinks. */
async function workspaceFile(workspaceRoot: string, file: string): Promise<{ path: string; display: string }> {
  const root = await canonicalWorkspace(workspaceRoot);
  const parts = file.replace(/^\.\//, "").split(/[\\/]/);
  if (isAbsolute(file) || parts.some((part) => !part || part === "." || part === "..")) {
    throw new Error("file must be a path inside the current workspace (no absolute paths or traversal).");
  }
  let current = root;
  for (const part of parts) {
    current = join(current, part);
    let info;
    try {
      info = lstatSync(current);
    } catch {
      throw new Error(`File not found in the workspace: ${file}`);
    }
    if (info.isSymbolicLink()) throw new Error("Labeling sources must be real files, not symlinks.");
  }
  if (!lstatSync(current).isFile()) throw new Error(`${file} is not a regular file.`);
  return { path: current, display: `./${relative(root, current).split(/[\\/]/).join("/")}` };
}

function countRows(text: string, format: "jsonl" | "csv"): number {
  const lines = text.split(/\r?\n/).filter((line) => line.trim());
  // Quoted CSV fields can span lines; the service reports the exact count.
  return format === "csv" ? Math.max(lines.length - 1, 0) : lines.length;
}

/** Inspect a raw file and the local spec, and describe what approval will do. */
export async function prepareLabelJob(
  workspaceRoot: string,
  input: { file: string; input_column?: string; name?: string; spec_path?: string },
): Promise<{ args: Record<string, unknown>; preview: Record<string, unknown>; summary: string }> {
  const spec = await inspectLocalSpec(workspaceRoot, input.spec_path ?? "tunedtensor.json");
  // Labeling is often how a spec gets its training data, so it need not be
  // trainable yet; it only needs readable instructions and examples.
  readLocalAdapterSpec(spec.path);
  const document = spec.document as Record<string, unknown>;
  const examples = Array.isArray(document.examples) ? document.examples.length : 0;
  if (examples === 0) {
    throw new Error(
      "Add a few labeled examples to the spec first: the teacher follows them as the output format, "
      + "and the cloud spec requires at least one.",
    );
  }
  const source = await workspaceFile(workspaceRoot, input.file);
  const { format, warnings } = validateUnlabeledFile(source.path, input.input_column);
  const bytes = readFileSync(source.path);
  const rows = countRows(bytes.toString("utf8"), format);
  return {
    summary: `Teacher-label ${rows} row(s) from ${source.display} under ${spec.displayPath}.`,
    args: {
      file: source.display,
      file_sha256: sha256(bytes),
      format,
      input_column: input.input_column ?? null,
      name: input.name ?? null,
      spec_path: spec.displayPath,
      spec_sha256: spec.sha256,
      workspace_fingerprint: spec.workspaceFingerprint,
    },
    preview: {
      file: source.display,
      format,
      rows,
      ...(warnings.length ? { warnings: warnings.slice(0, 5) } : {}),
      spec_path: spec.displayPath,
      spec_examples: examples,
      steps: [
        `Push ${spec.displayPath} to your TT account (same as tt cloud push; a new cloud spec ID is written back to the file).`,
        `Upload ${source.display} and start a hosted labeling job with the account's teacher model.`,
      ],
      cost: "Uses TT credits. The service reserves the estimate when the job starts and refuses the job if your balance is too low.",
      privacy: "Inputs are sent to Tuned Tensor and a third-party teacher model. Secrets are blocked and emails and phone numbers are redacted first.",
    },
  };
}

/** Re-check that the prepared file and spec are unchanged, then return execution arguments. */
export async function revalidateLabelJob(workspaceRoot: string, args: Record<string, unknown>): Promise<StartLabelJobArgs> {
  await validateLabelWorkspace(workspaceRoot, args);
  const file = String(args.file ?? "");
  const specPath = String(args.spec_path ?? "");
  const source = await workspaceFile(workspaceRoot, file);
  const fileBytes = readFileSync(source.path);
  if (sha256(fileBytes) !== args.file_sha256) {
    throw new Error(`${file} changed after this labeling job was prepared; prepare it again.`);
  }
  const spec = await inspectLocalSpec(workspaceRoot, specPath);
  if (spec.sha256 !== args.spec_sha256) {
    throw new Error(`${specPath} changed after this labeling job was prepared; prepare it again.`);
  }
  const format = args.format === "csv" ? "csv" : "jsonl";
  const inputColumn = typeof args.input_column === "string" ? args.input_column : null;
  validateUnlabeledFile(source.path, inputColumn ?? undefined);
  return {
    workspaceRoot,
    file: source.path,
    fileBytes,
    format,
    inputColumn,
    name: typeof args.name === "string" ? args.name : null,
    specPath: spec.path,
  };
}

function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

/** Compact progress view of a labeling job for the model and the user. */
export function summarizeLabelJob(value: unknown): Record<string, unknown> {
  const job = record(record(value)?.data ?? value) ?? {};
  const counts = record(job.counts);
  const total = typeof counts?.total === "number" ? counts.total : undefined;
  const pending = typeof counts?.pending === "number" ? counts.pending : undefined;
  const done = total !== undefined && pending !== undefined ? total - pending : undefined;
  const status = typeof job.status === "string" ? job.status : "unknown";
  const next: Record<string, string> = {
    preparing: "The service is parsing the file. Check again shortly.",
    labeling: "The teacher is labeling rows. Check again later; it runs in the cloud without this session.",
    awaiting_review: "Finished. Offer prepare_label_download. Ask whether to use only accepted/edited rows (review first with tt cloud label rows/accept) or include unreviewed teacher drafts.",
    promoted: "Finished and promoted. prepare_label_download can still write local files.",
    failed: "Labeling failed; report the error. Unused credits are released.",
    cancelled: "The job was cancelled.",
  };
  return {
    id: job.id,
    name: job.name,
    status,
    ...(done !== undefined && total ? {
      progress: { labeled_or_failed: done, total, percent: Math.round((done / total) * 100) },
    } : {}),
    ...(counts ? { review: counts } : {}),
    estimated_cost_cents: job.est_cost_cents,
    actual_cost_cents: job.actual_cost_cents ?? null,
    ...(job.error ? { error: job.error } : {}),
    created_at: job.created_at,
    next: next[status] ?? "Report the status to the user.",
  };
}

export async function prepareLabelDownload(
  workspaceRoot: string,
  job: Record<string, unknown>,
  input: { job_id: string; include_unreviewed?: boolean; validation_ratio?: number; overwrite?: boolean; spec_path?: string },
): Promise<{ args: Record<string, unknown>; preview: Record<string, unknown>; summary: string }> {
  const spec = await inspectLocalSpec(workspaceRoot, input.spec_path ?? "tunedtensor.json");
  readLocalAdapterSpec(spec.path);
  const status = String(job.status ?? "unknown");
  if (status !== "awaiting_review" && status !== "promoted") {
    throw new Error(`Labeling job is "${status}"; it can be downloaded once labeling finishes.`);
  }
  const counts = record(job.counts) ?? {};
  const reviewed = Number(counts.accepted ?? 0) + Number(counts.edited ?? 0);
  const unreviewed = Number(counts.labeled ?? 0);
  const includeUnreviewed = input.include_unreviewed === true;
  if (reviewed + (includeUnreviewed ? unreviewed : 0) < 2) {
    throw new Error(
      `Only ${reviewed} reviewed row(s) and ${unreviewed} unreviewed teacher draft(s). `
      + "Ask the user to accept rows (tt cloud label accept) or to include unreviewed drafts.",
    );
  }
  const outDir = labelingDownloadDir(spec.path, { id: input.job_id, name: String(job.name ?? "") });
  assertLabelDownloadPaths(outDir, input.overwrite === true);
  const existing = Object.values(labelingDownloadFiles(outDir)).filter((path) => existsSync(path));
  const root = await canonicalWorkspace(workspaceRoot);
  const display = `./${relative(root, outDir).split(/[\\/]/).join("/")}`;
  return {
    summary: `Download labeling job ${String(job.name ?? input.job_id)} to ${display}.`,
    args: {
      job_id: input.job_id,
      spec_path: spec.displayPath,
      spec_sha256: spec.sha256,
      workspace_fingerprint: spec.workspaceFingerprint,
      out_dir: display,
      include_unreviewed: includeUnreviewed,
      validation_ratio: input.validation_ratio ?? 0.1,
      overwrite: input.overwrite === true,
    },
    preview: {
      job: { id: input.job_id, name: job.name, status },
      rows: { accepted_or_edited: reviewed, unreviewed_drafts: unreviewed, rejected: counts.rejected ?? 0, failed: counts.failed ?? 0 },
      uses: includeUnreviewed ? "accepted, edited and unreviewed teacher drafts" : "accepted and edited rows only",
      output: [`${display}/training.jsonl`, `${display}/validation.jsonl`, `${display}/export.json`],
      ...(existing.length ? { overwrites: true } : {}),
      cleaning: "Drops rows blocked for secrets, duplicate inputs and inputs with conflicting labels; holds out a deterministic validation split.",
      next: "Then set dataset_prebuilt in the spec to these files.",
    },
  };
}

export async function validateLabelWorkspace(workspaceRoot: string, args: Record<string, unknown>): Promise<void> {
  if (await fingerprintWorkspace(await canonicalWorkspace(workspaceRoot)) !== args.workspace_fingerprint) {
    throw new SpecWorkspaceMismatchError("The labeling workspace changed; return to the original workspace before approving.");
  }
}

export async function revalidateLabelDownload(workspaceRoot: string, args: Record<string, unknown>): Promise<DownloadLabelJobArgs> {
  await validateLabelWorkspace(workspaceRoot, args);
  const spec = await inspectLocalSpec(workspaceRoot, String(args.spec_path ?? ""));
  if (spec.sha256 !== args.spec_sha256) {
    throw new Error("The spec changed after this download was prepared; prepare it again.");
  }
  const parts = String(args.out_dir ?? "").replace(/^\.\//, "").split(/[\\/]/);
  if (isAbsolute(String(args.out_dir)) || parts.some(part => !part || part === "." || part === "..")) {
    throw new Error("Download directory must be inside the current workspace.");
  }
  const outDir = join(spec.root, ...parts);
  assertLabelDownloadPaths(outDir, args.overwrite === true);
  return {
    workspaceRoot,
    jobId: String(args.job_id ?? ""),
    specPath: spec.path,
    outDir,
    includeUnreviewed: args.include_unreviewed === true,
    validationRatio: typeof args.validation_ratio === "number" ? args.validation_ratio : 0.1,
    overwrite: args.overwrite === true,
  };
}

export function isLabelingOperation(action: AgentAction): boolean {
  return (LABELING_OPERATIONS as readonly string[]).includes(action.operation ?? "");
}

/** The parts of an approved action the model needs for the next step. */
export function approvedResultForModel(operation: string | undefined, output: unknown): Record<string, unknown> | undefined {
  const value = record(output);
  if (!value) return undefined;
  if (operation === "prefetch_local") {
    return {
      operation, kind: value.kind, repo: value.repo, revision: value.revision,
      snapshot_path: value.snapshot_path, manifest_path: value.manifest_path,
      lock_path: value.lock_path, files: value.files,
      file_count: value.file_count, size_bytes: value.size_bytes, checksums_verified: value.checksums_verified,
      status: value.status, local_base_model_path: value.local_base_model_path,
      reason: value.reason,
    };
  }
  if (operation === "start_label_job") {
    return {
      operation,
      job_id: value.job_id,
      name: value.name,
      status: value.status,
      estimated_cost_cents: value.estimated_cost_cents,
      cloud_spec_id: value.cloud_spec_id,
    };
  }
  if (operation === "download_label_job") {
    const manifest = record(value.manifest) ?? {};
    return {
      operation,
      dataset_prebuilt: value.dataset_prebuilt,
      training_rows: manifest.training_rows,
      validation_rows: manifest.validation_rows,
      dropped_duplicate_inputs: manifest.dropped_duplicate_inputs,
      dropped_conflicting_inputs: manifest.dropped_conflicting_inputs,
      skipped_unreviewed: manifest.skipped_unreviewed,
      teacher_instructions_match_spec: manifest.teacher_instructions_match_spec,
    };
  }
  return undefined;
}

/** A note the next turn carries so the model can continue from approved results. */
export function approvedResultsNote(actions: AgentAction[]): string | undefined {
  const settled = new Set(["completed", "failed", "rejected", "outcome_unknown"]);
  const pending = actions.filter((action) => action.result_reported !== true
    && (action.result !== undefined || settled.has(action.status ?? "")));
  if (pending.length === 0) return undefined;
  for (const action of pending) action.result_reported = true;
  const outcomes = pending.map((action) => {
    const args = record(action.arguments);
    return JSON.stringify({
      action_id: action.id, operation: action.operation, status: action.status,
      // The proposal is already in the transcript. Carry its durable outcome,
      // not another copy of the spec or preview, into the next model turn.
      spec_path: args?.spec_path,
      result: action.result,
    });
  });
  return `[Action outcomes, recorded by tt]\n${outcomes.join("\n")}\nThese are historical action outcomes. Read the current spec before describing its current contents or proposing another edit.`;
}
