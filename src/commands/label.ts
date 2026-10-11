import { Command } from "commander";
import { createHash } from "node:crypto";
import { closeSync, constants, existsSync, lstatSync, mkdirSync, openSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { basename, dirname, parse, relative, resolve, sep } from "node:path";
import { ApiError, get, patch, post, type ClientOpts } from "../client.js";
import { buildSystemMessage, datasetInputIdentity } from "../local-runtime/dataset.js";
import { resolveLabelingJobId, resolveSpecId } from "../resolve.js";
import {
  printTable,
  printDetail,
  printSuccess,
  printJson,
  printError,
  isJsonMode,
  formatDate,
  formatStatus,
  shortId,
  truncate,
} from "../output.js";

interface LabelingJob {
  id: string;
  name: string;
  behavior_spec_id: string;
  teacher_model: string;
  source_format: string;
  row_count: number;
  labeled_count: number;
  failed_count: number;
  status: string;
  est_cost_cents: number;
  actual_cost_cents: number | null;
  promoted_dataset_id: string | null;
  error: string | null;
  created_at: string;
  updated_at: string;
}

interface RowCounts {
  total: number;
  pending: number;
  labeled: number;
  accepted: number;
  edited: number;
  rejected: number;
  failed: number;
}

interface LabelingRow {
  id: string;
  row_index: number;
  input: string;
  teacher_output: string | null;
  final_output: string | null;
  status: string;
  error: string | null;
  sanitization_status?: string | null;
}

/** Server-side limits, mirrored so a bad file fails before it is uploaded. */
export const MAX_LABELING_SOURCE_BYTES = 50 * 1024 * 1024;
export const MAX_LABELING_ROWS = 50_000;
export const MAX_LABELING_INPUT_CHARS = 32_000;

interface UploadUrl {
  path: string;
  upload_url: string;
  method: "PUT";
  headers?: Record<string, string>;
  source_format: "jsonl" | "csv";
}

function formatCents(cents: number): string {
  return `$${(cents / 100).toFixed(2)}`;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Network failures, 429 and 5xx are worth retrying; other API errors are final. */
function isTransientError(error: unknown): boolean {
  if (error instanceof ApiError) return error.status === 429 || error.status >= 500;
  return error instanceof TypeError;
}

/**
 * Retry read-only polling/listing calls through brief outages. Labeling runs
 * in the cloud, so a dropped connection should not end a long watch or export.
 */
export async function withTransientRetry<T>(
  call: () => Promise<T>,
  attempts = 5,
  baseDelayMs = 2000,
): Promise<T> {
  for (let attempt = 1; ; attempt += 1) {
    try {
      return await call();
    } catch (error) {
      if (attempt >= attempts || !isTransientError(error)) throw error;
      await sleep(baseDelayMs * 2 ** (attempt - 1));
    }
  }
}

/** Parse a CSV header line, honouring quoted fields. */
export function parseCsvHeader(line: string): string[] {
  const fields: string[] = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === '"') {
        if (line[i + 1] === '"') {
          current += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        current += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ",") {
      fields.push(current);
      current = "";
    } else {
      current += ch;
    }
  }
  fields.push(current);
  return fields.map((f) => f.trim());
}

export function validateUnlabeledFile(
  file: string,
  inputColumn?: string,
): { format: "jsonl" | "csv"; warnings: string[] } {
  const lower = file.toLowerCase();
  const warnings: string[] = [];
  const size = statSync(file).size;
  if (size > MAX_LABELING_SOURCE_BYTES) {
    throw new Error("Labeling source exceeds the 50 MB upload limit. Split it into several jobs.");
  }
  if (lower.endsWith(".csv")) {
    if (!inputColumn) {
      throw new Error(
        "CSV uploads need --input-column <name> to say which column holds the input text.",
      );
    }
    const firstLine = readFileSync(file, "utf8").split(/\r?\n/, 1)[0] ?? "";
    const columns = parseCsvHeader(firstLine);
    if (!columns.includes(inputColumn)) {
      throw new Error(
        `Column "${inputColumn}" not found in CSV header. Available columns: ${columns.join(", ")}`,
      );
    }
    return { format: "csv", warnings };
  }

  if (!lower.endsWith(".jsonl") && !lower.endsWith(".json")) {
    throw new Error("Labeling source file must be .jsonl or .csv");
  }

  const lines = readFileSync(file, "utf8").split(/\r?\n/);
  const errors: string[] = [];
  let rowCount = 0;
  for (const [index, rawLine] of lines.entries()) {
    const line = rawLine.trim();
    if (!line) continue;
    rowCount += 1;
    let row: unknown;
    try {
      row = JSON.parse(line);
    } catch {
      errors.push(`Row ${index + 1}: invalid JSON`);
      continue;
    }
    if (
      !row ||
      typeof row !== "object" ||
      Array.isArray(row) ||
      typeof (row as Record<string, unknown>).input !== "string"
    ) {
      errors.push(`Row ${index + 1}: expected an object with a string "input" field`);
      continue;
    }
    const { input, output } = row as Record<string, unknown>;
    if (!(input as string).trim()) {
      errors.push(`Row ${index + 1}: "input" is empty`);
    } else if ((input as string).length > MAX_LABELING_INPUT_CHARS) {
      warnings.push(`Row ${index + 1}: input exceeds ${MAX_LABELING_INPUT_CHARS.toLocaleString()} characters and will be marked failed`);
    }
    if (output !== undefined && typeof output !== "string") {
      errors.push(`Row ${index + 1}: "output" must be a string when present`);
    } else if (typeof output === "string" && !output.trim()) {
      errors.push(`Row ${index + 1}: "output" is empty - omit it to have the teacher label this row`);
    }
  }
  if (rowCount === 0) {
    errors.push("File contains no JSONL rows");
  } else if (rowCount > MAX_LABELING_ROWS) {
    errors.push(`File has ${rowCount.toLocaleString()} rows; the limit is ${MAX_LABELING_ROWS.toLocaleString()} per labeling job`);
  }
  if (errors.length > 0) {
    const preview = errors.slice(0, 5).join("\n");
    const suffix =
      errors.length > 5 ? `\n...and ${errors.length - 5} more error(s)` : "";
    throw new Error(
      `Invalid labeling source. Each JSONL row must be {"input": "..."}.\n${preview}${suffix}`,
    );
  }
  return { format: "jsonl", warnings };
}

function printJobDetail(job: LabelingJob, counts?: RowCounts) {
  printDetail([
    ["ID", job.id],
    ["Name", job.name],
    ["Status", formatStatus(job.status)],
    ["Spec", shortId(job.behavior_spec_id)],
    ["Teacher", job.teacher_model],
    ["Rows", String(job.row_count)],
    ["Labeled", String(job.labeled_count)],
    ["Failed", String(job.failed_count)],
    [
      "Cost",
      job.actual_cost_cents !== null
        ? formatCents(job.actual_cost_cents)
        : `~${formatCents(job.est_cost_cents)} (estimated)`,
    ],
    ["Dataset", job.promoted_dataset_id ? shortId(job.promoted_dataset_id) : undefined],
    ["Error", job.error ?? undefined],
    ["Created", formatDate(job.created_at)],
  ]);
  if (counts) {
    console.log(
      `\nReview: ${counts.labeled} awaiting · ${counts.accepted} accepted · ${counts.edited} edited · ${counts.rejected} rejected · ${counts.failed} failed`,
    );
  }
}

/**
 * Poll the job until the server-side Step Functions workflow finishes.
 * Labeling runs entirely in the cloud — closing the CLI does not stop it;
 * watch just re-attaches to progress.
 */
async function watchJob(jobId: string, opts: ClientOpts): Promise<LabelingJob & { counts: RowCounts }> {
  for (;;) {
    const { data } = await withTransientRetry(() => get<LabelingJob & { counts: RowCounts }>(
      `/labeling-jobs/${jobId}`,
      undefined,
      opts,
    ));

    if (!isJsonMode()) {
      if (data.status === "preparing") {
        process.stdout.write("\rPreparing - parsing the source file...   ");
      } else {
        const done = data.counts.total - data.counts.pending;
        process.stdout.write(`\rLabeling ${done}/${data.counts.total} rows   `);
      }
    }

    if (data.status !== "preparing" && data.status !== "labeling") {
      if (!isJsonMode()) process.stdout.write("\n");
      return data;
    }
    await sleep(5000);
  }
}

/**
 * Resolve 0-based row indexes to row ids. Rows are dense and ordered by
 * row_index, so index i lives on page floor(i/100)+1 of the unfiltered list.
 */
async function resolveRowIds(
  jobId: string,
  indexes: number[],
  opts: ClientOpts,
): Promise<Map<number, string>> {
  const PER_PAGE = 100;
  const byPage = new Map<number, number[]>();
  for (const index of indexes) {
    const page = Math.floor(index / PER_PAGE) + 1;
    byPage.set(page, [...(byPage.get(page) ?? []), index]);
  }

  const found = new Map<number, string>();
  for (const [page, wanted] of byPage) {
    const { data } = await get<LabelingRow[]>(
      `/labeling-jobs/${jobId}/rows`,
      { page, per_page: PER_PAGE },
      opts,
    );
    for (const row of data) {
      if (wanted.includes(row.row_index)) {
        found.set(row.row_index, row.id);
      }
    }
  }

  const missing = indexes.filter((i) => !found.has(i));
  if (missing.length > 0) {
    throw new Error(`Row index(es) not found: ${missing.join(", ")}`);
  }
  return found;
}

function parseIndexList(value: string): number[] {
  const indexes = value.split(",").map((part) => Number.parseInt(part.trim(), 10));
  if (indexes.some((n) => Number.isNaN(n) || n < 0)) {
    throw new Error(`Invalid row index list: "${value}" (expected e.g. "0,3,12")`);
  }
  return indexes;
}

export interface LabelExportOptions {
  includeUnreviewed: boolean;
  validationRatio: number;
}

export interface LabelExportResult {
  training: Array<{ input: string; output: string }>;
  validation: Array<{ input: string; output: string }>;
  stats: {
    rows: number;
    selected: number;
    skipped_unreviewed: number;
    skipped_rejected_or_failed: number;
    skipped_blocked: number;
    skipped_empty_output: number;
    dropped_duplicate_inputs: number;
    dropped_conflicting_inputs: number;
  };
}

/** Stable position in [0, 1) so re-exporting the same rows keeps the same split. */
function splitPosition(identity: string): number {
  return createHash("sha256").update(identity).digest().readUInt32BE(0) / 2 ** 32;
}

/**
 * Turn reviewed labeling rows into held-out training data. Accepted and edited
 * rows always count (edits win); unreviewed teacher drafts only on request.
 * Rows the service blocked for secrets never leave. Inputs are compared with
 * the same normalization the local trainer uses: exact duplicates keep one row,
 * and inputs the teacher or reviewer labeled differently are dropped entirely
 * because neither label can be trusted. The split is deterministic by input.
 */
export function buildLabelExport(
  rows: readonly LabelingRow[],
  options: LabelExportOptions,
): LabelExportResult {
  const stats: LabelExportResult["stats"] = {
    rows: rows.length,
    selected: 0,
    skipped_unreviewed: 0,
    skipped_rejected_or_failed: 0,
    skipped_blocked: 0,
    skipped_empty_output: 0,
    dropped_duplicate_inputs: 0,
    dropped_conflicting_inputs: 0,
  };
  const groups = new Map<string, { input: string; outputs: Set<string>; count: number }>();
  for (const row of [...rows].sort((a, b) => a.row_index - b.row_index)) {
    const reviewed = row.status === "accepted" || row.status === "edited";
    if (!reviewed && row.status !== "labeled") {
      stats.skipped_rejected_or_failed += 1;
      continue;
    }
    if (!reviewed && !options.includeUnreviewed) {
      stats.skipped_unreviewed += 1;
      continue;
    }
    if (row.sanitization_status === "blocked") {
      stats.skipped_blocked += 1;
      continue;
    }
    const output = row.final_output ?? row.teacher_output;
    if (!output?.trim() || !row.input.trim()) {
      stats.skipped_empty_output += 1;
      continue;
    }
    const identity = datasetInputIdentity(row.input);
    const group = groups.get(identity);
    if (group) {
      group.outputs.add(output.trim());
      group.count += 1;
    } else {
      groups.set(identity, { input: row.input, outputs: new Set([output.trim()]), count: 1 });
    }
  }

  const examples: Array<{ identity: string; input: string; output: string }> = [];
  for (const [identity, group] of groups) {
    if (group.outputs.size > 1) {
      stats.dropped_conflicting_inputs += group.count;
      continue;
    }
    stats.dropped_duplicate_inputs += group.count - 1;
    examples.push({ identity, input: group.input, output: [...group.outputs][0]! });
  }
  stats.selected = examples.length;
  if (examples.length < 2) {
    throw new Error(
      `Only ${examples.length} usable row(s) after review filtering and de-duplication; `
      + "at least 2 are needed for a training and a held-out validation split. "
      + (options.includeUnreviewed ? "" : "Accept more rows, or pass --include-unreviewed to use teacher drafts as-is."),
    );
  }

  const ranked = examples
    .map((example) => ({ ...example, position: splitPosition(example.identity) }))
    .sort((a, b) => a.position - b.position);
  let validationCount = ranked.filter((example) => example.position < options.validationRatio).length;
  validationCount = Math.min(Math.max(validationCount, 1), ranked.length - 1);
  const strip = ({ input, output }: { input: string; output: string }) => ({ input, output });
  // Keep source order within each split so the files are easy to audit.
  const order = new Map(examples.map((example, index) => [example.identity, index]));
  const bySource = (a: { identity: string }, b: { identity: string }) => order.get(a.identity)! - order.get(b.identity)!;
  return {
    validation: ranked.slice(0, validationCount).sort(bySource).map(strip),
    training: ranked.slice(validationCount).sort(bySource).map(strip),
    stats,
  };
}

export function readLocalAdapterSpec(specPath: string): { systemMessage: string } {
  if (!existsSync(specPath)) {
    throw new Error(`Spec file not found: ${specPath}. Run \`tt init\` or pass --spec <path>.`);
  }
  let spec: Record<string, unknown>;
  try {
    spec = JSON.parse(readFileSync(specPath, "utf8")) as Record<string, unknown>;
  } catch (error) {
    throw new Error(`${specPath} is not valid JSON: ${(error as Error).message}`);
  }
  const engine = spec.engine ?? "adapter";
  if (engine !== "adapter") {
    throw new Error(
      `${specPath} uses engine "${String(engine)}". Labeled exports feed adapter dataset_prebuilt; `
      + "decision and foundation specs use their own data workflows.",
    );
  }
  const strings = (value: unknown) => Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
  return {
    systemMessage: buildSystemMessage({
      system_prompt: typeof spec.system_prompt === "string" ? spec.system_prompt : "",
      guidelines: strings(spec.guidelines),
      constraints: strings(spec.constraints),
    }),
  };
}

function chatJsonl(systemMessage: string, rows: ReadonlyArray<{ input: string; output: string }>): string {
  return rows.map(({ input, output }) => JSON.stringify({
    messages: [
      { role: "system", content: systemMessage },
      { role: "user", content: input },
      { role: "assistant", content: output },
    ],
  })).join("\n") + "\n";
}

function toSpecRelative(specPath: string, target: string): string {
  const path = relative(dirname(resolve(specPath)), resolve(target)).split(sep).join("/");
  return path.startsWith(".") ? path : `./${path}`;
}

export const INSTRUCTIONS_MISMATCH_WARNING = "The teacher labeled under different instructions than this spec's "
  + "system_prompt, guidelines and constraints. Check that the labels still fit before training.";

/** A labeling job was certainly not created, so preparing it again is safe. */
export class LabelingNotStartedError extends Error {
  constructor(message: string, cause?: unknown) {
    super(message, { cause });
    this.name = "LabelingNotStartedError";
  }
}

/** Run a step that precedes job creation, or job creation the service refused. */
export async function beforeLabelingStarts<T>(step: () => Promise<T>): Promise<T> {
  try {
    return await step();
  } catch (error) {
    if (error instanceof LabelingNotStartedError) throw error;
    throw new LabelingNotStartedError(error instanceof Error ? error.message : String(error), error);
  }
}

/** Upload a validated source file and start a hosted labeling job under a cloud spec. */
export async function startLabelingJob(
  args: { file: string; fileBytes?: Buffer<ArrayBuffer>; format: "jsonl" | "csv"; specId: string; inputColumn?: string | null; name?: string },
  opts: ClientOpts,
): Promise<LabelingJob> {
  const fileBytes = args.fileBytes ?? readFileSync(args.file);
  const contentType = args.format === "csv" ? "text/csv" : "application/jsonl";
  const uploadUrl = await beforeLabelingStarts(async () => {
    const { data } = await post<UploadUrl>(
      "/labeling-jobs/upload-url",
      {
        // The service stores JSON Lines under .jsonl; a validated .json file is JSONL.
        filename: basename(args.file).replace(/\.json$/i, ".jsonl"),
        size: fileBytes.byteLength,
        contentType,
      },
      opts,
    );
    const uploadRes = await fetch(data.upload_url, {
      method: data.method,
      headers: data.headers ?? { "Content-Type": contentType },
      body: new Blob([fileBytes], { type: contentType }),
    });
    if (!uploadRes.ok) {
      throw new Error(`Upload failed: ${uploadRes.status} ${uploadRes.statusText}`);
    }
    return data;
  });

  try {
    const { data: job } = await post<LabelingJob>(
      "/labeling-jobs",
      {
        path: uploadUrl.path,
        name: args.name?.trim() || basename(args.file).replace(/\.(jsonl|json|csv)$/i, ""),
        behavior_spec_id: args.specId,
        source_format: args.format,
        input_column: args.inputColumn ?? null,
      },
      opts,
    );
    return job;
  } catch (error) {
    // The service validates, reserves credits and creates the job in one
    // request; a 4xx (bad file, insufficient credits, reused upload) creates nothing.
    if (error instanceof ApiError && error.status >= 400 && error.status < 500) {
      throw new LabelingNotStartedError(error.message, error);
    }
    throw error;
  }
}

export function labelingDownloadDir(specPath: string, job: { id: string; name: string }, outDir?: string): string {
  const safeName = job.name.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^[-.]+|-+$/g, "") || shortId(job.id);
  return resolve(outDir ?? resolve(dirname(resolve(specPath)), "datasets", safeName));
}

export function labelingDownloadFiles(outDir: string) {
  return {
    training: resolve(outDir, "training.jsonl"),
    validation: resolve(outDir, "validation.jsonl"),
    manifest: resolve(outDir, "export.json"),
  };
}

/** Do not let a download follow existing directory or output-file symlinks. */
export function assertLabelDownloadPaths(outDir: string, overwrite: boolean): void {
  const absolute = resolve(outDir);
  let current = parse(absolute).root;
  for (const part of relative(current, absolute).split(sep).filter(Boolean)) {
    current = resolve(current, part);
    const info = lstatSync(current, { throwIfNoEntry: false });
    if (info && (!info.isDirectory() || info.isSymbolicLink())) {
      throw new Error("Download directories must be real directories, not symlinks.");
    }
  }
  for (const file of Object.values(labelingDownloadFiles(absolute))) {
    const info = lstatSync(file, { throwIfNoEntry: false });
    if (info && (!info.isFile() || info.isSymbolicLink() || info.nlink > 1)) {
      throw new Error("Download outputs must be regular files, not symlinks or hard links.");
    }
    if (info && !overwrite) {
      throw new Error(`${file} already exists. Ask the user whether to overwrite it (--force) or choose another directory.`);
    }
  }
}

export function teacherInstructionsMatch(systemMessage: string, teacherMessage: unknown): boolean {
  if (typeof teacherMessage !== "string" || !systemMessage.trim()) return false;
  const examplesHeader = "\n\nLabeled examples. Follow their output format exactly and reply with the output only:\n\n";
  return teacherMessage === systemMessage || teacherMessage.startsWith(systemMessage + examplesHeader);
}

/** Fetch every row of a job, retrying transient failures, and check none were missed. */
export async function fetchLabelingRows(
  jobId: string,
  opts: ClientOpts,
  onProgress?: (fetched: number, total: number) => void,
): Promise<LabelingRow[]> {
  const PER_PAGE = 100;
  const rows: LabelingRow[] = [];
  let total: number | undefined;
  for (let page = 1; ; page += 1) {
    const { data, meta } = await withTransientRetry(() => get<LabelingRow[]>(
      `/labeling-jobs/${jobId}/rows`,
      { page, per_page: PER_PAGE },
      opts,
    ));
    rows.push(...data);
    total = meta?.total ?? total;
    if (total !== undefined) onProgress?.(rows.length, total);
    if (data.length < PER_PAGE) break;
  }
  if (total !== undefined && rows.length !== total) {
    throw new Error(`Fetched ${rows.length} of ${total} rows; rows changed during download. Retry.`);
  }
  return rows;
}

export interface LabelDownloadArgs {
  jobId: string;
  specPath: string;
  outDir?: string;
  includeUnreviewed: boolean;
  validationRatio: number;
  force: boolean;
  onProgress?: (fetched: number, total: number) => void;
}

/**
 * Download a finished labeling job as local chat JSONL under the local adapter
 * spec's system message, plus an export.json manifest.
 */
export async function downloadLabelingJob(args: LabelDownloadArgs, opts: ClientOpts) {
  const ratio = args.validationRatio;
  if (!Number.isFinite(ratio) || ratio < 0.05 || ratio > 0.5) {
    throw new Error("The validation ratio must be a number between 0.05 and 0.5.");
  }
  const { systemMessage } = readLocalAdapterSpec(args.specPath);
  const { data: job } = await withTransientRetry(() => get<LabelingJob & { system_message?: string }>(
    `/labeling-jobs/${args.jobId}`,
    undefined,
    opts,
  ));
  if (job.status !== "awaiting_review" && job.status !== "promoted") {
    throw new Error(
      `Labeling job is "${job.status}"; download needs a finished job (awaiting_review or promoted).`,
    );
  }
  const outDir = labelingDownloadDir(args.specPath, job, args.outDir);
  const files = labelingDownloadFiles(outDir);
  assertLabelDownloadPaths(outDir, args.force);

  const rows = await fetchLabelingRows(args.jobId, opts, args.onProgress);
  const result = buildLabelExport(rows, {
    includeUnreviewed: args.includeUnreviewed,
    validationRatio: ratio,
  });
  // Fetching can take minutes; recheck paths and overwrite consent at write time.
  assertLabelDownloadPaths(outDir, args.force);
  mkdirSync(outDir, { recursive: true });
  const writeOutput = (file: string, text: string) => {
    const fd = openSync(file, constants.O_WRONLY | constants.O_CREAT | constants.O_NOFOLLOW
      | (args.force ? constants.O_TRUNC : constants.O_EXCL), 0o600);
    try { writeFileSync(fd, text); } finally { closeSync(fd); }
  };
  writeOutput(files.training, chatJsonl(systemMessage, result.training));
  writeOutput(files.validation, chatJsonl(systemMessage, result.validation));
  // The teacher prompt starts with the hosted spec's compiled instructions
  // and may append few-shot examples after them.
  const instructionsMatch = teacherInstructionsMatch(systemMessage, job.system_message);
  const manifest = {
    labeling_job_id: job.id,
    labeling_job_name: job.name,
    teacher_model: job.teacher_model,
    exported_at: new Date().toISOString(),
    spec: toSpecRelative(args.specPath, args.specPath),
    system_message_sha256: createHash("sha256").update(systemMessage).digest("hex"),
    teacher_instructions_match_spec: instructionsMatch,
    include_unreviewed: args.includeUnreviewed,
    validation_ratio: ratio,
    training_rows: result.training.length,
    validation_rows: result.validation.length,
    ...result.stats,
  };
  writeOutput(files.manifest, JSON.stringify(manifest, null, 2) + "\n");
  return {
    out_dir: outDir,
    files,
    manifest,
    dataset_prebuilt: {
      training: toSpecRelative(args.specPath, files.training),
      validation: toSpecRelative(args.specPath, files.validation),
    },
  };
}

export function registerLabelCommands(parent: Command) {
  const label = parent
    .command("label")
    .description("Teacher-label real data into training datasets");

  label
    .command("upload")
    .description("Upload unlabeled inputs (.jsonl or .csv) and start a labeling job")
    .argument("<file>", "Path to JSONL ({\"input\": ...} per line) or CSV file")
    .requiredOption("-s, --spec <id>", "Behaviour spec the teacher labels under")
    .option("-c, --input-column <name>", "CSV column that holds the input text")
    .option("-n, --name <name>", "Job name (defaults to filename)")
    .option("--watch", "Block and show progress until labeling finishes")
    .action(async (file: string, cmdOpts) => {
      const opts = parent.optsWithGlobals() as ClientOpts;

      if (!existsSync(file)) {
        printError(`File not found: ${file}`);
        process.exit(1);
      }

      const { format, warnings } = validateUnlabeledFile(file, cmdOpts.inputColumn);
      if (!isJsonMode()) {
        for (const warning of warnings.slice(0, 5)) console.error(`Warning: ${warning}`);
        if (warnings.length > 5) console.error(`Warning: ...and ${warnings.length - 5} more`);
      }
      const specId = await resolveSpecId(cmdOpts.spec, opts);
      const job = await startLabelingJob({
        file,
        format,
        specId,
        inputColumn: cmdOpts.inputColumn,
        name: cmdOpts.name,
      }, opts);

      if (!cmdOpts.watch) {
        if (isJsonMode()) return printJson(job);
        printSuccess(
          `Labeling started: ${job.name} (${shortId(job.id)}) — est. ${formatCents(job.est_cost_cents)}. Runs in the cloud; no need to stay connected.`,
        );
        console.log(`Follow progress with \`tt cloud label watch ${shortId(job.id)}\`.`);
        return;
      }

      const finished = await watchJob(job.id, opts);
      if (isJsonMode()) return printJson(finished);
      printSuccess(
        `Labeling ${finished.status === "awaiting_review" ? "complete" : finished.status}: ${finished.labeled_count}/${finished.row_count} rows labeled.`,
      );
      console.log(
        `Review with \`tt cloud label rows ${shortId(job.id)} --status labeled\`, then \`tt cloud label download ${shortId(job.id)}\` to write local training files.`,
      );
    });

  label
    .command("watch")
    .description("Watch a labeling job until it is ready for review")
    .argument("<id>", "Labeling job ID (full UUID or 4+ char prefix)")
    .action(async (id: string) => {
      const opts = parent.optsWithGlobals() as ClientOpts;
      const jobId = await resolveLabelingJobId(id, opts);
      const job = await watchJob(jobId, opts);

      if (isJsonMode()) return printJson(job);
      if (job.status === "awaiting_review") {
        printSuccess(
          `Labeling complete: ${job.labeled_count}/${job.row_count} rows labeled, cost ${job.actual_cost_cents !== null ? formatCents(job.actual_cost_cents) : "n/a"}.`,
        );
      } else {
        printError(`Labeling ended with status: ${job.status}${job.error ? ` — ${job.error}` : ""}`);
        process.exitCode = 1;
      }
    });

  label
    .command("list")
    .description("List labeling jobs")
    .option("-p, --page <n>", "Page number", "1")
    .option("--per-page <n>", "Results per page", "20")
    .action(async (cmdOpts) => {
      const opts = parent.optsWithGlobals() as ClientOpts;
      const { data, meta } = await get<LabelingJob[]>(
        "/labeling-jobs",
        { page: cmdOpts.page, per_page: cmdOpts.perPage },
        opts,
      );

      if (isJsonMode()) return printJson({ data, meta });

      printTable(
        ["ID", "Name", "Status", "Progress", "Cost", "Created"],
        data.map((j) => [
          shortId(j.id),
          truncate(j.name, 30),
          formatStatus(j.status),
          `${j.labeled_count}/${j.row_count}`,
          j.actual_cost_cents !== null
            ? formatCents(j.actual_cost_cents)
            : `~${formatCents(j.est_cost_cents)}`,
          formatDate(j.created_at),
        ]),
        meta,
      );
    });

  label
    .command("status")
    .description("Show labeling job details and review progress")
    .argument("<id>", "Labeling job ID (full UUID or 4+ char prefix)")
    .action(async (id: string) => {
      const opts = parent.optsWithGlobals() as ClientOpts;
      const jobId = await resolveLabelingJobId(id, opts);
      const { data } = await get<LabelingJob & { counts: RowCounts }>(
        `/labeling-jobs/${jobId}`,
        undefined,
        opts,
      );

      if (isJsonMode()) return printJson(data);
      printJobDetail(data, data.counts);
    });

  label
    .command("rows")
    .description("List rows in a labeling job")
    .argument("<id>", "Labeling job ID (full UUID or 4+ char prefix)")
    .option(
      "--status <status>",
      "Filter by row status (pending|labeled|accepted|edited|rejected|failed)",
    )
    .option("-p, --page <n>", "Page number", "1")
    .option("--per-page <n>", "Results per page", "50")
    .action(async (id: string, cmdOpts) => {
      const opts = parent.optsWithGlobals() as ClientOpts;
      const jobId = await resolveLabelingJobId(id, opts);
      const query: Record<string, string> = {
        page: cmdOpts.page,
        per_page: cmdOpts.perPage,
      };
      if (cmdOpts.status) query.status = cmdOpts.status;
      const { data, meta } = await get<LabelingRow[]>(
        `/labeling-jobs/${jobId}/rows`,
        query,
        opts,
      );

      if (isJsonMode()) return printJson({ data, meta });

      printTable(
        ["Row", "Status", "Input", "Output"],
        data.map((row) => [
          String(row.row_index),
          formatStatus(row.status),
          truncate(row.input.replace(/\s+/g, " "), 40),
          truncate(
            (row.final_output ?? row.teacher_output ?? "—").replace(/\s+/g, " "),
            40,
          ),
        ]),
        meta,
      );
    });

  label
    .command("accept")
    .description("Accept teacher-labeled rows")
    .argument("<id>", "Labeling job ID (full UUID or 4+ char prefix)")
    .option("--all", "Accept all unreviewed labeled rows")
    .option("--rows <indexes>", "Comma-separated row indexes (e.g. 0,3,12)")
    .action(async (id: string, cmdOpts) => {
      const opts = parent.optsWithGlobals() as ClientOpts;
      if (!cmdOpts.all && !cmdOpts.rows) {
        printError("Provide --all or --rows <indexes>.");
        process.exit(1);
      }
      const jobId = await resolveLabelingJobId(id, opts);

      if (cmdOpts.all) {
        const { data } = await post<{ accepted: number }>(
          `/labeling-jobs/${jobId}/rows/bulk`,
          { action: "accept", all_labeled: true },
          opts,
        );
        if (isJsonMode()) return printJson(data);
        return printSuccess(`Accepted ${data.accepted} rows.`);
      }

      const indexes = parseIndexList(cmdOpts.rows);
      const rowIds = await resolveRowIds(jobId, indexes, opts);
      for (const index of indexes) {
        await patch(
          `/labeling-jobs/${jobId}/rows/${rowIds.get(index)}`,
          { action: "accept" },
          opts,
        );
      }
      if (isJsonMode()) return printJson({ accepted: indexes.length });
      printSuccess(`Accepted ${indexes.length} row(s).`);
    });

  label
    .command("reject")
    .description("Reject rows so they are excluded from promotion")
    .argument("<id>", "Labeling job ID (full UUID or 4+ char prefix)")
    .requiredOption("--rows <indexes>", "Comma-separated row indexes (e.g. 0,3,12)")
    .action(async (id: string, cmdOpts) => {
      const opts = parent.optsWithGlobals() as ClientOpts;
      const jobId = await resolveLabelingJobId(id, opts);
      const indexes = parseIndexList(cmdOpts.rows);
      const rowIds = await resolveRowIds(jobId, indexes, opts);
      for (const index of indexes) {
        await patch(
          `/labeling-jobs/${jobId}/rows/${rowIds.get(index)}`,
          { action: "reject" },
          opts,
        );
      }
      if (isJsonMode()) return printJson({ rejected: indexes.length });
      printSuccess(`Rejected ${indexes.length} row(s).`);
    });

  label
    .command("edit")
    .description("Replace a row's output with your own")
    .argument("<id>", "Labeling job ID (full UUID or 4+ char prefix)")
    .requiredOption("--row <index>", "Row index to edit")
    .option("-o, --output <text>", "Replacement output text")
    .option("-f, --file <path>", "Read replacement output from a file")
    .action(async (id: string, cmdOpts) => {
      const opts = parent.optsWithGlobals() as ClientOpts;
      if (!cmdOpts.output && !cmdOpts.file) {
        printError("Provide --output <text> or --file <path>.");
        process.exit(1);
      }
      const output: string = cmdOpts.output ?? readFileSync(cmdOpts.file, "utf8");
      const jobId = await resolveLabelingJobId(id, opts);
      const indexes = parseIndexList(cmdOpts.row);
      const rowIds = await resolveRowIds(jobId, indexes, opts);
      await patch(
        `/labeling-jobs/${jobId}/rows/${rowIds.get(indexes[0])}`,
        { action: "edit", output },
        opts,
      );
      if (isJsonMode()) return printJson({ edited: indexes[0] });
      printSuccess(`Row ${indexes[0]} edited.`);
    });

  label
    .command("promote")
    .description("Promote reviewed rows into a validated dataset")
    .argument("<id>", "Labeling job ID (full UUID or 4+ char prefix)")
    .requiredOption("-n, --name <name>", "Name for the new dataset")
    .option("-d, --description <desc>", "Dataset description")
    .option("--include-unreviewed", "Also include unreviewed labeled rows")
    .action(async (id: string, cmdOpts) => {
      const opts = parent.optsWithGlobals() as ClientOpts;
      const jobId = await resolveLabelingJobId(id, opts);
      const { data } = await post<{ id: string; name: string; row_count: number }>(
        `/labeling-jobs/${jobId}/promote`,
        {
          name: cmdOpts.name,
          description: cmdOpts.description ?? null,
          include_unreviewed_labeled: Boolean(cmdOpts.includeUnreviewed),
        },
        opts,
      );

      if (isJsonMode()) return printJson(data);
      printSuccess(
        `Dataset created: ${data.name} (${shortId(data.id)}) — ${data.row_count} rows.`,
      );
      console.log(
        `Write local training files with \`tt cloud label download ${shortId(jobId)}\`.`,
      );
    });

  label
    .command("download")
    .alias("export")
    .description("Download reviewed rows as local training and validation files for tt pipeline run")
    .argument("<id>", "Labeling job ID (full UUID or 4+ char prefix)")
    .option("--spec <path>", "Local adapter spec whose system message the rows train under", "tunedtensor.json")
    .option("-o, --out-dir <dir>", "Output directory (default: datasets/<job-name> next to the spec)")
    .option("--validation-ratio <ratio>", "Share of unique inputs held out for validation (0.05-0.5)", "0.1")
    .option("--include-unreviewed", "Also include teacher drafts nobody accepted or edited")
    .option("--force", "Overwrite existing files")
    .action(async (id: string, cmdOpts) => {
      const opts = parent.optsWithGlobals() as ClientOpts;
      const jobId = await resolveLabelingJobId(id, opts);
      const result = await downloadLabelingJob({
        jobId,
        specPath: cmdOpts.spec as string,
        outDir: cmdOpts.outDir,
        includeUnreviewed: Boolean(cmdOpts.includeUnreviewed),
        validationRatio: Number(cmdOpts.validationRatio),
        force: Boolean(cmdOpts.force),
        onProgress: isJsonMode()
          ? undefined
          : (fetched, total) => process.stderr.write(`\rFetching rows ${fetched}/${total}   `),
      }, opts);
      if (!isJsonMode()) process.stderr.write("\n");

      if (isJsonMode()) return printJson(result);
      const { manifest } = result;
      printSuccess(
        `Downloaded ${manifest.training_rows} training and ${manifest.validation_rows} validation rows to ${result.out_dir}`,
      );
      console.log(
        `Skipped: ${manifest.skipped_unreviewed} unreviewed · ${manifest.skipped_rejected_or_failed} rejected/failed/pending · `
        + `${manifest.skipped_blocked} blocked · ${manifest.skipped_empty_output} empty · `
        + `${manifest.dropped_duplicate_inputs} duplicate · ${manifest.dropped_conflicting_inputs} conflicting-label rows`,
      );
      if (!manifest.teacher_instructions_match_spec) {
        console.error(`Warning: ${INSTRUCTIONS_MISMATCH_WARNING}`);
      }
      console.log(`\nAdd this to ${cmdOpts.spec}, then run \`tt pipeline run --spec ${cmdOpts.spec}\`:`);
      console.log(JSON.stringify({ dataset_prebuilt: result.dataset_prebuilt }, null, 2));
    });

  label
    .command("cancel")
    .description("Cancel a labeling job and release unused credits")
    .argument("<id>", "Labeling job ID (full UUID or 4+ char prefix)")
    .action(async (id: string) => {
      const opts = parent.optsWithGlobals() as ClientOpts;
      const jobId = await resolveLabelingJobId(id, opts);
      const { data } = await post<LabelingJob>(
        `/labeling-jobs/${jobId}/cancel`,
        {},
        opts,
      );
      if (isJsonMode()) return printJson(data);
      printSuccess(`Labeling job cancelled: ${shortId(jobId)}`);
    });
}
