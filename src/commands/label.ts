import { Command } from "commander";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { basename, dirname, relative, resolve, sep } from "node:path";
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

function readLocalAdapterSpec(specPath: string): { systemMessage: string } {
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
      const name =
        cmdOpts.name || basename(file).replace(/\.(jsonl|json|csv)$/i, "");

      const fileBytes = readFileSync(file);
      const contentType = format === "csv" ? "text/csv" : "application/jsonl";
      const { data: uploadUrl } = await post<UploadUrl>(
        "/labeling-jobs/upload-url",
        {
          // The service stores JSON Lines under .jsonl; a validated .json file is JSONL.
          filename: basename(file).replace(/\.json$/i, ".jsonl"),
          size: statSync(file).size,
          contentType,
        },
        opts,
      );

      const uploadRes = await fetch(uploadUrl.upload_url, {
        method: uploadUrl.method,
        headers: uploadUrl.headers ?? { "Content-Type": contentType },
        body: new Blob([fileBytes], { type: contentType }),
      });
      if (!uploadRes.ok) {
        throw new Error(`Upload failed: ${uploadRes.status} ${uploadRes.statusText}`);
      }

      const { data: job } = await post<LabelingJob & { teacher_row_count: number }>(
        "/labeling-jobs",
        {
          path: uploadUrl.path,
          name,
          behavior_spec_id: specId,
          source_format: format,
          input_column: cmdOpts.inputColumn ?? null,
        },
        opts,
      );

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
        `Review with \`tt cloud label rows ${shortId(job.id)} --status labeled\`, then \`tt cloud label export ${shortId(job.id)}\` to write local training files.`,
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
        `Write local training files with \`tt cloud label export ${shortId(jobId)}\`.`,
      );
    });

  label
    .command("export")
    .description("Write reviewed rows as local training and validation chat JSONL for tt pipeline run")
    .argument("<id>", "Labeling job ID (full UUID or 4+ char prefix)")
    .option("--spec <path>", "Local adapter spec whose system message the rows train under", "tunedtensor.json")
    .option("-o, --out-dir <dir>", "Output directory (default: datasets/<job-name> next to the spec)")
    .option("--validation-ratio <ratio>", "Share of unique inputs held out for validation (0.05-0.5)", "0.1")
    .option("--include-unreviewed", "Also include teacher drafts nobody accepted or edited")
    .option("--force", "Overwrite existing export files")
    .action(async (id: string, cmdOpts) => {
      const opts = parent.optsWithGlobals() as ClientOpts;
      const ratio = Number(cmdOpts.validationRatio);
      if (!Number.isFinite(ratio) || ratio < 0.05 || ratio > 0.5) {
        throw new Error("--validation-ratio must be a number between 0.05 and 0.5");
      }
      const specPath = cmdOpts.spec as string;
      const { systemMessage } = readLocalAdapterSpec(specPath);
      const jobId = await resolveLabelingJobId(id, opts);
      const { data: job } = await withTransientRetry(() => get<LabelingJob & { system_message?: string }>(
        `/labeling-jobs/${jobId}`,
        undefined,
        opts,
      ));
      if (job.status !== "awaiting_review" && job.status !== "promoted") {
        throw new Error(
          `Labeling job is "${job.status}"; export needs a finished job (awaiting_review or promoted).`,
        );
      }

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
        if (!isJsonMode() && meta?.total) {
          process.stderr.write(`\rFetching rows ${rows.length}/${meta.total}   `);
        }
        if (data.length < PER_PAGE) break;
      }
      if (!isJsonMode()) process.stderr.write("\n");
      if (total !== undefined && rows.length !== total) {
        throw new Error(`Fetched ${rows.length} of ${total} rows; rows changed during export. Retry.`);
      }

      const result = buildLabelExport(rows, {
        includeUnreviewed: Boolean(cmdOpts.includeUnreviewed),
        validationRatio: ratio,
      });
      const safeName = job.name.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^[-.]+|-+$/g, "") || shortId(job.id);
      const outDir = resolve(cmdOpts.outDir ?? resolve(dirname(resolve(specPath)), "datasets", safeName));
      const files = {
        training: resolve(outDir, "training.jsonl"),
        validation: resolve(outDir, "validation.jsonl"),
        manifest: resolve(outDir, "export.json"),
      };
      const existing = Object.values(files).filter((path) => existsSync(path));
      if (existing.length > 0 && !cmdOpts.force) {
        throw new Error(`${existing[0]} already exists. Pass --force to overwrite it.`);
      }
      mkdirSync(outDir, { recursive: true });
      writeFileSync(files.training, chatJsonl(systemMessage, result.training));
      writeFileSync(files.validation, chatJsonl(systemMessage, result.validation));
      // The teacher prompt starts with the hosted spec's compiled instructions
      // and may append few-shot examples after them.
      const collapse = (value: string) => value.replace(/\s+/g, " ").trim();
      const instructionsMatch = typeof job.system_message !== "string"
        || collapse(job.system_message).startsWith(collapse(systemMessage));
      const manifest = {
        labeling_job_id: job.id,
        labeling_job_name: job.name,
        teacher_model: job.teacher_model,
        exported_at: new Date().toISOString(),
        spec: toSpecRelative(specPath, specPath),
        system_message_sha256: createHash("sha256").update(systemMessage).digest("hex"),
        teacher_instructions_match_spec: instructionsMatch,
        include_unreviewed: Boolean(cmdOpts.includeUnreviewed),
        validation_ratio: ratio,
        training_rows: result.training.length,
        validation_rows: result.validation.length,
        ...result.stats,
      };
      writeFileSync(files.manifest, JSON.stringify(manifest, null, 2) + "\n");
      const datasetPrebuilt = {
        training: toSpecRelative(specPath, files.training),
        validation: toSpecRelative(specPath, files.validation),
      };

      if (isJsonMode()) return printJson({ ...manifest, files, dataset_prebuilt: datasetPrebuilt });
      printSuccess(
        `Exported ${result.training.length} training and ${result.validation.length} validation rows to ${outDir}`,
      );
      const s = result.stats;
      console.log(
        `Skipped: ${s.skipped_unreviewed} unreviewed · ${s.skipped_rejected_or_failed} rejected/failed/pending · `
        + `${s.skipped_blocked} blocked · ${s.skipped_empty_output} empty · `
        + `${s.dropped_duplicate_inputs} duplicate · ${s.dropped_conflicting_inputs} conflicting-label rows`,
      );
      if (!instructionsMatch) {
        console.error(
          "Warning: the teacher labeled under different instructions than this spec's system_prompt, guidelines and constraints. "
          + "Check that the labels still fit before training.",
        );
      }
      console.log(`\nAdd this to ${specPath}, then run \`tt pipeline run --spec ${specPath}\`:`);
      console.log(JSON.stringify({ dataset_prebuilt: datasetPrebuilt }, null, 2));
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
