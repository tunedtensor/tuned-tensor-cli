import { Command } from "commander";
import { existsSync, readFileSync, statSync } from "node:fs";
import { get, patch, post, type ClientOpts } from "../client.js";
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
}

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
): { format: "jsonl" | "csv" } {
  const lower = file.toLowerCase();
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
    return { format: "csv" };
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
    }
  }
  if (rowCount === 0) {
    errors.push("File contains no JSONL rows");
  }
  if (errors.length > 0) {
    const preview = errors.slice(0, 5).join("\n");
    const suffix =
      errors.length > 5 ? `\n...and ${errors.length - 5} more error(s)` : "";
    throw new Error(
      `Invalid labeling source. Each JSONL row must be {"input": "..."}.\n${preview}${suffix}`,
    );
  }
  return { format: "jsonl" };
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
    const { data } = await get<LabelingJob & { counts: RowCounts }>(
      `/labeling-jobs/${jobId}`,
      undefined,
      opts,
    );

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
      const opts = parent.opts() as ClientOpts;

      if (!existsSync(file)) {
        printError(`File not found: ${file}`);
        process.exit(1);
      }

      const { format } = validateUnlabeledFile(file, cmdOpts.inputColumn);
      const specId = await resolveSpecId(cmdOpts.spec, opts);
      const name =
        cmdOpts.name || file.split("/").pop()!.replace(/\.(jsonl|json|csv)$/, "");

      const fileBytes = readFileSync(file);
      const contentType = format === "csv" ? "text/csv" : "application/jsonl";
      const { data: uploadUrl } = await post<UploadUrl>(
        "/labeling-jobs/upload-url",
        {
          filename: file.split("/").pop()!,
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
        console.log(`Follow progress with \`tt label watch ${shortId(job.id)}\`.`);
        return;
      }

      const finished = await watchJob(job.id, opts);
      if (isJsonMode()) return printJson(finished);
      printSuccess(
        `Labeling ${finished.status === "awaiting_review" ? "complete" : finished.status}: ${finished.labeled_count}/${finished.row_count} rows labeled.`,
      );
      console.log(
        `Review with \`tt label rows ${shortId(job.id)} --status labeled\`, then \`tt label promote ${shortId(job.id)} --name <dataset-name>\`.`,
      );
    });

  label
    .command("watch")
    .description("Watch a labeling job until it is ready for review")
    .argument("<id>", "Labeling job ID (full UUID or 4+ char prefix)")
    .action(async (id: string) => {
      const opts = parent.opts() as ClientOpts;
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
      const opts = parent.opts() as ClientOpts;
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
      const opts = parent.opts() as ClientOpts;
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
      const opts = parent.opts() as ClientOpts;
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
      const opts = parent.opts() as ClientOpts;
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
      const opts = parent.opts() as ClientOpts;
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
      const opts = parent.opts() as ClientOpts;
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
      const opts = parent.opts() as ClientOpts;
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
        `Start a run with \`tt runs start <spec-id> --dataset ${shortId(data.id)}\`.`,
      );
    });

  label
    .command("cancel")
    .description("Cancel a labeling job and release unused credits")
    .argument("<id>", "Labeling job ID (full UUID or 4+ char prefix)")
    .action(async (id: string) => {
      const opts = parent.opts() as ClientOpts;
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
