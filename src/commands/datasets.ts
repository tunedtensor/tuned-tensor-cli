import { Command } from "commander";
import { existsSync, readFileSync, statSync } from "node:fs";
import { get, del, post, type ClientOpts } from "../client.js";
import { resolveDatasetId } from "../resolve.js";
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

interface Dataset {
  id: string;
  name: string;
  description: string | null;
  storage_path: string;
  file_size_bytes: number;
  row_count: number;
  format: string;
  status: string;
  validation_errors: string[] | null;
  created_at: string;
  updated_at: string;
}

interface DatasetUploadUrl {
  path: string;
  upload_url: string;
  method: "PUT";
  headers?: Record<string, string>;
}

type DatasetFormat = "jsonl" | "document_ocr_jsonl";
const DOCUMENT_OCR_MIME_TYPES = new Set(["image/png", "image/jpeg", "image/webp"]);

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// Invisible characters that JSON parsers accept but downstream JSONL consumers
// often mishandle. Concretely: Python's str.splitlines() treats every char in
// this set as a line break, so a row containing one of them gets cut in half
// before json.loads sees it. We strip these at validate-time so the failure
// mode is "row 1099 has U+0085 at offset 4078" (seconds, locally) instead of
// "AlgorithmError: exit code 1" after a training launch (minutes, remote).
const FORBIDDEN_CONTROL_CODEPOINTS: ReadonlyArray<{ code: number; name: string }> = [
  // C0 control chars except whitespace we explicitly allow inside string values.
  // U+0009 TAB, U+000A LF, U+000D CR are kept since they appear in normal text.
  { code: 0x00, name: "NULL" },
  { code: 0x01, name: "START OF HEADING" },
  { code: 0x02, name: "START OF TEXT" },
  { code: 0x03, name: "END OF TEXT" },
  { code: 0x04, name: "END OF TRANSMISSION" },
  { code: 0x05, name: "ENQUIRY" },
  { code: 0x06, name: "ACKNOWLEDGE" },
  { code: 0x07, name: "BELL" },
  { code: 0x08, name: "BACKSPACE" },
  { code: 0x0b, name: "VERTICAL TAB" },
  { code: 0x0c, name: "FORM FEED" },
  { code: 0x0e, name: "SHIFT OUT" },
  { code: 0x0f, name: "SHIFT IN" },
  // 0x10..0x1F all rejected; named lookup falls back to "CONTROL CHARACTER".
  { code: 0x1c, name: "FILE SEPARATOR" },
  { code: 0x1d, name: "GROUP SEPARATOR" },
  { code: 0x1e, name: "RECORD SEPARATOR" },
  // C1 control range U+0080..U+009F (all invisible, often mojibake from
  // Windows-1252 smart quotes mis-tagged as Unicode codepoints).
  { code: 0x0085, name: "NEXT LINE" },
  // Unicode line/paragraph separators - cut lines too.
  { code: 0x2028, name: "LINE SEPARATOR" },
  { code: 0x2029, name: "PARAGRAPH SEPARATOR" },
];

function findControlChar(value: string): { codepoint: number; offset: number; name: string } | null {
  for (let i = 0; i < value.length; i++) {
    const code = value.charCodeAt(i);
    // Whitelist real whitespace; everything in C0 below 0x20 (besides tab/LF/CR) is hazardous.
    if (code < 0x20 && code !== 0x09 && code !== 0x0a && code !== 0x0d) {
      const named = FORBIDDEN_CONTROL_CODEPOINTS.find((entry) => entry.code === code);
      return { codepoint: code, offset: i, name: named?.name ?? "CONTROL CHARACTER" };
    }
    // C1 control range - all hazardous (NEL, etc.); JSON parsers accept these
    // but Python's splitlines() splits on several of them.
    if (code >= 0x80 && code <= 0x9f) {
      const named = FORBIDDEN_CONTROL_CODEPOINTS.find((entry) => entry.code === code);
      return { codepoint: code, offset: i, name: named?.name ?? "C1 CONTROL CHARACTER" };
    }
    if (code === 0x2028 || code === 0x2029) {
      const named = FORBIDDEN_CONTROL_CODEPOINTS.find((entry) => entry.code === code);
      return { codepoint: code, offset: i, name: named?.name ?? "LINE BREAK" };
    }
  }
  return null;
}

function formatCodepoint(code: number): string {
  return `U+${code.toString(16).toUpperCase().padStart(4, "0")}`;
}

function detectRowFormat(record: Record<string, unknown>): DatasetFormat | null {
  const input = record.input as Record<string, unknown> | undefined;
  if (
    input &&
    typeof input === "object" &&
    typeof input.prompt === "string" &&
    Array.isArray(input.assets) &&
    input.assets.length > 0 &&
    typeof record.output === "string"
  ) {
    return "document_ocr_jsonl";
  }
  if (
    typeof record.input === "string" ||
    typeof record.output === "string" ||
    !("input" in record) ||
    !("output" in record)
  ) {
    return "jsonl";
  }
  return null;
}

function validateDocumentOcrRow(record: Record<string, unknown>, rowNumber: number, errors: string[]): void {
  const input = record.input as Record<string, unknown>;
  const assets = input.assets as unknown[];
  if (typeof input.prompt !== "string" || input.prompt.trim().length === 0) {
    errors.push(`Row ${rowNumber}: OCR input.prompt must be a non-empty string`);
  } else {
    const hit = findControlChar(input.prompt);
    if (hit) {
      errors.push(
        `Row ${rowNumber}: OCR input.prompt contains invisible control character ${formatCodepoint(hit.codepoint)} (${hit.name}) at offset ${hit.offset} — strip or escape before uploading; some training paths split on it`,
      );
    }
  }
  if (typeof record.output !== "string" || record.output.trim().length === 0) {
    errors.push(`Row ${rowNumber}: OCR output must be a non-empty string`);
  } else {
    const hit = findControlChar(record.output);
    if (hit) {
      errors.push(
        `Row ${rowNumber}: OCR output contains invisible control character ${formatCodepoint(hit.codepoint)} (${hit.name}) at offset ${hit.offset} — strip or escape before uploading; some training paths split on it`,
      );
    }
  }
  if (assets.length > 8) {
    errors.push(`Row ${rowNumber}: OCR rows support at most 8 image assets`);
  }
  for (const [assetIndex, asset] of assets.entries()) {
    if (!asset || typeof asset !== "object" || Array.isArray(asset)) {
      errors.push(`Row ${rowNumber}: asset ${assetIndex + 1} must be an object`);
      continue;
    }
    const a = asset as Record<string, unknown>;
    if (a.type !== undefined && a.type !== "image") {
      errors.push(`Row ${rowNumber}: asset ${assetIndex + 1} type must be "image"`);
    }
    if (a.mime_type !== undefined && (
      typeof a.mime_type !== "string" || !DOCUMENT_OCR_MIME_TYPES.has(a.mime_type)
    )) {
      errors.push(
        `Row ${rowNumber}: asset ${assetIndex + 1} mime_type must be one of image/png, image/jpeg, image/webp`,
      );
    }
    if (a.page !== undefined && (
      typeof a.page !== "number" || !Number.isInteger(a.page) || a.page < 1
    )) {
      errors.push(`Row ${rowNumber}: asset ${assetIndex + 1} page must be a positive integer`);
    }
    const hasReference =
      (typeof a.data_uri === "string" && a.data_uri.trim().length > 0) ||
      (typeof a.uri === "string" && a.uri.trim().length > 0) ||
      (typeof a.path === "string" && a.path.trim().length > 0);
    if (!hasReference) {
      errors.push(`Row ${rowNumber}: asset ${assetIndex + 1} must include data_uri, uri, or path`);
    }
    if (typeof a.data_uri === "string") {
      const match = /^data:(image\/(?:png|jpeg|webp));base64,[A-Za-z0-9+/=\s]+$/.exec(a.data_uri);
      if (!match) {
        errors.push(
          `Row ${rowNumber}: asset ${assetIndex + 1} data_uri must be a base64 image/png, image/jpeg, or image/webp data URI`,
        );
      } else if (typeof a.mime_type === "string" && a.mime_type !== match[1]) {
        errors.push(`Row ${rowNumber}: asset ${assetIndex + 1} mime_type does not match data_uri media type`);
      }
    }
  }
}

function parseDatasetFormat(value: string | undefined): DatasetFormat | undefined {
  if (!value) return undefined;
  if (value === "jsonl" || value === "document_ocr_jsonl") return value;
  throw new Error("--format must be one of: jsonl, document_ocr_jsonl");
}

function validateDatasetFile(file: string, requestedFormat?: DatasetFormat): DatasetFormat {
  const lines = readFileSync(file, "utf8").split(/\r?\n/);
  const errors: string[] = [];
  let rowCount = 0;
  let detectedFormat: DatasetFormat | null = null;

  for (const [index, rawLine] of lines.entries()) {
    const line = rawLine.trim();
    if (!line) continue;

    rowCount += 1;
    const rowNumber = index + 1;
    let row: unknown;

    try {
      row = JSON.parse(line);
    } catch {
      errors.push(`Row ${rowNumber}: invalid JSON`);
      continue;
    }

    if (!row || typeof row !== "object" || Array.isArray(row)) {
      errors.push(`Row ${rowNumber}: expected an object with "input" and "output" fields`);
      continue;
    }

    const record = row as Record<string, unknown>;
    if ("messages" in record && !("input" in record) && !("output" in record)) {
      errors.push(
        `Row ${rowNumber}: found OpenAI SFT-style "messages"; Tuned Tensor datasets require flat "input" and "output" strings`,
      );
      continue;
    }

    const rowFormat = detectRowFormat(record);
    if (!rowFormat) {
      errors.push(
        `Row ${rowNumber}: expected text {"input": string, "output": string} or OCR {"input":{"prompt": string, "assets": [...]}, "output": string}`,
      );
      continue;
    }
    detectedFormat ??= rowFormat;
    if (requestedFormat && rowFormat !== requestedFormat) {
      errors.push(`Row ${rowNumber}: expected ${requestedFormat} row, found ${rowFormat}`);
      continue;
    }
    if (detectedFormat !== rowFormat) {
      errors.push(`Row ${rowNumber}: cannot mix ${detectedFormat} and ${rowFormat} rows in one dataset`);
      continue;
    }

    if (rowFormat === "document_ocr_jsonl") {
      validateDocumentOcrRow(record, rowNumber, errors);
      continue;
    }

    if (typeof record.input !== "string") {
      errors.push(`Row ${rowNumber}: missing string "input" field`);
    }
    if (typeof record.output !== "string") {
      errors.push(`Row ${rowNumber}: missing string "output" field`);
    }

    for (const field of ["input", "output"] as const) {
      const value = record[field];
      if (typeof value !== "string") continue;
      const hit = findControlChar(value);
      if (hit) {
        errors.push(
          `Row ${rowNumber}: field "${field}" contains invisible control character ${formatCodepoint(hit.codepoint)} (${hit.name}) at offset ${hit.offset} — strip or escape before uploading; some training paths split on it`,
        );
      }
    }
  }

  if (rowCount === 0) {
    errors.push("File contains no JSONL rows");
  }

  if (errors.length > 0) {
    const preview = errors.slice(0, 5).join("\n");
    const suffix = errors.length > 5 ? `\n...and ${errors.length - 5} more error(s)` : "";
    throw new Error(
      `Invalid dataset format. Each JSONL row must be a valid text or document OCR example.\n${preview}${suffix}`,
    );
  }
  return detectedFormat ?? requestedFormat ?? "jsonl";
}

export function registerDatasetsCommands(parent: Command) {
  const datasets = parent
    .command("datasets")
    .description("Manage datasets");

  datasets
    .command("list")
    .description("List datasets")
    .option("-p, --page <n>", "Page number", "1")
    .option("--per-page <n>", "Results per page", "20")
    .action(async (cmdOpts) => {
      const opts = parent.opts() as ClientOpts;
      const { data, meta } = await get<Dataset[]>(
        "/datasets",
        { page: cmdOpts.page, per_page: cmdOpts.perPage },
        opts,
      );

      if (isJsonMode()) return printJson({ data, meta });

      printTable(
        ["ID", "Name", "Rows", "Size", "Status", "Created"],
        data.map((d) => [
          shortId(d.id),
          truncate(d.name, 30),
          String(d.row_count),
          formatBytes(d.file_size_bytes),
          formatStatus(d.status),
          formatDate(d.created_at),
        ]),
        meta,
      );
    });

  datasets
    .command("get")
    .description("Show dataset details")
    .argument("<id>", "Dataset ID (full UUID or 4+ char prefix)")
    .action(async (id: string) => {
      const opts = parent.opts() as ClientOpts;
      const fullId = await resolveDatasetId(id, opts);
      const { data } = await get<Dataset>(`/datasets/${fullId}`, undefined, opts);

      if (isJsonMode()) return printJson(data);

      printDetail([
        ["ID", data.id],
        ["Name", data.name],
        ["Description", data.description ?? undefined],
        ["Format", data.format],
        ["Rows", String(data.row_count)],
        ["Size", formatBytes(data.file_size_bytes)],
        ["Status", formatStatus(data.status)],
        ["Created", formatDate(data.created_at)],
        ["Updated", formatDate(data.updated_at)],
      ]);

      if (data.validation_errors?.length) {
        console.log("\nValidation Errors:");
        for (const err of data.validation_errors) {
          console.log(`  - ${err}`);
        }
      }
    });

  datasets
    .command("upload")
    .description("Upload a JSONL dataset file")
    .argument("<file>", "Path to JSONL file")
    .option("-n, --name <name>", "Dataset name (defaults to filename)")
    .option("-d, --description <desc>", "Dataset description")
    .option("--format <format>", "Dataset format: jsonl or document_ocr_jsonl")
    .action(async (file: string, cmdOpts) => {
      const opts = parent.opts() as ClientOpts;

      if (!existsSync(file)) {
        printError(`File not found: ${file}`);
        process.exit(1);
      }

      const format = validateDatasetFile(file, parseDatasetFormat(cmdOpts.format));

      const name = cmdOpts.name || file.split("/").pop()!.replace(/\.jsonl?$/, "");
      const fileBytes = readFileSync(file);
      const uploadUrl = await post<DatasetUploadUrl>(
        "/datasets/upload-url",
        {
          name,
          description: cmdOpts.description ?? null,
          filename: file.split("/").pop()!,
          size: statSync(file).size,
          contentType: "application/jsonl",
          format,
        },
        opts,
      );

      const uploadRes = await fetch(uploadUrl.data.upload_url, {
        method: uploadUrl.data.method,
        headers: uploadUrl.data.headers ?? { "Content-Type": "application/jsonl" },
        body: new Blob([fileBytes], { type: "application/jsonl" }),
      });
      if (!uploadRes.ok) {
        throw new Error(`Dataset upload failed: ${uploadRes.status} ${uploadRes.statusText}`);
      }

      const { data } = await post<Dataset>(
        "/datasets/finalize",
        {
          path: uploadUrl.data.path,
          name,
          description: cmdOpts.description ?? null,
        },
        opts,
      );

      if (isJsonMode()) return printJson(data);
      printSuccess(
        `Dataset uploaded: ${data.name} (${shortId(data.id)}) — ${data.row_count} rows`,
      );
    });

  datasets
    .command("delete")
    .description("Delete a dataset")
    .argument("<id>", "Dataset ID (full UUID or 4+ char prefix)")
    .action(async (id: string) => {
      const opts = parent.opts() as ClientOpts;
      const fullId = await resolveDatasetId(id, opts);
      await del(`/datasets/${fullId}`, opts);

      if (isJsonMode()) return printJson({ id: fullId, deleted: true });
      printSuccess(`Dataset deleted: ${shortId(fullId)}`);
    });
}
