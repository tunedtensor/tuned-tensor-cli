import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { createInterface } from "node:readline";
import { join } from "node:path";
import { asyncBufferFromFile, parquetMetadataAsync, parquetReadObjects } from "hyparquet";
import { compressors } from "hyparquet-compressors";
import type { AssetManifest } from "./assets.js";
import { datasetInputIdentity } from "./dataset.js";
import { columnNames, firstExchange, mappedField, type DatasetColumns } from "./dataset-prefetch.js";

/**
 * Read a downloaded Hugging Face dataset file and summarize what is in it:
 * a deterministic sample of rows plus per-field statistics, so people (and the
 * agent) can judge the data before training on it. Reads only the local
 * cache recorded by a download manifest; never touches the network.
 */

export const DEFAULT_PREVIEW_ROWS = 5;
export const MAX_PREVIEW_ROWS = 20;
/** Statistics stop after this many records so huge splits stay fast. */
export const DEFAULT_PREVIEW_SCAN_LIMIT = 50_000;
/** Serialized size of all samples; keeps the report inside the agent's tool-output limit. */
const SAMPLE_CHARACTER_BUDGET = 16_000;
const MAX_VALUE_CHARACTERS = 800;
const MIN_VALUE_CHARACTERS = 40;
/** Wide tables report statistics for the first fields only. */
const MAX_REPORTED_FIELDS = 50;
const MAX_LISTED_FILES = 20;
/** Below the agent's 32,000-character tool-output limit. */
const MAX_REPORT_CHARACTERS = 30_000;
const SUPPORTED_FILE = /\.(jsonl|parquet)$/i;

type Format = "jsonl" | "parquet";

interface FieldStats {
  present: number;
  blank: number;
  types: Set<string>;
  chars: number[];
  items: number[];
}

export interface DatasetPreviewField {
  name: string;
  present: number;
  blank: number;
  types: string[];
  chars?: Distribution;
  items?: Distribution;
}

interface Distribution {
  min: number;
  median: number;
  p95: number;
  max: number;
}

export interface DatasetPreviewMapping {
  columns: DatasetColumns;
  missing_columns: string[];
  usable: number;
  skipped_missing_fields: number;
  duplicate_inputs: number;
  input_equals_output: number;
  /** Training rejects JSONL files with malformed lines; fix them before using this mapping. */
  blocked_by_malformed_lines?: boolean;
}

export interface DatasetPreview {
  repo: string;
  revision: string;
  file: string;
  format: Format;
  size_bytes: number;
  /** Records in the file. Past the scan limit, JSONL lines are counted without being parsed. */
  records: number;
  scanned: number;
  scan_complete: boolean;
  malformed_lines: number;
  fields: DatasetPreviewField[];
  fields_omitted?: number;
  duplicate_records: number;
  mapping?: DatasetPreviewMapping;
  samples: Array<{ index: number; record: Record<string, unknown> }>;
  samples_omitted?: number;
  other_files: string[];
  other_files_omitted?: number;
}

function valueType(value: unknown): string {
  if (typeof value === "string") return "string";
  if (typeof value === "number") return Number.isInteger(value) ? "integer" : "number";
  if (typeof value === "bigint") return "integer";
  if (typeof value === "boolean") return "boolean";
  if (value instanceof Uint8Array) return "binary";
  if (value instanceof Date) return "date";
  if (Array.isArray(value)) return "array";
  return "object";
}

function distribution(values: number[]): Distribution | undefined {
  if (values.length === 0) return undefined;
  const sorted = [...values].sort((a, b) => a - b);
  const at = (q: number) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))]!;
  return { min: sorted[0]!, median: at(0.5), p95: at(0.95), max: sorted.at(-1)! };
}

/** Record identity that ignores key order, so reordered duplicates still match. */
function stableJson(value: unknown): string {
  return JSON.stringify(value, (_key, item: unknown) => {
    if (typeof item === "bigint") return item.toString();
    if (item instanceof Uint8Array) return createHash("sha256").update(item).digest("hex");
    if (item && typeof item === "object" && !Array.isArray(item) && !(item instanceof Date)) {
      return Object.fromEntries(Object.entries(item).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)));
    }
    return item;
  });
}

/** Bound a sampled value for display: long strings, wide objects and binary columns are elided. */
function truncateValue(value: unknown, limit: number, depth = 0): unknown {
  if (typeof value === "string") {
    return value.length > limit ? `${value.slice(0, limit)}… [+${value.length - limit} chars]` : value;
  }
  if (typeof value === "bigint") return value.toString();
  if (value instanceof Date) return value.toISOString();
  if (value instanceof Uint8Array) return `<binary ${value.byteLength} bytes>`;
  if (value === null || typeof value !== "object") return value;
  if (depth >= 4) return "<nested>";
  if (Array.isArray(value)) {
    const items = value.slice(0, 8).map((item) => truncateValue(item, limit, depth + 1));
    return value.length > 8 ? [...items, `… [+${value.length - 8} items]`] : items;
  }
  const entries = Object.entries(value);
  const kept = Object.fromEntries(entries.slice(0, 20).map(([key, item]) => [key, truncateValue(item, limit, depth + 1)]));
  return entries.length > 20 ? { ...kept, "…": `+${entries.length - 20} fields` } : kept;
}

function boundSamples(samples: Array<{ index: number; record: Record<string, unknown> }>, limit: number) {
  return samples.map(({ index, record }) => ({ index, record: truncateValue(record, limit) as Record<string, unknown> }));
}

/** Deterministic PRNG so the same file always yields the same sample. */
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface Scan {
  records: AsyncGenerator<Record<string, unknown>>;
  /** Called after iteration stops: total records in the file and malformed lines seen. */
  finish(scanned: number): Promise<{ records: number; malformed: number; complete: boolean }>;
}

async function scanParquet(path: string, limit: number): Promise<Scan> {
  const file = await asyncBufferFromFile(path);
  const metadata = await parquetMetadataAsync(file);
  const total = Number(metadata.num_rows);
  async function* records() {
    let rowStart = 0;
    for (const group of metadata.row_groups) {
      if (rowStart >= limit) return;
      const rowEnd = Math.min(rowStart + Number(group.num_rows), limit);
      yield* await parquetReadObjects({ file, metadata, compressors, rowStart, rowEnd });
      rowStart += Number(group.num_rows);
    }
  }
  return {
    records: records(),
    finish: async (scanned) => ({ records: total, malformed: 0, complete: scanned >= total }),
  };
}

async function countNonEmptyLines(path: string): Promise<number> {
  let count = 0;
  let content = false;
  for await (const chunk of createReadStream(path) as AsyncIterable<Buffer>) {
    for (const byte of chunk) {
      if (byte === 0x0a) {
        if (content) count += 1;
        content = false;
      } else if (byte !== 0x20 && byte !== 0x09 && byte !== 0x0d) {
        content = true;
      }
    }
  }
  return content ? count + 1 : count;
}

async function scanJsonl(path: string, limit: number): Promise<Scan> {
  const lines = createInterface({ input: createReadStream(path, { encoding: "utf8" }), crlfDelay: Infinity })[Symbol.asyncIterator]();
  let malformed = 0;
  let nonEmpty = 0;
  async function* records() {
    let parsed = 0;
    while (parsed < limit) {
      const next = await lines.next();
      if (next.done) return;
      if (!next.value.trim()) continue;
      nonEmpty += 1;
      let value: unknown;
      try {
        value = JSON.parse(next.value);
      } catch {
        malformed += 1;
        continue;
      }
      if (!value || typeof value !== "object" || Array.isArray(value)) {
        malformed += 1;
        continue;
      }
      parsed += 1;
      yield value as Record<string, unknown>;
    }
  }
  return {
    records: records(),
    finish: async () => {
      const next = await lines.next();
      await lines.return?.();
      if (next.done) return { records: nonEmpty - malformed, malformed, complete: true };
      // Stopped at the scan limit: count lines on raw bytes instead of
      // decoding and splitting the rest of a possibly huge file.
      return { records: await countNonEmptyLines(path) - malformed, malformed, complete: false };
    },
  };
}

function chooseFile(asset: AssetManifest, requested?: string): { file: string; supported: string[] } {
  const supported = Object.keys(asset.files).filter((file) => SUPPORTED_FILE.test(file)).sort();
  if (supported.length === 0) {
    throw new Error(`${asset.repo}@${asset.revision} has no downloaded .jsonl or .parquet files to preview.`);
  }
  if (requested) {
    if (!supported.includes(requested)) {
      throw new Error(`${requested} is not a downloaded .jsonl or .parquet file of ${asset.repo}@${asset.revision}. Downloaded files: ${supported.join(", ")}`);
    }
    return { file: requested, supported };
  }
  const training = supported.find((file) => /^train/i.test(file.split("/").at(-1)!))
    ?? supported.find((file) => file.split("/").slice(0, -1).some((part) => /^train(ing)?$/i.test(part)));
  return { file: training ?? supported[0]!, supported };
}

export async function previewDataset(args: {
  asset: AssetManifest;
  file?: string;
  rows?: number;
  columns?: DatasetColumns;
  scanLimit?: number;
}): Promise<DatasetPreview> {
  const { file, supported } = chooseFile(args.asset, args.file);
  const path = join(args.asset.snapshot_path, ...file.split("/"));
  const metadata = await stat(path).catch(() => null);
  if (!metadata?.isFile()) {
    throw new Error(`${file} is missing from ${args.asset.snapshot_path}. Download the dataset again before previewing it.`);
  }
  const format: Format = file.toLowerCase().endsWith(".parquet") ? "parquet" : "jsonl";
  const rows = Math.max(1, Math.min(MAX_PREVIEW_ROWS, args.rows ?? DEFAULT_PREVIEW_ROWS));
  const scanLimit = Math.max(1, args.scanLimit ?? DEFAULT_PREVIEW_SCAN_LIMIT);
  const scan = format === "parquet" ? await scanParquet(path, scanLimit) : await scanJsonl(path, scanLimit);

  const fields = new Map<string, FieldStats>();
  const omittedFields = new Set<string>();
  const seenRecords = new Set<string>();
  let duplicateRecords = 0;
  const mappingColumns = args.columns ? columnNames(args.columns) : [];
  const seenInputs = new Set<string>();
  const mapping: DatasetPreviewMapping | undefined = args.columns ? {
    columns: args.columns, missing_columns: [] as string[], usable: 0, skipped_missing_fields: 0, duplicate_inputs: 0, input_equals_output: 0,
  } : undefined;
  const random = mulberry32(0x5eed);
  const reservoir: Array<{ index: number; record: Record<string, unknown> }> = [];

  let scanned = 0;
  for await (const record of scan.records) {
    const index = scanned;
    scanned += 1;
    for (const [name, value] of Object.entries(record)) {
      let stats = fields.get(name);
      if (!stats) {
        if (fields.size >= MAX_REPORTED_FIELDS) {
          omittedFields.add(name);
          continue;
        }
        stats = { present: 0, blank: 0, types: new Set(), chars: [], items: [] };
        fields.set(name, stats);
      }
      if (value === null || value === undefined) continue;
      stats.present += 1;
      stats.types.add(valueType(value));
      if (typeof value === "string") {
        stats.chars.push(value.length);
        if (!value.trim()) stats.blank += 1;
      } else if (Array.isArray(value)) {
        stats.items.push(value.length);
        if (value.length === 0) stats.blank += 1;
      }
    }
    const identity = createHash("sha1").update(stableJson(record)).digest("hex");
    if (seenRecords.has(identity)) duplicateRecords += 1;
    else seenRecords.add(identity);
    if (mapping && args.columns) {
      const example = "messages" in args.columns
        ? firstExchange(record[args.columns.messages])
        : { input: mappedField(record, args.columns.input), output: mappedField(record, args.columns.output) };
      if (example === null || example.input === null || example.output === null) {
        mapping.skipped_missing_fields += 1;
      } else {
        mapping.usable += 1;
        const input = datasetInputIdentity(example.input);
        if (seenInputs.has(input)) mapping.duplicate_inputs += 1;
        else seenInputs.add(input);
        if (input === datasetInputIdentity(example.output)) mapping.input_equals_output += 1;
      }
    }
    // Reservoir sampling keeps a uniform sample without knowing the length up front.
    if (reservoir.length < rows) {
      reservoir.push({ index, record });
    } else {
      const slot = Math.floor(random() * scanned);
      if (slot < rows) reservoir[slot] = { index, record };
    }
  }
  const finished = await scan.finish(scanned);
  if (mapping) mapping.missing_columns = mappingColumns.filter((name) => !fields.has(name) && !omittedFields.has(name));
  // Training conversion stops at the first malformed JSONL line, so a usable
  // count would overstate what a run can use.
  if (mapping && finished.malformed > 0) mapping.blocked_by_malformed_lines = true;

  // Nested chat turns hold many strings per field, so shrink the per-value
  // limit until the serialized sample fits rather than estimating it.
  reservoir.sort((a, b) => a.index - b.index);
  let valueLimit = MAX_VALUE_CHARACTERS;
  let samples = boundSamples(reservoir, valueLimit);
  while (JSON.stringify(samples).length > SAMPLE_CHARACTER_BUDGET && valueLimit > MIN_VALUE_CHARACTERS) {
    valueLimit = Math.max(MIN_VALUE_CHARACTERS, Math.floor(valueLimit / 2));
    samples = boundSamples(reservoir, valueLimit);
  }
  const otherFiles = supported.filter((name) => name !== file);
  const report: DatasetPreview = {
    repo: args.asset.repo,
    revision: args.asset.revision,
    file,
    format,
    size_bytes: metadata.size,
    records: finished.records,
    scanned,
    scan_complete: finished.complete,
    malformed_lines: finished.malformed,
    fields: [...fields.entries()].map(([name, stats]) => {
      const chars = distribution(stats.chars);
      const items = distribution(stats.items);
      return {
        name,
        present: stats.present,
        blank: stats.blank,
        types: [...stats.types].sort(),
        ...(chars ? { chars } : {}),
        ...(items ? { items } : {}),
      };
    }),
    ...(omittedFields.size > 0 ? { fields_omitted: omittedFields.size } : {}),
    duplicate_records: duplicateRecords,
    ...(mapping ? { mapping } : {}),
    samples,
    other_files: otherFiles.slice(0, MAX_LISTED_FILES),
    ...(otherFiles.length > MAX_LISTED_FILES ? { other_files_omitted: otherFiles.length - MAX_LISTED_FILES } : {}),
  };
  // Value truncation cannot bound deeply nested or very wide rows on its own;
  // drop trailing samples until the whole report fits.
  while (report.samples.length > 0 && JSON.stringify(report).length > MAX_REPORT_CHARACTERS) {
    report.samples = report.samples.slice(0, -1);
    report.samples_omitted = reservoir.length - report.samples.length;
  }
  return report;
}
