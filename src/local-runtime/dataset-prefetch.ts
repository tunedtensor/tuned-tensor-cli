import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { mkdir, readFile, realpath, stat } from "node:fs/promises";
import { createInterface } from "node:readline";
import { basename, isAbsolute, join, relative, resolve, sep } from "node:path";
import {
  baseModelRevisionSchema,
  datasetSourceSchema,
  fineTuneRunRequestSchema,
  type DatasetSource,
  type FineTuneRunRequest,
  type HuggingFaceDatasetSource,
  type LocalRunnerConfig,
  type BehaviorSpecExample,
  type SpecSnapshot,
} from "./contracts.js";
import { buildSystemMessage, datasetInputIdentity, exampleToChatRow } from "./dataset.js";
import { fileUri, writeFileAtomic, writeJson } from "./artifacts.js";
import { safeName } from "./prefetch.js";
import {
  minimalMachineLearningEnvironment,
  withHuggingFaceCacheEnvironment,
} from "./huggingface-cache.js";
import {
  buildBundledPythonCommand,
  runLoggedProcess,
  withBundledPythonEnvironment,
} from "./process-runner.js";
import type { LocalRunReporter } from "./run-reporter.js";

type Split = "training" | "validation" | "test";
const SPLITS: readonly Split[] = ["training", "validation", "test"];

/** The spec names a Hugging Face dataset that is not in the local cache yet. */
export class DatasetNotDownloadedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DatasetNotDownloadedError";
  }
}

export interface ResolvedHuggingFaceDataset {
  source: DatasetSource;
  localPaths: Partial<Record<Split, string>>;
}

export interface DatasetPrefetchReport extends DatasetSource {
  ok: true;
  status: "completed" | "verified";
  local_files_only: boolean;
  hub_cache: string;
  size_bytes: number;
  artifact_dir?: string;
  log_uri?: string;
  command?: string[];
}

type HuggingFaceDatasetPrebuilt = NonNullable<FineTuneRunRequest["dataset_prebuilt"]> & {
  huggingface: HuggingFaceDatasetSource;
};

export function huggingFaceDataset(
  request: Pick<FineTuneRunRequest, "dataset_prebuilt">,
): HuggingFaceDatasetPrebuilt | undefined {
  const dataset = request.dataset_prebuilt;
  return dataset?.huggingface ? dataset as HuggingFaceDatasetPrebuilt : undefined;
}

function datasetFiles(dataset: HuggingFaceDatasetPrebuilt): Partial<Record<Split, string>> {
  const files: Partial<Record<Split, string>> = {};
  for (const split of SPLITS) if (dataset[split]) files[split] = dataset[split];
  return files;
}

export function huggingFaceHubCache(modelCache?: string): string {
  return withHuggingFaceCacheEnvironment(process.env, modelCache).HF_HUB_CACHE!;
}

function repositoryDirectory(hubCache: string, repo: string): string {
  return join(hubCache, `datasets--${repo.replaceAll("/", "--")}`);
}

function isStrictDescendant(parent: string, child: string): boolean {
  const path = relative(parent, child);
  return path !== "" && path !== ".." && !path.startsWith(`..${sep}`) && !isAbsolute(path);
}

async function digestFile(path: string, size: number): Promise<{ sha256: string; gitSha1: string }> {
  const sha256 = createHash("sha256");
  const gitSha1 = createHash("sha1").update(`blob ${size}\0`);
  for await (const chunk of createReadStream(path)) {
    sha256.update(chunk);
    gitSha1.update(chunk);
  }
  return { sha256: sha256.digest("hex"), gitSha1: gitSha1.digest("hex") };
}

function describe(dataset: HuggingFaceDatasetSource): string {
  return `${dataset.repo}@${dataset.revision ?? "main"}`;
}

/**
 * Resolve a Hugging Face dataset from the local cache without network access.
 *
 * Each split file must be a symlink into the repo's content-addressed blob
 * store, and its digest must match the blob name, so a corrupted or tampered
 * cache entry fails here instead of silently training on different data.
 */
export async function resolveCachedHuggingFaceDataset(args: {
  dataset: HuggingFaceDatasetPrebuilt;
  modelCache?: string;
}): Promise<ResolvedHuggingFaceDataset> {
  const source = args.dataset.huggingface;
  const hubCache = huggingFaceHubCache(args.modelCache);
  const repository = repositoryDirectory(hubCache, source.repo);
  const prefetchHint = "Run `tt datasets prefetch` with this spec (for example `tt datasets prefetch tunedtensor.json`) to download it.";
  let revision = source.revision;
  if (!revision) {
    const ref = await readFile(join(repository, "refs", "main"), "utf8").catch(() => null);
    const parsed = baseModelRevisionSchema.safeParse(ref?.trim());
    if (!parsed.success) {
      throw new DatasetNotDownloadedError(
        `Hugging Face dataset ${describe(source)} is not in the local cache (${hubCache}). ${prefetchHint}`,
      );
    }
    revision = parsed.data;
  }
  const snapshot = join(repository, "snapshots", revision);
  const physicalRepository = await realpath(repository).catch(() => null);
  const blobs = physicalRepository ? join(physicalRepository, "blobs") : null;
  const files: Partial<Record<Split, DatasetSource["files"]["training"]>> = {};
  const localPaths: ResolvedHuggingFaceDataset["localPaths"] = {};
  for (const [split, path] of Object.entries(datasetFiles(args.dataset)) as Array<[Split, string]>) {
    const local = join(snapshot, ...path.split(/[/\\]/));
    const metadata = await stat(local).catch(() => null);
    if (!metadata || !physicalRepository || !blobs) {
      throw new DatasetNotDownloadedError(
        `Hugging Face dataset file ${path} (${split}) from ${source.repo}@${revision} is not in the local cache (${hubCache}). ${prefetchHint}`,
      );
    }
    const target = await realpath(local);
    // Normal caches link into content-addressed blobs. Caches without symlink
    // support (Windows without Developer Mode) hold plain copies in snapshots/.
    if (!metadata.isFile() || !isStrictDescendant(physicalRepository, target)) {
      throw new Error(`Cached dataset file ${local} must resolve inside its Hugging Face cache repository ${physicalRepository}.`);
    }
    if (metadata.size === 0) throw new Error(`Cached dataset file ${path} (${split}) is empty: ${local}`);
    const digests = await digestFile(target, metadata.size);
    const blobName = isStrictDescendant(blobs, target) ? basename(target).toLowerCase() : "";
    const expected = /^[0-9a-f]{64}$/.test(blobName) ? digests.sha256 : /^[0-9a-f]{40}$/.test(blobName) ? digests.gitSha1 : null;
    if (expected !== null && expected !== blobName) {
      throw new Error(
        `Cached dataset file ${path} (${split}) failed its checksum (${local}). Delete ${target} and run \`tt datasets prefetch\` again.`,
      );
    }
    files[split] = { path, size_bytes: metadata.size, sha256: digests.sha256 };
    localPaths[split] = local;
  }
  return {
    source: datasetSourceSchema.parse({
      provider: "huggingface",
      repo: source.repo,
      requested_revision: source.revision ?? null,
      revision,
      pinned: Boolean(source.revision),
      columns: source.columns ?? null,
      snapshot_path: snapshot,
      files,
    }),
    localPaths,
  };
}

type ConversionStats = NonNullable<NonNullable<DatasetSource["conversion"]>["validation"]>;

function mappedField(row: Record<string, unknown>, name: string): string | null {
  const value = row[name];
  if (typeof value === "string") return value.trim() ? value : null;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return null;
}

async function readMappedRecords(path: string, split: Split, columns: NonNullable<HuggingFaceDatasetSource["columns"]>) {
  const examples: BehaviorSpecExample[] = [];
  let records = 0;
  let skipped = 0;
  const availableFields = new Set<string>();
  // Stream: Hub splits can exceed the maximum string length of a single read.
  const lines = createInterface({ input: createReadStream(path, { encoding: "utf8" }), crlfDelay: Infinity });
  let index = -1;
  for await (const line of lines) {
    index += 1;
    if (!line.trim()) continue;
    records += 1;
    let value: unknown;
    try {
      value = JSON.parse(line);
    } catch (error) {
      throw new Error(`${split} line ${index + 1}: malformed JSON in ${path}`, { cause: error });
    }
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      throw new Error(`${split} line ${index + 1}: expected a JSON object in ${path}`);
    }
    const row = value as Record<string, unknown>;
    for (const name of Object.keys(row)) availableFields.add(name);
    const input = mappedField(row, columns.input);
    const output = mappedField(row, columns.output);
    if (input === null || output === null) {
      skipped += 1;
      continue;
    }
    examples.push({ input, output });
  }
  // Diagnose a misspelled mapping only after considering all records. A sparse
  // first record follows the same skip rule as a sparse record anywhere else.
  if (records > 0) {
    for (const name of [columns.input, columns.output]) {
      if (!availableFields.has(name)) {
        throw new Error(
          `${split} records have no "${name}" field (dataset_prebuilt.huggingface.columns). `
          + `Available fields: ${[...availableFields].join(", ")}`,
        );
      }
    }
  }
  return { examples, records, skipped };
}

/**
 * Convert plain JSONL records into TT chat rows with the spec's compiled
 * system message. Hub datasets cannot be edited in place, so the conversion
 * applies three deterministic, recorded rules: records with a missing or blank
 * mapped field are skipped, evaluation splits keep the first of any duplicate
 * input, and training drops inputs that also appear in an evaluation split so
 * held-out metrics stay held out. Output paths are content-addressed by the
 * source digests, mapping and system message, so conversions are reused.
 */
async function convertMappedDataset(args: {
  resolved: ResolvedHuggingFaceDataset;
  columns: NonNullable<HuggingFaceDatasetSource["columns"]>;
  spec: SpecSnapshot;
  outputRoot: string;
}): Promise<{ localPaths: ResolvedHuggingFaceDataset["localPaths"]; conversion: NonNullable<DatasetSource["conversion"]> }> {
  const system = buildSystemMessage(args.spec);
  const splits = Object.keys(args.resolved.localPaths) as Split[];
  const key = createHash("sha256")
    .update(JSON.stringify({
      version: 1,
      sources: splits.map((split) => [split, args.resolved.source.files[split]?.sha256]),
      columns: args.columns,
      system,
    }))
    .digest("hex")
    .slice(0, 16);
  const loaded = new Map<Split, Awaited<ReturnType<typeof readMappedRecords>>>();
  for (const split of splits) {
    loaded.set(split, await readMappedRecords(args.resolved.localPaths[split]!, split, args.columns));
  }
  const evaluationInputs = new Set<string>();
  const kept = new Map<Split, BehaviorSpecExample[]>();
  const conversion: Partial<Record<Split, ConversionStats>> = {};
  for (const split of splits.filter((name) => name !== "training")) {
    const seen = new Set<string>();
    const data = loaded.get(split)!;
    const examples = data.examples.filter((example) => {
      const identity = datasetInputIdentity(example.input);
      if (seen.has(identity)) return false;
      seen.add(identity);
      evaluationInputs.add(identity);
      return true;
    });
    kept.set(split, examples);
    conversion[split] = {
      records: data.records,
      kept: examples.length,
      skipped_missing_fields: data.skipped,
      dropped_duplicate_inputs: data.examples.length - examples.length,
      dropped_eval_overlap: 0,
    };
  }
  const training = loaded.get("training")!;
  const trainingExamples = training.examples.filter((example) => !evaluationInputs.has(datasetInputIdentity(example.input)));
  kept.set("training", trainingExamples);
  conversion.training = {
    records: training.records,
    kept: trainingExamples.length,
    skipped_missing_fields: training.skipped,
    dropped_duplicate_inputs: 0,
    dropped_eval_overlap: training.examples.length - trainingExamples.length,
  };
  await mkdir(args.outputRoot, { recursive: true });
  const localPaths: ResolvedHuggingFaceDataset["localPaths"] = {};
  for (const split of splits) {
    const examples = kept.get(split)!;
    if (examples.length === 0) {
      throw new Error(`No usable ${split} records remain after applying dataset_prebuilt.huggingface.columns.`);
    }
    const output = join(args.outputRoot, `${split}-${key}.jsonl`);
    const content = `${examples.map((example) => JSON.stringify(exampleToChatRow(args.spec, example))).join("\n")}\n`;
    // A content-addressed name does not guarantee that a local file stayed intact.
    // Regenerate changed conversions from the independently verified source.
    if (await readFile(output, "utf8").catch(() => null) !== content) {
      await writeFileAtomic(output, content);
    }
    localPaths[split] = output;
  }
  return { localPaths, conversion: conversion as NonNullable<DatasetSource["conversion"]> };
}

/**
 * Replace a Hugging Face `dataset_prebuilt` with its cached local files and
 * record the resolved provenance on the request. Requests with local files are
 * returned unchanged, so the call is idempotent.
 */
export async function resolveRequestDataset(
  request: FineTuneRunRequest,
  config: Pick<LocalRunnerConfig, "paths" | "artifactRoot">,
): Promise<FineTuneRunRequest> {
  const dataset = huggingFaceDataset(request);
  if (!dataset) return request;
  const resolved = await resolveCachedHuggingFaceDataset({ dataset, modelCache: config.paths.modelCache });
  let localPaths = resolved.localPaths;
  let source = resolved.source;
  const columns = dataset.huggingface.columns;
  if (columns) {
    const converted = await convertMappedDataset({
      resolved,
      columns,
      spec: request.spec_snapshot,
      outputRoot: resolve(config.artifactRoot, "datasets", "huggingface", safeName(dataset.huggingface.repo), resolved.source.revision),
    });
    localPaths = converted.localPaths;
    source = datasetSourceSchema.parse({ ...source, conversion: converted.conversion });
  }
  return fineTuneRunRequestSchema.parse({
    ...request,
    dataset_prebuilt: {
      ...localPaths,
      format: dataset.format,
    },
    dataset_source: source,
  });
}

/**
 * Download the spec's Hugging Face dataset files into the shared cache, the
 * same way `tt models prefetch` fetches a base model. With `localOnly`, only
 * verify what is already cached (no Python, no network).
 */
export async function prefetchHuggingFaceDataset(args: {
  request: Pick<FineTuneRunRequest, "dataset_prebuilt">;
  config: LocalRunnerConfig;
  reporter?: LocalRunReporter;
  localOnly?: boolean;
}): Promise<DatasetPrefetchReport> {
  const dataset = huggingFaceDataset(args.request);
  if (!dataset) {
    throw new Error(
      "This spec has no dataset_prebuilt.huggingface source. Local dataset files need no download; "
      + "set dataset_prebuilt.huggingface.repo to train on a Hugging Face dataset.",
    );
  }
  const modelCache = args.config.paths.modelCache ? resolve(args.config.paths.modelCache) : undefined;
  const hubCache = huggingFaceHubCache(modelCache);
  if (args.localOnly) {
    const resolved = await resolveCachedHuggingFaceDataset({ dataset, modelCache });
    return {
      ok: true,
      status: "verified",
      local_files_only: true,
      hub_cache: hubCache,
      size_bytes: Object.values(resolved.source.files).reduce((total, file) => total + (file?.size_bytes ?? 0), 0),
      ...resolved.source,
    };
  }

  const artifactDir = resolve(
    args.config.artifactRoot,
    "dataset-prefetch",
    `${safeName(dataset.huggingface.repo)}-${new Date().toISOString().replace(/[:.]/g, "-")}`,
  );
  await mkdir(artifactDir, { recursive: true });
  const inputPath = join(artifactDir, "prefetch-input.json");
  const outputPath = join(artifactDir, "prefetch-output.json");
  const logPath = join(artifactDir, "prefetch.log");
  await writeJson(inputPath, {
    repo: dataset.huggingface.repo,
    ...(dataset.huggingface.revision ? { revision: dataset.huggingface.revision } : {}),
    files: datasetFiles(dataset),
    ...(modelCache ? { model_cache: modelCache } : {}),
  });
  const entrypoint = buildBundledPythonCommand("prefetch_dataset.py", ["--input", inputPath, "--output", outputPath]);
  await args.reporter?.onEvent?.({
    stage: "dataset_prefetch",
    status: "running",
    message: "Downloading Hugging Face dataset files.",
    details: { repo: dataset.huggingface.repo, revision: dataset.huggingface.revision ?? "main", hub_cache: hubCache, log_path: logPath },
  });
  const { exitCode, stderr } = await runLoggedProcess({
    command: entrypoint.command,
    commandArgs: entrypoint.commandArgs,
    env: withBundledPythonEnvironment(withHuggingFaceCacheEnvironment(minimalMachineLearningEnvironment(process.env), modelCache)),
    logPath,
    reporter: args.reporter ? { ...args.reporter, verbose: true } : undefined,
    stage: "dataset_prefetch",
  });
  if (exitCode !== 0) {
    const reason = stderr.trim().split("\n").filter(Boolean).at(-1);
    throw new Error(
      `Dataset prefetch exited with code ${exitCode}${reason ? `: ${reason}` : ""}. See ${logPath}. `
      + "Rerunning resumes from the files already downloaded.",
    );
  }
  const output = JSON.parse(await readFile(outputPath, "utf8")) as {
    snapshot_revision?: string;
    files?: Record<string, { sha256?: string }>;
  };
  const revision = baseModelRevisionSchema.parse(output.snapshot_revision);
  // Verify independently from Node; this is the same check every run performs.
  const resolved = await resolveCachedHuggingFaceDataset({
    dataset: { ...dataset, huggingface: { ...dataset.huggingface, revision } },
    modelCache,
  });
  for (const [split, file] of Object.entries(resolved.source.files)) {
    if (output.files?.[split]?.sha256 !== file?.sha256) {
      throw new Error(`Dataset prefetch reported a different digest for ${split} than the cached file.`);
    }
  }
  const source = datasetSourceSchema.parse({
    ...resolved.source,
    requested_revision: dataset.huggingface.revision ?? null,
    pinned: Boolean(dataset.huggingface.revision),
  });
  const sizeBytes = Object.values(source.files).reduce((total, file) => total + (file?.size_bytes ?? 0), 0);
  await args.reporter?.onEvent?.({
    stage: "dataset_prefetch",
    status: "completed",
    message: "Dataset files are available in the local Hugging Face cache.",
    details: { repo: source.repo, revision: source.revision, pinned: source.pinned, size_bytes: sizeBytes },
  });
  return {
    ok: true,
    status: "completed",
    local_files_only: false,
    hub_cache: hubCache,
    size_bytes: sizeBytes,
    ...source,
    artifact_dir: artifactDir,
    log_uri: fileUri(logPath),
    command: entrypoint.displayCommand,
  };
}

/**
 * Make sure a Hugging Face dataset is cached before a run, downloading it with
 * progress when it is missing. Returns the request with its dataset resolved
 * to local files, so later validation does not hash the files again.
 */
export async function ensureDatasetCached<T extends FineTuneRunRequest>(args: {
  request: T;
  config: LocalRunnerConfig;
  reporter?: LocalRunReporter;
}): Promise<{ request: FineTuneRunRequest; prefetch: DatasetPrefetchReport | null }> {
  if (!huggingFaceDataset(args.request)) return { request: args.request, prefetch: null };
  try {
    return { request: await resolveRequestDataset(args.request, args.config), prefetch: null };
  } catch (error) {
    if (!(error instanceof DatasetNotDownloadedError) || args.config.dryRun) throw error;
  }
  const prefetch = await prefetchHuggingFaceDataset(args);
  return { request: await resolveRequestDataset(args.request, args.config), prefetch };
}
