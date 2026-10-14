import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile, readdir, realpath, stat } from "node:fs/promises";
import { basename, dirname, join, relative, sep } from "node:path";
import { getTunedTensorHome } from "../paths.js";
import { writeJsonAtomic } from "./artifacts.js";
import type { FineTuneRunRequest, LocalRunnerConfig } from "./contracts.js";
import { withHuggingFaceCacheEnvironment } from "./huggingface-cache.js";
import { resolveRequestedBaseModelRevision, resolveTrainingModel } from "./model-registry.js";
import type { LocalRunReporter } from "./run-reporter.js";

/**
 * Downloaded Hugging Face assets.
 *
 * Files stay in the Hugging Face cache, which is already content-addressed.
 * TT adds two small records on top of it:
 *
 * - a manifest per download under `~/.tuned-tensor/assets/`, so people can find
 *   what they downloaded and where it lives (`tt assets list|path|verify`);
 * - `tunedtensor.lock.json` beside a spec, which pins the exact revision and
 *   file digests a project trains on. Runs refuse to use anything else.
 *
 * A digest is the file's Hugging Face content address: `sha256:<hex>` for
 * LFS files and `git-sha1:<hex>` for small files stored in git.
 */
export type AssetKind = "dataset" | "model";

export interface AssetSnapshot {
  repo: string;
  revision: string;
  files: Record<string, string>;
}

export interface AssetManifest extends AssetSnapshot {
  version: 1;
  kind: AssetKind;
  fetched_at: string;
  snapshot_path: string;
  size_bytes: number;
}

export interface SpecLock {
  version: 1;
  dataset?: AssetSnapshot;
  base_model?: AssetSnapshot;
}

export type LockKey = keyof Omit<SpecLock, "version">;

const COMMIT_SHA = /^[0-9a-f]{40}$/;

export class SpecLockMismatchError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SpecLockMismatchError";
  }
}

export function assetsRoot(env: NodeJS.ProcessEnv = process.env): string {
  return join(getTunedTensorHome(env), "assets");
}

/** `<hub>/datasets--org--name/snapshots/<revision>`, as huggingface_hub lays it out. */
export function snapshotDirectory(hubCache: string, kind: AssetKind, repo: string, revision: string): string {
  return join(hubCache, `${kind}s--${repo.replaceAll("/", "--")}`, "snapshots", revision);
}

async function contentDigest(path: string, algorithm: "sha256" | "git-sha1"): Promise<string> {
  const size = (await stat(path)).size;
  const hash = algorithm === "sha256" ? createHash("sha256") : createHash("sha1").update(`blob ${size}\0`);
  // Match the bounded buffer used by model verification for large weight files.
  for await (const chunk of createReadStream(path, { highWaterMark: 8 * 1024 * 1024 })) hash.update(chunk);
  return `${algorithm}:${hash.digest("hex")}`;
}

async function listFiles(root: string, directory = root): Promise<string[]> {
  const files: string[] = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const child = join(directory, entry.name);
    if (entry.isDirectory()) files.push(...await listFiles(root, child));
    else files.push(relative(root, child).split(sep).join("/"));
  }
  return files.sort();
}

/**
 * Hash the actual snapshot bytes. A blob filename selects the digest format,
 * but cannot prove that the cached bytes have remained intact.
 */
export async function snapshotDigests(
  snapshot: string,
  paths?: readonly string[],
): Promise<{ files: Record<string, string>; sizeBytes: number }> {
  const files: Record<string, string> = {};
  let sizeBytes = 0;
  for (const path of paths ?? await listFiles(snapshot)) {
    const local = join(snapshot, ...path.split("/"));
    const target = await realpath(local);
    sizeBytes += (await stat(target)).size;
    const blob = target !== local && basename(dirname(target)) === "blobs" ? basename(target).toLowerCase() : "";
    files[path] = await contentDigest(target, /^[0-9a-f]{40}$/.test(blob) ? "git-sha1" : "sha256");
    if (/^(?:[0-9a-f]{40}|[0-9a-f]{64})$/.test(blob) && files[path].split(":")[1] !== blob) {
      throw new SpecLockMismatchError(`Cached file ${path} failed its checksum. Remove the corrupt blob ${target} and prefetch it again.`);
    }
  }
  return { files, sizeBytes };
}

function manifestPath(kind: AssetKind, repo: string, revision: string, env?: NodeJS.ProcessEnv): string {
  return join(assetsRoot(env), `${kind}s`, repo.replaceAll("/", "--"), revision, "manifest.json");
}

/** Record a completed download so people (and TT) can find and verify it later. */
export async function recordAssetDownload(args: {
  kind: AssetKind;
  repo: string;
  revision: string;
  snapshotPath: string;
  paths?: readonly string[];
  env?: NodeJS.ProcessEnv;
}): Promise<{ manifest: AssetManifest; path: string }> {
  const path = manifestPath(args.kind, args.repo, args.revision, args.env);
  let paths = args.paths;
  if (paths) {
    // Dataset downloads fetch only the named split files; keep files that an
    // earlier download of the same revision already recorded.
    const previous = Object.keys((await readJsonIfPresent<AssetManifest>(path))?.files ?? {});
    const stillCached = await Promise.all(previous.map(async (file) =>
      await stat(join(args.snapshotPath, ...file.split("/"))).then(() => file, () => null)));
    paths = [...new Set([...stillCached.filter((file): file is string => file !== null), ...paths])].sort();
  }
  const { files, sizeBytes } = await snapshotDigests(args.snapshotPath, paths);
  const manifest: AssetManifest = {
    version: 1,
    kind: args.kind,
    repo: args.repo,
    revision: args.revision,
    fetched_at: new Date().toISOString(),
    snapshot_path: args.snapshotPath,
    size_bytes: sizeBytes,
    files,
  };
  await writeJsonAtomic(path, manifest);
  return { manifest, path };
}

async function readJsonIfPresent<T>(path: string): Promise<T | undefined> {
  try {
    return JSON.parse(await readFile(path, "utf8")) as T;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
    throw error;
  }
}

export async function listAssets(env?: NodeJS.ProcessEnv): Promise<Array<AssetManifest & { manifest_path: string }>> {
  const root = assetsRoot(env);
  const found: Array<AssetManifest & { manifest_path: string }> = [];
  for (const kind of ["datasets", "models"]) {
    const repos = await readdir(join(root, kind)).catch(() => []);
    for (const repo of repos) {
      for (const revision of await readdir(join(root, kind, repo)).catch(() => [])) {
        const path = join(root, kind, repo, revision, "manifest.json");
        const manifest = await readJsonIfPresent<AssetManifest>(path);
        if (manifest) found.push({ ...manifest, manifest_path: path });
      }
    }
  }
  return found.sort((a, b) => b.fetched_at.localeCompare(a.fetched_at));
}

/** Find a download by `org/name` or `org/name@revision`; newest first. */
export async function findAsset(reference: string, env?: NodeJS.ProcessEnv) {
  const [repo, revision] = reference.split("@");
  const match = (await listAssets(env)).find((asset) =>
    asset.repo === repo && (!revision || asset.revision.startsWith(revision.toLowerCase())));
  if (!match) throw new Error(`No downloaded asset matches ${reference}. Run \`tt assets list\` to see downloads.`);
  return match;
}

/** Re-hash every recorded file and compare it with the manifest. */
export async function verifyAsset(manifest: AssetManifest): Promise<{ ok: boolean; problems: string[] }> {
  const problems: string[] = [];
  for (const [path, expected] of Object.entries(manifest.files)) {
    const local = join(manifest.snapshot_path, ...path.split("/"));
    const target = await realpath(local).catch(() => null);
    if (!target) {
      problems.push(`${path}: missing from ${manifest.snapshot_path}`);
      continue;
    }
    const algorithm = expected.startsWith("git-sha1:") ? "git-sha1" : "sha256";
    if (await contentDigest(target, algorithm) !== expected) problems.push(`${path}: content does not match ${expected}`);
  }
  return { ok: problems.length === 0, problems };
}

/**
 * `tunedtensor.json` locks to `tunedtensor.lock.json` beside it. Temporary
 * copies that `tt` and the agent write beside a spec
 * (`.tunedtensor.json.tt-local-….json`) share the original spec's lock.
 */
export function specLockPath(specPath: string): string {
  const name = basename(specPath).replace(/^\.(.+)\.tt-(?:local|agent)-[^/\\]*\.json$/, "$1");
  return join(dirname(specPath), `${name.replace(/\.json$/i, "")}.lock.json`);
}

export async function readSpecLock(specPath: string): Promise<SpecLock | undefined> {
  const path = specLockPath(specPath);
  const lock = await readJsonIfPresent<SpecLock>(path);
  if (lock && lock.version !== 1) throw new Error(`${path} has unsupported version ${String(lock.version)}.`);
  return lock;
}

export async function writeSpecLockEntry(specPath: string, key: LockKey, entry: AssetSnapshot): Promise<void> {
  const lock = await readSpecLock(specPath) ?? { version: 1 };
  await writeJsonAtomic(specLockPath(specPath), { ...lock, version: 1, [key]: sortedEntry(entry) });
}

function sortedEntry(entry: AssetSnapshot): AssetSnapshot {
  return {
    repo: entry.repo,
    revision: entry.revision,
    files: Object.fromEntries(Object.entries(entry.files).sort(([a], [b]) => a.localeCompare(b))),
  };
}

const PREFETCH_COMMAND: Record<LockKey, string> = {
  dataset: "tt datasets prefetch",
  base_model: "tt models prefetch",
};

/**
 * Compare what a run is about to use with the spec's lock. A missing entry is
 * recorded (trust on first use, like `terraform init`); a different repo,
 * revision or file digest stops the run. Prefetch commands update the lock.
 */
export async function checkSpecLock(specPath: string, key: LockKey, actual: AssetSnapshot): Promise<"recorded" | "matched"> {
  const locked = (await readSpecLock(specPath))?.[key];
  if (!locked) {
    await writeSpecLockEntry(specPath, key, actual);
    return "recorded";
  }
  const lockFile = basename(specLockPath(specPath));
  const update = `If this change is intended, run \`${PREFETCH_COMMAND[key]}\` with this spec to update ${lockFile}.`;
  const label = key === "dataset" ? "dataset" : "base model";
  if (locked.repo !== actual.repo || locked.revision !== actual.revision) {
    throw new SpecLockMismatchError(
      `The spec's ${label} is ${actual.repo}@${actual.revision}, but ${lockFile} pins ${locked.repo}@${locked.revision}. ${update}`,
    );
  }
  if (key === "base_model") {
    for (const path of Object.keys(locked.files)) {
      if (actual.files[path] === undefined) {
        throw new SpecLockMismatchError(`base model file ${path} is missing from the cached snapshot. ${update}`);
      }
    }
  }
  for (const [path, digest] of Object.entries(actual.files)) {
    if (locked.files[path] === undefined) {
      throw new SpecLockMismatchError(`${label} file ${path} is not in ${lockFile}. ${update}`);
    }
    if (locked.files[path] !== digest) {
      throw new SpecLockMismatchError(
        `${label} file ${path} is ${digest}, but ${lockFile} expects ${locked.files[path]}. The cached file differs from the one this project was locked to. ${update}`,
      );
    }
  }
  return "matched";
}

/** The revision the lock pins for a repo, when it is a valid commit. */
export function lockedRevision(lock: SpecLock | undefined, key: LockKey, repo: string): string | undefined {
  const entry = lock?.[key];
  return entry?.repo === repo && COMMIT_SHA.test(entry.revision) ? entry.revision : undefined;
}

/** Use the lock's revision for a spec dataset that does not pin one itself. */
export function applySpecLock<T extends FineTuneRunRequest>(request: T, lock: SpecLock | undefined): T {
  const source = request.dataset_prebuilt?.huggingface;
  if (!source || source.revision) return request;
  const revision = lockedRevision(lock, "dataset", source.repo);
  if (!revision) return request;
  return { ...request, dataset_prebuilt: { ...request.dataset_prebuilt!, huggingface: { ...source, revision } } };
}

/**
 * Check a run's resolved Hugging Face dataset and base model against the
 * spec's lock, recording entries the lock does not have yet. Runs that load
 * the base model from paths.baseModel or a remote GPU host skip the model.
 */
export async function enforceSpecLock(args: {
  specPath: string;
  request: FineTuneRunRequest;
  config: LocalRunnerConfig;
  reporter?: LocalRunReporter;
}): Promise<void> {
  if (args.config.dryRun) return;
  const dataset = args.request.dataset_source;
  if (dataset) {
    const paths = Object.values(dataset.files).flatMap((file) => file ? [file.path] : []);
    const { files } = await snapshotDigests(dataset.snapshot_path, paths);
    await checkSpecLock(args.specPath, "dataset", { repo: dataset.repo, revision: dataset.revision, files });
  }
  if (args.config.paths.baseModel || args.config.gpu) return;
  const model = resolveTrainingModel(args.request.spec_snapshot.base_model);
  const revision = resolveRequestedBaseModelRevision(model.id, args.request.hyperparameters.base_model_revision);
  if (!revision) return;
  const hubCache = withHuggingFaceCacheEnvironment(process.env, args.config.paths.modelCache).HF_HUB_CACHE!;
  const snapshot = snapshotDirectory(hubCache, "model", model.id, revision);
  if (!await stat(snapshot).then(() => true, () => false)) {
    if ((await readSpecLock(args.specPath))?.base_model) {
      throw new SpecLockMismatchError(
        `Base model ${model.id}@${revision} is not downloaded (${snapshot}). Run \`tt models prefetch\` with this spec first.`,
      );
    }
    // A first run must download and lock the model before training starts.
    const { prefetchBaseModel } = await import("./prefetch.js");
    await prefetchBaseModel({ request: args.request, config: args.config, reporter: args.reporter });
  }
  const { files } = await snapshotDigests(snapshot);
  await checkSpecLock(args.specPath, "base_model", { repo: model.id, revision, files });
}
