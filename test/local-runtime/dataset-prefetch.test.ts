import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import { mkdir, mkdtemp, readFile, realpath, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { test } from "node:test";
import { promisify } from "node:util";
import {
  fineTuneRunRequestSchema,
  localAdapterSpecFileSchema,
  localRunnerConfigSchema,
  type FineTuneRunRequest,
} from "../../src/local-runtime/contracts.js";
import {
  DatasetNotDownloadedError,
  ensureDatasetCached,
  prefetchHuggingFaceDataset,
  resolveCachedHuggingFaceDataset,
  resolveRequestDataset,
} from "../../src/local-runtime/dataset-prefetch.js";
import { resolveLocalRunInputPaths } from "../../src/local-runtime/local-project.js";

const execFileAsync = promisify(execFile);
const revision = "0123456789abcdef0123456789abcdef01234567";
const repo = "org/tweets";
const spec = {
  name: "Sentiment",
  description: "",
  system_prompt: "Classify sentiment.",
  guidelines: [],
  constraints: [],
  examples: [],
  base_model: "Qwen/Qwen3.5-2B",
};

function chatRow(input: string, output: string, system = "Classify sentiment."): string {
  return JSON.stringify({ messages: [
    { role: "system", content: system },
    { role: "user", content: input },
    { role: "assistant", content: output },
  ] });
}

/** Lay files out exactly as huggingface_hub does: snapshot symlinks into content-addressed blobs. */
async function cacheDataset(hfHome: string, files: Record<string, string>, options: { ref?: boolean } = {}) {
  const repository = join(hfHome, "hub", "datasets--org--tweets");
  const snapshot = join(repository, "snapshots", revision);
  await mkdir(join(repository, "blobs"), { recursive: true });
  for (const [name, content] of Object.entries(files)) {
    const blob = join(repository, "blobs", createHash("sha256").update(content).digest("hex"));
    await writeFile(blob, content);
    await mkdir(join(snapshot, name, ".."), { recursive: true });
    await symlink(blob, join(snapshot, name));
  }
  if (options.ref !== false) {
    await mkdir(join(repository, "refs"), { recursive: true });
    await writeFile(join(repository, "refs", "main"), revision);
  }
  return { repository, snapshot };
}

function request(dataset: Record<string, unknown>): FineTuneRunRequest {
  return fineTuneRunRequestSchema.parse({
    run_id: "11111111-1111-4111-8111-111111111111",
    user_id: "local-user",
    behavior_spec_id: "22222222-2222-4222-8222-222222222222",
    run_number: 1,
    spec_snapshot: spec,
    dataset_prebuilt: dataset,
  });
}

function config(root: string, overrides: Record<string, unknown> = {}) {
  return localRunnerConfigSchema.parse({
    artifactRoot: join(root, "artifacts"),
    storeRoot: join(root, "store"),
    paths: { modelCache: join(root, "hf") },
    ...overrides,
  });
}

test("a Hugging Face dataset source requires contained .jsonl repo paths", () => {
  const base = { ...spec, dataset_prebuilt: { huggingface: { repo }, training: "data/train.jsonl", test: "data/test.jsonl" } };
  assert.equal(localAdapterSpecFileSchema.safeParse(base).success, true);
  for (const training of ["/abs/train.jsonl", "../train.jsonl", "data/../../x.jsonl", "C:\\train.jsonl", "file:///x.jsonl", "data//x.jsonl"]) {
    const parsed = localAdapterSpecFileSchema.safeParse({ ...base, dataset_prebuilt: { ...base.dataset_prebuilt, training } });
    assert.equal(parsed.success, false, training);
    assert.match(JSON.stringify(parsed.error?.issues), /file path inside the dataset repo/);
  }
  const parquet = localAdapterSpecFileSchema.safeParse({ ...base, dataset_prebuilt: { ...base.dataset_prebuilt, training: "train.parquet" } });
  assert.match(JSON.stringify(parquet.error?.issues), /Parquet, CSV and other formats are not supported yet/);
  for (const huggingface of [{ repo: "no-slash" }, { repo, revision: "main" }, { repo, extra: true }, { repo, columns: { input: "text" } }]) {
    assert.equal(localAdapterSpecFileSchema.safeParse({ ...base, dataset_prebuilt: { ...base.dataset_prebuilt, huggingface } }).success, false, JSON.stringify(huggingface));
  }
});

test("Hugging Face split paths are not resolved against the spec directory", () => {
  const resolved = resolveLocalRunInputPaths({
    dataset_prebuilt: { huggingface: { repo }, training: "train.jsonl", test: "test.jsonl" },
  }, "/project/tunedtensor.json") as { dataset_prebuilt: Record<string, unknown> };
  assert.equal(resolved.dataset_prebuilt.training, "train.jsonl");
  const local = resolveLocalRunInputPaths({
    dataset_prebuilt: { training: "train.jsonl", test: "test.jsonl" },
  }, "/project/tunedtensor.json") as { dataset_prebuilt: Record<string, unknown> };
  assert.equal(local.dataset_prebuilt.training, "/project/train.jsonl");
});

test("an uncached dataset fails offline with the prefetch command", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-dataset-missing-"));
  try {
    const run = request({ huggingface: { repo }, training: "train.jsonl", test: "test.jsonl" });
    await assert.rejects(resolveRequestDataset(run, config(root)), (error: unknown) => {
      assert.ok(error instanceof DatasetNotDownloadedError);
      assert.match(error.message, /org\/tweets@main is not in the local cache/);
      assert.match(error.message, /tt datasets prefetch tunedtensor\.json/);
      return true;
    });
    // A dry run never downloads.
    await assert.rejects(
      ensureDatasetCached({ request: run, config: config(root, { dryRun: true }) }),
      DatasetNotDownloadedError,
    );
    // A pinned revision that is not cached is also reported, not fetched.
    await cacheDataset(join(root, "hf"), { "train.jsonl": chatRow("a", "positive") });
    const pinned = request({ huggingface: { repo, revision: "f".repeat(40) }, training: "train.jsonl", test: "test.jsonl" });
    await assert.rejects(resolveRequestDataset(pinned, config(root)), /ffffffff.* is not in the local cache/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("a cached chat JSONL dataset resolves to verified snapshot files with provenance", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-dataset-chat-"));
  try {
    const training = `${chatRow("great", "positive")}\n${chatRow("awful", "negative")}\n`;
    const test = `${chatRow("fine", "positive")}\n`;
    const { snapshot } = await cacheDataset(join(root, "hf"), { "data/train.jsonl": training, "data/test.jsonl": test });
    const run = request({ huggingface: { repo }, training: "data/train.jsonl", test: "data/test.jsonl" });
    const resolved = await resolveRequestDataset(run, config(root));
    assert.equal(resolved.dataset_prebuilt?.huggingface, undefined);
    assert.equal(resolved.dataset_prebuilt?.training, join(snapshot, "data", "train.jsonl"));
    assert.deepEqual(resolved.dataset_source, {
      provider: "huggingface",
      repo,
      requested_revision: null,
      revision,
      pinned: false,
      columns: null,
      snapshot_path: snapshot,
      files: {
        training: { path: "data/train.jsonl", size_bytes: Buffer.byteLength(training), sha256: createHash("sha256").update(training).digest("hex") },
        test: { path: "data/test.jsonl", size_bytes: Buffer.byteLength(test), sha256: createHash("sha256").update(test).digest("hex") },
      },
    });
    // Resolution is idempotent: a resolved request is already local.
    assert.deepEqual(await resolveRequestDataset(resolved, config(root)), resolved);
    const pinned = await resolveRequestDataset(
      request({ huggingface: { repo, revision }, training: "data/train.jsonl", test: "data/test.jsonl" }),
      config(root),
    );
    assert.equal(pinned.dataset_source?.pinned, true);
    assert.equal(pinned.dataset_source?.requested_revision, revision);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("a corrupted or escaping cache entry is rejected", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-dataset-corrupt-"));
  try {
    const { snapshot } = await cacheDataset(join(root, "hf"), { "train.jsonl": chatRow("a", "b"), "test.jsonl": chatRow("c", "d") });
    const dataset = { huggingface: { repo }, training: "train.jsonl", test: "test.jsonl", format: "chat_jsonl" as const };
    await writeFile(await realpath(join(snapshot, "train.jsonl")), chatRow("tampered", "b"));
    await assert.rejects(resolveCachedHuggingFaceDataset({ dataset, modelCache: join(root, "hf") }), /failed its checksum/);
    const outside = join(root, "outside.jsonl");
    await writeFile(outside, chatRow("x", "y"));
    await rm(join(snapshot, "test.jsonl"));
    await symlink(outside, join(snapshot, "test.jsonl"));
    await assert.rejects(
      resolveCachedHuggingFaceDataset({ dataset: { ...dataset, training: "test.jsonl" }, modelCache: join(root, "hf") }),
      /must resolve inside the Hugging Face blob store/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("column mapping converts records with recorded, deterministic cleanup", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-dataset-columns-"));
  try {
    const records = (rows: Array<Record<string, unknown>>) => `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`;
    await cacheDataset(join(root, "hf"), {
      "train.jsonl": records([
        { text: "I love it", label: "positive" },
        { text: "", label: "neutral" },
        { text: "shared prompt", label: "negative" },
        { text: "count", label: 3 },
      ]),
      "test.jsonl": records([
        { text: "Shared   PROMPT", label: "negative" },
        { text: "unique", label: "positive" },
        { text: "unique", label: "positive" },
        { text: "no label" },
      ]),
    });
    const run = request({ huggingface: { repo, columns: { input: "text", output: "label" } }, training: "train.jsonl", test: "test.jsonl" });
    const resolved = await resolveRequestDataset(run, config(root));
    assert.deepEqual(resolved.dataset_source?.conversion, {
      training: { records: 4, kept: 2, skipped_missing_fields: 1, dropped_duplicate_inputs: 0, dropped_eval_overlap: 1 },
      test: { records: 4, kept: 2, skipped_missing_fields: 1, dropped_duplicate_inputs: 1, dropped_eval_overlap: 0 },
    });
    const trainingPath = resolved.dataset_prebuilt!.training;
    assert.ok(trainingPath.startsWith(resolve(root, "artifacts", "datasets", "huggingface", "org-tweets", revision)));
    const rows = (await readFile(trainingPath, "utf8")).trim().split("\n").map((line) => JSON.parse(line));
    assert.deepEqual(rows[0].messages, [
      { role: "system", content: "Classify sentiment." },
      { role: "user", content: "I love it" },
      { role: "assistant", content: "positive" },
    ]);
    assert.equal(rows[1].messages[2].content, "3");
    // Same inputs reuse the content-addressed conversion.
    const again = await resolveRequestDataset(run, config(root));
    assert.equal(again.dataset_prebuilt!.training, trainingPath);
    // A different instruction yields a different conversion.
    const changed = await resolveRequestDataset(
      fineTuneRunRequestSchema.parse({ ...run, spec_snapshot: { ...spec, system_prompt: "Different." } }),
      config(root),
    );
    assert.notEqual(changed.dataset_prebuilt!.training, trainingPath);
    // A misspelled column names the available fields.
    await assert.rejects(
      resolveRequestDataset(request({ huggingface: { repo, columns: { input: "txt", output: "label" } }, training: "train.jsonl", test: "test.jsonl" }), config(root)),
      /no "txt" field.*Available fields: text, label/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("bundled dataset prefetch pins the commit, emits progress and verifies files", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-dataset-prefetch-python-"));
  try {
    const fakePackage = join(root, "fake-python", "huggingface_hub");
    await mkdir(fakePackage, { recursive: true });
    const content = `${chatRow("great", "positive")}\n`;
    const digest = createHash("sha256").update(content).digest("hex");
    await writeFile(join(fakePackage, "__init__.py"), `
import os
from pathlib import Path
from types import SimpleNamespace

class _Constants:
    HF_HOME = os.environ["HF_HOME"]
    HF_HUB_CACHE = os.environ["HF_HUB_CACHE"]

constants = _Constants()
CALLS = Path(os.environ["HF_HOME"]) / "calls.txt"

class HfApi:
    def __init__(self, token=None):
        pass
    def repo_info(self, repo_id, repo_type=None, revision=None, files_metadata=False):
        assert repo_type == "dataset" and files_metadata
        return SimpleNamespace(sha=${JSON.stringify(revision)}, siblings=[
            SimpleNamespace(rfilename="data/train.jsonl", size=${Buffer.byteLength(content)}, blob_id="x" * 40, lfs=SimpleNamespace(sha256=${JSON.stringify(digest)})),
            SimpleNamespace(rfilename="README.md", size=10, blob_id="y" * 40, lfs=None),
        ])

def snapshot_download(**kwargs):
    with CALLS.open("a") as calls:
        calls.write(f"{kwargs['revision']} {kwargs['allow_patterns']}\\n")
    if CALLS.read_text().count("\\n") == 1:
        raise ConnectionResetError("wifi dropped")
    repository = Path(constants.HF_HUB_CACHE) / "datasets--org--tweets"
    snapshot = repository / "snapshots" / ${JSON.stringify(revision)} / "data"
    snapshot.mkdir(parents=True, exist_ok=True)
    blob = repository / "blobs" / ${JSON.stringify(digest)}
    blob.write_text(${JSON.stringify(content)})
    link = snapshot / "train.jsonl"
    if not link.exists():
        link.symlink_to(blob)
    return str(snapshot.parent)
`, "utf8");
    await writeFile(join(fakePackage, "utils.py"), `
import fnmatch
def filter_repo_objects(items, allow_patterns=None, ignore_patterns=None, key=None):
    return [item for item in items if not allow_patterns or any(fnmatch.fnmatch(key(item), pattern) for pattern in allow_patterns)]
`, "utf8");
    const hfHome = join(root, "hf");
    await mkdir(hfHome, { recursive: true });
    const inputPath = join(root, "input.json");
    const outputPath = join(root, "output.json");
    await writeFile(inputPath, JSON.stringify({ repo, files: { training: "data/train.jsonl" }, model_cache: hfHome }));
    const result = await execFileAsync("uv", [
      "run", "python", resolve("training/adapter/src/prefetch_dataset.py"), "--input", inputPath, "--output", outputPath,
    ], {
      cwd: root,
      env: { ...process.env, PYTHONPATH: join(root, "fake-python"), TT_HF_DOWNLOAD_MAX_ATTEMPTS: "2" },
      timeout: 60_000,
    });
    // The first attempt failed, the retry resumed against the same pinned commit.
    assert.equal(await readFile(join(hfHome, "calls.txt"), "utf8"), `${revision} ['data/train.jsonl']\n`.repeat(2));
    assert.match(result.stdout, /Downloading org\/tweets interrupted \(ConnectionResetError: wifi dropped\)\. Retrying in 2s \(attempt 2\/2\)/);
    const progress = result.stdout.split("\n").filter((line) => line.startsWith("@@tt-progress ")).map((line) => JSON.parse(line.slice(14)));
    assert.deepEqual(progress.at(-1), {
      label: "dataset_prefetch", phase: "downloaded", completed_bytes: Buffer.byteLength(content),
      total_bytes: Buffer.byteLength(content), files_completed: 1, files_total: 1,
    });
    const output = JSON.parse(await readFile(outputPath, "utf8"));
    assert.equal(output.snapshot_revision, revision);
    assert.equal(output.files.training.sha256, digest);
    // An unpinned request records refs/main so later offline runs find the snapshot.
    assert.equal(await readFile(join(hfHome, "hub", "datasets--org--tweets", "refs", "main"), "utf8"), revision);
    const resolved = await resolveCachedHuggingFaceDataset({
      dataset: { huggingface: { repo }, training: "data/train.jsonl", format: "chat_jsonl", test: "data/train.jsonl" },
      modelCache: hfHome,
    });
    assert.equal(resolved.source.revision, revision);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("prefetch explains when a spec has no Hugging Face dataset", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-dataset-none-"));
  try {
    const run = fineTuneRunRequestSchema.parse({
      run_id: "11111111-1111-4111-8111-111111111111",
      user_id: "local-user",
      behavior_spec_id: "22222222-2222-4222-8222-222222222222",
      run_number: 1,
      spec_snapshot: { ...spec, examples: [{ input: "a", output: "b" }] },
    });
    await assert.rejects(prefetchHuggingFaceDataset({ request: run, config: config(root) }), /no dataset_prebuilt\.huggingface source/);
    assert.equal(await ensureDatasetCached({ request: run, config: config(root) }), null);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("download tuning settings reach the Python downloader; unrelated secrets do not", async () => {
  const { minimalMachineLearningEnvironment } = await import("../../src/local-runtime/huggingface-cache.js");
  const env = minimalMachineLearningEnvironment({
    TT_HF_DOWNLOAD_MAX_ATTEMPTS: "20",
    HF_HUB_DOWNLOAD_TIMEOUT: "120",
    HF_TOKEN: "hf_x",
    AWS_SECRET_ACCESS_KEY: "secret",
  });
  assert.equal(env.TT_HF_DOWNLOAD_MAX_ATTEMPTS, "20");
  assert.equal(env.HF_HUB_DOWNLOAD_TIMEOUT, "120");
  assert.equal(env.HF_TOKEN, "hf_x");
  assert.equal(env.AWS_SECRET_ACCESS_KEY, undefined);
});
