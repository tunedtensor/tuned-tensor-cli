import { canCreateFileSymlinks, linkOrCopySnapshot } from "../../src/__tests__/platform-fixtures.js";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import {
  applySpecLock,
  checkSpecLock,
  enforceSpecLock,
  findAsset,
  listAssets,
  readSpecLock,
  recordAssetDownload,
  snapshotDigests,
  specLockPath,
  SpecLockMismatchError,
  verifyAsset,
} from "../../src/local-runtime/assets.js";
import { fineTuneRunRequestSchema, localRunnerConfigSchema } from "../../src/local-runtime/contracts.js";

const revision = "0123456789abcdef0123456789abcdef01234567";
const otherRevision = "89abcdef0123456789abcdef0123456789abcdef";

const sha256 = (content: string) => createHash("sha256").update(content).digest("hex");
const gitSha1 = (content: string) => createHash("sha1").update(`blob ${Buffer.byteLength(content)}\0${content}`).digest("hex");

/** Lay files out as huggingface_hub does: snapshot symlinks into content-addressed blobs. */
async function cacheRepo(hub: string, kind: "datasets" | "models", repo: string, files: Record<string, string>, rev = revision) {
  const repository = join(hub, `${kind}--${repo.replace("/", "--")}`);
  const snapshot = join(repository, "snapshots", rev);
  await mkdir(join(repository, "blobs"), { recursive: true });
  for (const [name, content] of Object.entries(files)) {
    // Large (LFS) files are named by SHA-256; small git files by git SHA-1.
    const blob = join(repository, "blobs", name.endsWith(".json") ? gitSha1(content) : sha256(content));
    await writeFile(blob, content);
    await mkdir(join(snapshot, name, ".."), { recursive: true });
    await linkOrCopySnapshot(blob, join(snapshot, name));
  }
  return snapshot;
}

async function temporary(): Promise<string> {
  return await mkdtemp(join(tmpdir(), "tt-assets-"));
}

test("snapshot digests hash Hugging Face blobs and plain copies", { skip: !canCreateFileSymlinks }, async () => {
  const root = await temporary();
  try {
    const snapshot = await cacheRepo(join(root, "hub"), "models", "org/m", { "config.json": "{}", "model.safetensors": "weights" });
    await writeFile(join(snapshot, "copied.txt"), "plain");
    // A plain file whose name happens to be a hash is not a content-addressed blob.
    await writeFile(join(snapshot, "a".repeat(40)), "plain");
    const { files, sizeBytes } = await snapshotDigests(snapshot);
    assert.deepEqual(files, {
      "config.json": `git-sha1:${gitSha1("{}")}`,
      "copied.txt": `sha256:${sha256("plain")}`,
      "model.safetensors": `sha256:${sha256("weights")}`,
      ["a".repeat(40)]: `sha256:${sha256("plain")}`,
    });
    assert.equal(sizeBytes, 2 + 5 + 7 + 5);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test("manifests make downloads findable and detect changed bytes", async () => {
  const root = await temporary();
  const env = { ...process.env, TUNED_TENSOR_HOME: join(root, "tt") };
  try {
    const snapshot = await cacheRepo(join(root, "hub"), "datasets", "org/data", { "train.jsonl": "a\n", "test.jsonl": "b\n" });
    const first = await recordAssetDownload({ kind: "dataset", repo: "org/data", revision, snapshotPath: snapshot, paths: ["train.jsonl"], env });
    assert.equal(first.path, join(root, "tt", "assets", "datasets", "org--data", revision, "manifest.json"));
    // A later download of another split from the same revision adds to the manifest.
    await recordAssetDownload({ kind: "dataset", repo: "org/data", revision, snapshotPath: snapshot, paths: ["test.jsonl"], env });
    const asset = await findAsset(`org/data@${revision.slice(0, 7)}`, env);
    assert.deepEqual(Object.keys(asset.files), ["test.jsonl", "train.jsonl"]);
    assert.equal(asset.size_bytes, 4);
    assert.equal((await listAssets(env)).length, 1);
    assert.deepEqual(await verifyAsset(asset), { ok: true, problems: [] });

    await writeFile(join(snapshot, "train.jsonl"), "tampered\n");
    const result = await verifyAsset(asset);
    assert.equal(result.ok, false);
    assert.match(result.problems[0]!, /train\.jsonl: content does not match/);
    await assert.rejects(findAsset("org/missing", env), /No downloaded asset matches/);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test("each spec has its own lock, shared by temporary copies beside it", () => {
  assert.equal(specLockPath("/p/tunedtensor.json"), join("/p", "tunedtensor.lock.json"));
  assert.equal(specLockPath("/p/support.json"), join("/p", "support.lock.json"));
  assert.equal(specLockPath("/p/.tunedtensor.json.tt-local-12-ab.json"), join("/p", "tunedtensor.lock.json"));
  assert.equal(specLockPath("/p/.support.json.tt-agent-ab.json"), join("/p", "support.lock.json"));
});

test("the lock records a missing entry, then refuses a different revision or file", async () => {
  const root = await temporary();
  const spec = join(root, "tunedtensor.json");
  const entry = { repo: "org/data", revision, files: { "train.jsonl": "sha256:aa" } };
  try {
    assert.equal(await checkSpecLock(spec, "dataset", entry), "recorded");
    assert.deepEqual((await readSpecLock(spec))?.dataset, entry);
    assert.equal(await checkSpecLock(spec, "dataset", entry), "matched");
    await assert.rejects(
      checkSpecLock(spec, "dataset", { ...entry, revision: otherRevision }),
      (error: unknown) => error instanceof SpecLockMismatchError && /pins org\/data@0123/.test(error.message) && /tt datasets prefetch/.test(error.message),
    );
    await assert.rejects(
      checkSpecLock(spec, "dataset", { ...entry, files: { "train.jsonl": "sha256:bb" } }),
      /train\.jsonl is sha256:bb, but tunedtensor\.lock\.json expects sha256:aa/,
    );
    await assert.rejects(checkSpecLock(spec, "dataset", { ...entry, files: { "other.jsonl": "sha256:aa" } }), /not in tunedtensor\.lock\.json/);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test("an unpinned spec dataset uses the lock's revision", () => {
  const request = fineTuneRunRequestSchema.parse({
    run_id: "11111111-1111-4111-8111-111111111111",
    user_id: "local-user",
    behavior_spec_id: "22222222-2222-4222-8222-222222222222",
    run_number: 1,
    spec_snapshot: { name: "S", description: "", system_prompt: "s", guidelines: [], constraints: [], examples: [], base_model: "Qwen/Qwen3.5-2B" },
    dataset_prebuilt: { huggingface: { repo: "org/data" }, training: "train.jsonl", test: "test.jsonl" },
  });
  const lock = { version: 1 as const, dataset: { repo: "org/data", revision, files: {} } };
  assert.equal(applySpecLock(request, lock).dataset_prebuilt?.huggingface?.revision, revision);
  assert.equal(applySpecLock(request, { version: 1, dataset: { ...lock.dataset, repo: "org/other" } }), request);
  assert.equal(applySpecLock(request, undefined), request);
});

test("runs check the resolved dataset and cached base model against the lock", async () => {
  const root = await temporary();
  const spec = join(root, "tunedtensor.json");
  const hf = join(root, "hf");
  try {
    const dataSnapshot = await cacheRepo(join(hf, "hub"), "datasets", "org/data", { "train.jsonl": "a\n" });
    const modelRevision = "1111111111111111111111111111111111111111";
    const modelSnapshot = await cacheRepo(join(hf, "hub"), "models", "org/model", { "config.json": "{}", "model.safetensors": "w" }, modelRevision);
    const request = fineTuneRunRequestSchema.parse({
      run_id: "11111111-1111-4111-8111-111111111111",
      user_id: "local-user",
      behavior_spec_id: "22222222-2222-4222-8222-222222222222",
      run_number: 1,
      spec_snapshot: { name: "S", description: "", system_prompt: "s", guidelines: [], constraints: [], examples: [{ input: "a", output: "b" }, { input: "c", output: "d" }], base_model: "org/model" },
      hyperparameters: { base_model_revision: modelRevision },
      dataset_source: {
        provider: "huggingface", repo: "org/data", requested_revision: null, revision, pinned: false, columns: null,
        snapshot_path: dataSnapshot, files: { training: { path: "train.jsonl", size_bytes: 2, sha256: sha256("a\n") } },
      },
    });
    const config = localRunnerConfigSchema.parse({ artifactRoot: join(root, "a"), storeRoot: join(root, "s"), paths: { modelCache: hf } });
    await enforceSpecLock({ specPath: spec, request, config: { ...config, dryRun: true } });
    assert.equal(await readSpecLock(spec), undefined);
    await enforceSpecLock({ specPath: spec, request, config });
    const lock = JSON.parse(await readFile(join(root, "tunedtensor.lock.json"), "utf8"));
    assert.deepEqual(lock.dataset, { repo: "org/data", revision, files: { "train.jsonl": `sha256:${sha256("a\n")}` } });
    assert.equal(lock.base_model.revision, modelRevision);
    assert.deepEqual(Object.keys(lock.base_model.files), ["config.json", "model.safetensors"]);

    // The same run passes; a snapshot that now points at other weights does not.
    await enforceSpecLock({ specPath: spec, request, config });
    // Editing the blob in place must fail even when its content-addressed name is unchanged.
    await writeFile(join(modelSnapshot, "model.safetensors"), "corrupt");
    await assert.rejects(enforceSpecLock({ specPath: spec, request, config }),
      canCreateFileSymlinks ? /model\.safetensors failed its checksum/ : /model\.safetensors is sha256:/);
    await writeFile(join(modelSnapshot, "model.safetensors"), "w");
    await rm(join(modelSnapshot, "config.json"));
    await assert.rejects(enforceSpecLock({ specPath: spec, request, config }), /config\.json is missing/);
    await linkOrCopySnapshot(join(hf, "hub", "models--org--model", "blobs", gitSha1("{}")), join(modelSnapshot, "config.json"));
    const swapped = join(hf, "hub", "models--org--model", "blobs", sha256("other"));
    await writeFile(swapped, "other");
    await rm(join(modelSnapshot, "model.safetensors"));
    await linkOrCopySnapshot(swapped, join(modelSnapshot, "model.safetensors"));
    await assert.rejects(enforceSpecLock({ specPath: spec, request, config }), /model\.safetensors is sha256:/);

    // A locked base model that is not downloaded asks for a prefetch.
    await rm(join(hf, "hub", "models--org--model"), { recursive: true });
    await assert.rejects(enforceSpecLock({ specPath: spec, request, config }), /not downloaded.*tt models prefetch/);
  } finally { await rm(root, { recursive: true, force: true }); }
});
