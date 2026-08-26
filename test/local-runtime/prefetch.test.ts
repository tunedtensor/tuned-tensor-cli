import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { access, mkdir, mkdtemp, readFile, realpath, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { test } from "node:test";
import { promisify } from "node:util";
import { fineTuneRunRequestSchema, localRunnerConfigSchema } from "../../src/local-runtime/contracts.js";
import {
  buildModelPrefetchPayload,
  prefetchBaseModel,
  verifyLocalBaseModel,
  verifyModelPrefetchCacheReport,
} from "../../src/local-runtime/prefetch.js";
import { QWEN_3_5_2B_REVISION } from "../../src/local-runtime/model-registry.js";
import {
  resolveHuggingFaceCacheLayout,
  withHuggingFaceCacheEnvironment,
  withOfflineHuggingFaceCacheEnvironment,
} from "../../src/local-runtime/huggingface-cache.js";

const execFileAsync = promisify(execFile);
const pinnedRevision = QWEN_3_5_2B_REVISION;
const downloadedRevision = "fedcba9876543210fedcba9876543210fedcba98";
const nemotronRevision = "ce38b6ab8b252b4b8ee7165b4605e93191cafd73";

const certifiedConfig = {
  architectures: ["Qwen3_5ForConditionalGeneration"],
  model_type: "qwen3_5",
  text_config: {
    model_type: "qwen3_5_text",
    hidden_size: 2048,
    num_hidden_layers: 24,
    num_attention_heads: 8,
    num_key_value_heads: 2,
    intermediate_size: 6144,
    vocab_size: 248320,
  },
};
const certifiedConfigJson = JSON.stringify(certifiedConfig);

const request = fineTuneRunRequestSchema.parse({
  run_id: "11111111-1111-4111-8111-111111111111",
  user_id: "local-user",
  behavior_spec_id: "22222222-2222-4222-8222-222222222222",
  run_number: 1,
  spec_snapshot: {
    name: "Prefetch",
    system_prompt: "Return labels.",
    base_model: "qwen/qwen3.5-2b",
    examples: [{ input: "hello", output: "greeting" }],
  },
});

test("prefetch payload defaults Qwen to the reviewed immutable revision", () => {
  assert.deepEqual(buildModelPrefetchPayload(
    request,
    localRunnerConfigSchema.parse({
      paths: { modelCache: ".cache/huggingface" },
    }),
  ), {
    base_model: "Qwen/Qwen3.5-2B",
    revision: pinnedRevision,
    model_cache: resolve(".cache/huggingface"),
  });
  const explicit = fineTuneRunRequestSchema.parse({
    ...request,
    hyperparameters: {
      ...request.hyperparameters,
      base_model_revision: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    },
  });
  assert.throws(
    () => buildModelPrefetchPayload(explicit, localRunnerConfigSchema.parse({ paths: {} })),
    /must use certified revision/,
  );
});

test("Nemotron prefetch defaults to the reviewed immutable revision", () => {
  const nemotronRequest = fineTuneRunRequestSchema.parse({
    ...request,
    spec_snapshot: {
      name: "Prefetch Nemotron",
      system_prompt: "Return labels.",
      base_model: "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
      examples: [{ input: "hello", output: "greeting" }],
    },
  });
  assert.deepEqual(buildModelPrefetchPayload(
    nemotronRequest,
    localRunnerConfigSchema.parse({ paths: {} }),
  ), {
    base_model: "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    revision: nemotronRevision,
  });
  // A different explicit revision is rejected before any download begins.
  const explicit = fineTuneRunRequestSchema.parse({
    ...nemotronRequest,
    hyperparameters: {
      ...nemotronRequest.hyperparameters,
      base_model_revision: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    },
  });
  assert.throws(
    () => buildModelPrefetchPayload(explicit, localRunnerConfigSchema.parse({ paths: {} })),
    /must use certified revision/,
  );
});

test("local Nemotron prefetch rejects a different explicit revision before filesystem work", async () => {
  const nemotronRequest = fineTuneRunRequestSchema.parse({
    ...request,
    spec_snapshot: {
      ...request.spec_snapshot,
      base_model: "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    },
    hyperparameters: {
      ...request.hyperparameters,
      base_model_revision: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    },
  });
  const root = await mkdtemp(join(tmpdir(), "tt-local-nemotron-revision-"));
  try {
    const config = localRunnerConfigSchema.parse({
      artifactRoot: join(root, "artifacts"),
      paths: { baseModel: join(root, "missing-snapshot") },
    });
    await assert.rejects(prefetchBaseModel({ request: nemotronRequest, config }), /must use certified revision/);
    await assert.rejects(access(config.artifactRoot));
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("a local Nemotron snapshot must match the certified runtime-file manifest", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-nemotron-provenance-"));
  const nemotronConfig = JSON.stringify({
    architectures: ["NemotronHForCausalLM"],
    model_type: "nemotron_h",
    hidden_size: 2688,
    num_hidden_layers: 52,
    num_attention_heads: 32,
    num_key_value_heads: 2,
    intermediate_size: 1856,
    vocab_size: 131072,
    n_routed_experts: 128,
    num_experts_per_tok: 6,
    num_nextn_predict_layers: 1,
    max_position_embeddings: 262144,
  });
  const writeSnapshot = async (path: string) => {
    await mkdir(path, { recursive: true });
    await Promise.all([
      writeFile(join(path, "config.json"), nemotronConfig),
      writeFile(join(path, "tokenizer_config.json"), "{}"),
      writeFile(join(path, "tokenizer.json"), "{}"),
      writeFile(join(path, "model.safetensors"), "weights"),
    ]);
  };
  try {
    const arbitrary = join(root, "copied-model");
    await writeSnapshot(arbitrary);
    await assert.rejects(
      verifyLocalBaseModel(
        arbitrary,
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        join(root, "huggingface"),
      ),
      /certified Hugging Face cache snapshot/,
    );

    const renamed = join(
      root,
      "models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
      "snapshots",
      "ce38b6ab8b252b4b8ee7165b4605e93191cafd73",
    );
    await writeSnapshot(renamed);
    await assert.rejects(
      verifyLocalBaseModel(
        renamed,
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        join(root, "different-huggingface"),
      ),
      /certified Hugging Face cache snapshot/,
    );

    const cacheHome = join(root, "huggingface");
    const canonical = join(
      cacheHome,
      "hub",
      "models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
      "snapshots",
      nemotronRevision,
    );
    await writeSnapshot(canonical);
    await mkdir(join(cacheHome, "hub", "models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", "blobs"), { recursive: true });
    const cacheAlias = join(root, "huggingface-alias");
    await symlink(cacheHome, cacheAlias, "dir");
    await assert.rejects(
      verifyLocalBaseModel(
        join(
          cacheAlias,
          "hub",
          "models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
          "snapshots",
          nemotronRevision,
        ),
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        cacheHome,
      ),
      /Certified snapshot contains unexpected file/,
    );

    const outsideHub = join(root, "outside-hub");
    const symlinkedHubHome = join(root, "symlinked-hub-home");
    const symlinkedHubSnapshot = join(
      outsideHub,
      "models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
      "snapshots",
      nemotronRevision,
    );
    await writeSnapshot(symlinkedHubSnapshot);
    await mkdir(join(outsideHub, "models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", "blobs"), { recursive: true });
    await mkdir(symlinkedHubHome, { recursive: true });
    await symlink(outsideHub, join(symlinkedHubHome, "hub"), "dir");
    await assert.rejects(
      verifyLocalBaseModel(
        symlinkedHubSnapshot,
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        symlinkedHubHome,
      ),
      /hub cache escapes its configured cache root/,
    );

    const repositoryName = "models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16";
    const symlinkedRepositoryHome = join(root, "symlinked-repository-home");
    const outsideRepository = join(root, "outside-repository");
    const symlinkedRepositorySnapshot = join(outsideRepository, "snapshots", nemotronRevision);
    await writeSnapshot(symlinkedRepositorySnapshot);
    await mkdir(join(outsideRepository, "blobs"), { recursive: true });
    await mkdir(join(symlinkedRepositoryHome, "hub"), { recursive: true });
    await symlink(outsideRepository, join(symlinkedRepositoryHome, "hub", repositoryName), "dir");
    await assert.rejects(
      verifyLocalBaseModel(
        symlinkedRepositorySnapshot,
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        symlinkedRepositoryHome,
      ),
      /repository escapes its configured hub cache/,
    );

    const externalBlobs = join(root, "external-blobs");
    const canonicalBlobRoot = join(cacheHome, "hub", repositoryName, "blobs");
    await mkdir(externalBlobs, { recursive: true });
    await rm(canonicalBlobRoot, { recursive: true, force: true });
    await symlink(externalBlobs, canonicalBlobRoot, "dir");
    await assert.rejects(
      verifyLocalBaseModel(
        canonical,
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        cacheHome,
      ),
      /blob store escapes its repository/,
    );
    await rm(canonicalBlobRoot, { force: true });
    await mkdir(canonicalBlobRoot, { recursive: true });

    await mkdir(join(canonical, "nested"), { recursive: true });
    await symlink(arbitrary, join(canonical, "nested", "linked"), "dir");
    await assert.rejects(
      verifyLocalBaseModel(
        canonical,
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        cacheHome,
      ),
      /symbolic link/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("modelCache defines one Hugging Face home and removes legacy overrides", () => {
  const configured = join(tmpdir(), "tt-local-huggingface-home");
  const layout = resolveHuggingFaceCacheLayout(configured);
  assert.deepEqual(layout, {
    hfHome: resolve(configured),
    hubCache: join(resolve(configured), "hub"),
  });
  const env = withHuggingFaceCacheEnvironment({
    HF_HOME: "/old/home",
    HF_HUB_CACHE: "/old/hub",
    HUGGINGFACE_HUB_CACHE: "/old/legacy",
    TRANSFORMERS_CACHE: "/old/transformers",
    PYTORCH_TRANSFORMERS_CACHE: "/old/pytorch",
    PYTORCH_PRETRAINED_BERT_CACHE: "/old/bert",
    KEEP_ME: "yes",
  }, configured);
  assert.equal(env.HF_HOME, layout.hfHome);
  assert.equal(env.HF_HUB_CACHE, layout.hubCache);
  assert.equal(env.HUGGINGFACE_HUB_CACHE, layout.hubCache);
  assert.equal(env.TRANSFORMERS_CACHE, undefined);
  assert.equal(env.PYTORCH_TRANSFORMERS_CACHE, undefined);
  assert.equal(env.PYTORCH_PRETRAINED_BERT_CACHE, undefined);
  assert.equal(env.KEEP_ME, "yes");

  const offline = withOfflineHuggingFaceCacheEnvironment({}, configured);
  assert.equal(offline.HF_HUB_OFFLINE, "1");
  assert.equal(offline.TRANSFORMERS_OFFLINE, "1");
});

test("bundled prefetch verifies the cached Qwen snapshot and blob checksums", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-prefetch-python-"));
  try {
    const fakePackage = join(root, "fake-python", "huggingface_hub");
    await mkdir(fakePackage, { recursive: true });
    await writeFile(join(fakePackage, "__init__.py"), `
import hashlib
import json
import os
from pathlib import Path

class _Constants:
    HF_HOME = os.environ["HF_HOME"]
    HF_HUB_CACHE = os.environ["HF_HUB_CACHE"]

constants = _Constants()

def snapshot_download(**kwargs):
    assert kwargs["repo_id"] == "Qwen/Qwen3.5-2B"
    assert kwargs["revision"] == ${JSON.stringify(pinnedRevision)}
    assert isinstance(kwargs["local_files_only"], bool)
    assert "*.py" not in kwargs["allow_patterns"]
    repository = Path(constants.HF_HUB_CACHE) / "models--Qwen--Qwen3.5-2B"
    snapshot = repository / "snapshots" / ${JSON.stringify(pinnedRevision)}
    blobs = repository / "blobs"
    snapshot.mkdir(parents=True, exist_ok=True)
    blobs.mkdir(parents=True, exist_ok=True)

    def add(name, contents, lfs=False):
        digest = hashlib.sha256(contents).hexdigest() if lfs else hashlib.sha1(
            f"blob {len(contents)}\\0".encode() + contents
        ).hexdigest()
        blob = blobs / digest
        if not blob.exists():
            blob.write_bytes(contents)
        link = snapshot / name
        if not link.exists():
            link.symlink_to(blob)

    add("config.json", json.dumps({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "intermediate_size": 6144,
            "vocab_size": 248320,
        },
    }).encode())
    add("tokenizer_config.json", b"{}")
    add("tokenizer.json", b"{}")
    add("model.safetensors", b"weights", lfs=True)
    return str(snapshot)
`, "utf8");
    const hfHome = join(root, "huggingface");
    const inputPath = join(root, "input.json");
    const outputPath = join(root, "output.json");
    await writeFile(inputPath, JSON.stringify({
      base_model: "Qwen/Qwen3.5-2B",
      revision: pinnedRevision,
      model_cache: hfHome,
      local_files_only: true,
    }), "utf8");

    const command = [
      "run",
      "python",
      resolve("training/adapter/src/prefetch.py"),
      "--input",
      inputPath,
      "--output",
      outputPath,
    ];
    const result = await execFileAsync("uv", command, {
      cwd: root,
      env: {
        ...process.env,
        PYTHONPATH: join(root, "fake-python"),
        HF_HOME: "/wrong/home",
        HF_HUB_CACHE: "/wrong/hub",
        TRANSFORMERS_CACHE: "/wrong/transformers",
      },
    });
    const output = JSON.parse(await readFile(outputPath, "utf8"));
    assert.equal(output.base_model, "Qwen/Qwen3.5-2B");
    assert.equal(output.hf_home, hfHome);
    assert.equal(output.hub_cache, join(hfHome, "hub"));
    assert.equal(output.snapshot_revision, pinnedRevision);
    assert.equal(output.file_count, 4);
    assert.equal(output.verified_blob_count, 4);
    assert.match(result.stdout, /Verifying cached Qwen\/Qwen3\.5-2B/);

    const weight = join(
      hfHome,
      "hub",
      "models--Qwen--Qwen3.5-2B",
      "snapshots",
      pinnedRevision,
      "model.safetensors",
    );
    await writeFile(await realpath(weight), "corrupt", "utf8");
    await assert.rejects(
      execFileAsync("uv", command, {
        cwd: root,
        env: { ...process.env, PYTHONPATH: join(root, "fake-python") },
      }),
      /checksum mismatch/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("bundled prefetch rejects a downloader result from a different requested revision", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-prefetch-revision-mismatch-"));
  try {
    const receivedRevision = join(root, "received-revision.txt");
    const fakePackage = join(root, "fake-python", "huggingface_hub");
    await mkdir(fakePackage, { recursive: true });
    await writeFile(join(fakePackage, "__init__.py"), `
import os
from pathlib import Path
class _Constants:
    HF_HOME = os.environ["HF_HOME"]
    HF_HUB_CACHE = os.environ["HF_HUB_CACHE"]
constants = _Constants()
def snapshot_download(**kwargs):
    Path(${JSON.stringify(receivedRevision)}).write_text(str(kwargs["revision"]))
    snapshot = Path(constants.HF_HUB_CACHE) / "models--Qwen--Qwen3.5-2B" / "snapshots" / ${JSON.stringify(downloadedRevision)}
    snapshot.mkdir(parents=True, exist_ok=True)
    return str(snapshot)
`, "utf8");
    const hfHome = join(root, "huggingface");
    const inputPath = join(root, "input.json");
    const outputPath = join(root, "output.json");
    await writeFile(inputPath, JSON.stringify({
      base_model: "Qwen/Qwen3.5-2B",
      revision: pinnedRevision,
      model_cache: hfHome,
    }), "utf8");
    const command = [
      "run",
      "python",
      resolve("training/adapter/src/prefetch.py"),
      "--input",
      inputPath,
      "--output",
      outputPath,
    ];
    await assert.rejects(
      execFileAsync("uv", command, {
        cwd: root,
        env: { ...process.env, PYTHONPATH: join(root, "fake-python") },
      }),
      (error: unknown) => {
        const stderr = typeof error === "object" && error && "stderr" in error
          ? String(error.stderr)
          : String(error);
        assert.match(
          stderr,
          new RegExp(`Hugging Face returned revision ${downloadedRevision}, not requested revision ${pinnedRevision}`),
        );
        return true;
      },
    );
    assert.equal(await readFile(receivedRevision, "utf8"), pinnedRevision);
    await assert.rejects(access(outputPath));
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("prefetch report rejects downloader-substituted cache metadata", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-prefetch-cache-substitution-"));
  try {
    const modelId = "Qwen/Qwen3.5-2B";
    const configuredHome = join(root, "configured-cache");
    const repository = join(configuredHome, "hub", "models--Qwen--Qwen3.5-2B");
    const snapshot = join(repository, "snapshots", pinnedRevision);
    const attackerHome = join(root, "attacker-cache");
    await Promise.all([
      mkdir(snapshot, { recursive: true }),
      mkdir(join(repository, "blobs"), { recursive: true }),
      mkdir(join(attackerHome, "hub"), { recursive: true }),
    ]);
    await assert.rejects(
      verifyModelPrefetchCacheReport({
        modelId,
        revision: pinnedRevision,
        modelCache: configuredHome,
        output: {
          model_cache: attackerHome,
          hf_home: attackerHome,
          hub_cache: join(attackerHome, "hub"),
          snapshot_path: snapshot,
        },
      }),
      /unexpected model cache/,
    );

    const configuredAlias = join(root, "configured-cache-alias");
    await symlink(configuredHome, configuredAlias, "dir");
    await verifyModelPrefetchCacheReport({
      modelId,
      revision: pinnedRevision,
      modelCache: configuredHome,
      output: {
        model_cache: configuredAlias,
        hf_home: configuredAlias,
        hub_cache: join(configuredAlias, "hub"),
        snapshot_path: join(
          configuredAlias,
          "hub",
          "models--Qwen--Qwen3.5-2B",
          "snapshots",
          pinnedRevision,
        ),
      },
    });

    const externalBlobs = join(root, "qwen-external-blobs");
    await mkdir(externalBlobs, { recursive: true });
    await rm(join(repository, "blobs"), { recursive: true, force: true });
    await symlink(externalBlobs, join(repository, "blobs"), "dir");
    await assert.rejects(
      verifyModelPrefetchCacheReport({
        modelId,
        revision: pinnedRevision,
        modelCache: configuredHome,
        output: {
          model_cache: configuredHome,
          hf_home: configuredHome,
          hub_cache: join(configuredHome, "hub"),
          snapshot_path: snapshot,
        },
      }),
      /blob store escapes its repository/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("a verified local snapshot skips network prefetch and supports local-only checks", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-prefetch-local-"));
  try {
    const baseModel = join(root, "base-model");
    await mkdir(baseModel);
    await Promise.all([
      writeFile(join(baseModel, "config.json"), certifiedConfigJson),
      writeFile(join(baseModel, "tokenizer_config.json"), "{}"),
      writeFile(join(baseModel, "tokenizer.json"), "{}"),
      writeFile(join(baseModel, "model.safetensors"), "weights"),
    ]);
    const config = localRunnerConfigSchema.parse({
      artifactRoot: join(root, "artifacts"),
      paths: { baseModel, modelCache: join(root, "cache") },
    });

    const report = await prefetchBaseModel({ request, config });
    assert.equal(report.status, "skipped");
    assert.equal(report.local_base_model_path, baseModel);
    assert.equal(report.command, undefined);
    assert.match(report.reason ?? "", /no Hugging Face download is needed/);

    const localOnly = await prefetchBaseModel({ request, config, localOnly: true });
    assert.equal(localOnly.status, "completed");
    assert.equal(localOnly.file_count, 4);
    assert.equal(localOnly.size_bytes, Buffer.byteLength(certifiedConfigJson) + 11);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("local snapshot integrity rejects files, optimizer state, invalid config, and missing shards", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-prefetch-invalid-"));
  try {
    await t.test("standalone file", async () => {
      const path = join(root, "model.safetensors");
      await writeFile(path, "weights");
      await assert.rejects(verifyLocalBaseModel(path), /snapshot directory/);
    });

    await t.test("optimizer-only directory", async () => {
      const path = join(root, "optimizer-only");
      await mkdir(path);
      await Promise.all([
        writeFile(join(path, "config.json"), certifiedConfigJson),
        writeFile(join(path, "tokenizer_config.json"), "{}"),
        writeFile(join(path, "tokenizer.json"), "{}"),
        writeFile(join(path, "optimizer.pt"), "state"),
      ]);
      await assert.rejects(verifyLocalBaseModel(path), /no non-empty Transformers model weights/);
    });

    await t.test("invalid config", async () => {
      const path = join(root, "invalid-config");
      await mkdir(path);
      await Promise.all([
        writeFile(join(path, "config.json"), "{"),
        writeFile(join(path, "tokenizer_config.json"), "{}"),
        writeFile(join(path, "tokenizer.json"), "{}"),
        writeFile(join(path, "model.safetensors"), "weights"),
      ]);
      await assert.rejects(verifyLocalBaseModel(path), /invalid or unsupported config\.json/);
    });

    await t.test("missing indexed shard", async () => {
      const path = join(root, "missing-shard");
      await mkdir(path);
      await Promise.all([
        writeFile(join(path, "config.json"), certifiedConfigJson),
        writeFile(join(path, "tokenizer_config.json"), "{}"),
        writeFile(join(path, "tokenizer.json"), "{}"),
        writeFile(join(path, "model-00001-of-00002.safetensors"), "weights"),
        writeFile(join(path, "model.safetensors.index.json"), JSON.stringify({
          weight_map: {
            layer0: "model-00001-of-00002.safetensors",
            layer1: "model-00002-of-00002.safetensors",
          },
        })),
      ]);
      await assert.rejects(verifyLocalBaseModel(path), /missing indexed weight shard/);
    });
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
