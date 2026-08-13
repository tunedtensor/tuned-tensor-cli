import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { localRunnerConfigSchema } from "../../src/local-runtime/contracts.js";
import {
  buildLocalBaseModelServerLaunch,
  buildLocalModelServerLaunch,
  serveLocalModel,
  verifyLocalModelServerLaunch,
} from "../../src/local-runtime/model-server.js";
import type { LocalModelRecord } from "../../src/local-runtime/store.js";

const model: LocalModelRecord = {
  id: "local-11111111-1111-4111-8111-111111111111",
  run_id: "11111111-1111-4111-8111-111111111111",
  behavior_spec_id: "22222222-2222-4222-8222-222222222222",
  name: "Qwen adapter",
  provider: "local-uv",
  base_model: "Qwen/Qwen3.5-2B",
  artifact_uri: "file:///tmp/model.tar.gz",
  artifact_dir: "/tmp/run",
  metrics: null,
  created_at: "2026-07-13T00:00:00.000Z",
};

test("serving launches the bundled text-only Qwen adapter with safe model settings", () => {
  const config = localRunnerConfigSchema.parse({
    paths: {
      baseModel: "/tmp/qwen-snapshot",
      modelCache: "/tmp/huggingface",
    },
    evaluation: {
      inference: {
        device: "cuda",
      },
      scoring: { mode: "exact_match" },
    },
  });
  const launch = buildLocalModelServerLaunch({
    model,
    config,
    options: {
      port: 8123,
      systemPrompt: "Be concise.",
      maxTokens: 64,
      maxConcurrentRequests: 2,
      baseModelArtifactUri: "file:///tmp/qwen-snapshot",
      baseModelRevision: "ignored-for-local-snapshot",
    },
  });

  assert.equal(launch.command, "uv");
  assert.ok(launch.commandArgs.includes("--project"));
  assert.ok(launch.commandArgs.some((value) =>
    value.endsWith("training/local-runner/src/serve.py")
  ));
  assert.equal(launch.env.TT_MODEL_ARTIFACT, "/tmp/model.tar.gz");
  assert.equal(launch.env.TT_BASE_MODEL, "Qwen/Qwen3.5-2B");
  assert.equal(launch.env.TT_MODEL_SOURCE, "/tmp/qwen-snapshot");
  assert.equal(launch.env.TT_BASE_MODEL_REVISION, undefined);
  assert.equal(launch.env.TT_MODEL_LOADER, "causal_lm");
  assert.equal(launch.env.TT_TRUST_REMOTE_CODE, "false");
  assert.equal(launch.env.TT_CHAT_TEMPLATE_KWARGS, undefined);
  assert.equal(launch.env.HF_HOME, "/tmp/huggingface");
  assert.equal(launch.env.HF_HUB_CACHE, "/tmp/huggingface/hub");
  assert.equal(launch.env.HF_HUB_OFFLINE, "1");
  assert.equal(launch.env.TRANSFORMERS_OFFLINE, "1");
  assert.ok(launch.env.UV_PROJECT_ENVIRONMENT);
  assert.equal(launch.env.TT_SYSTEM_PROMPT, "Be concise.");
  assert.equal(launch.env.TT_MAX_CONCURRENT_REQUESTS, "2");
  assert.equal(launch.url, "http://127.0.0.1:8123");
});

test("protected-base serving omits the adapter while remaining offline", () => {
  const launch = buildLocalBaseModelServerLaunch({
    baseModel: "Qwen/Qwen3.5-2B",
    config: localRunnerConfigSchema.parse({
      paths: {
        baseModel: "/tmp/qwen-snapshot",
        modelCache: "/tmp/huggingface",
      },
    }),
    options: {
      baseModelRevision: "0123456789abcdef0123456789abcdef01234567",
    },
  });
  assert.equal(launch.artifactPath, undefined);
  assert.equal(launch.env.TT_MODEL_ARTIFACT, undefined);
  assert.equal(launch.env.TT_BASE_MODEL, "Qwen/Qwen3.5-2B");
  assert.equal(launch.env.TT_MODEL_SOURCE, "/tmp/qwen-snapshot");
  assert.equal(launch.env.TT_BASE_MODEL_REVISION, undefined);
  assert.equal(launch.env.HF_HUB_OFFLINE, "1");
  assert.equal(launch.env.TRANSFORMERS_OFFLINE, "1");
  assert.equal(launch.modelName, "base:Qwen/Qwen3.5-2B");
});

test("local Nemotron serving exports canonical identity separately from its snapshot", () => {
  const snapshot = "/tmp/nemotron-snapshot";
  const launch = buildLocalBaseModelServerLaunch({
    baseModel: "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    config: localRunnerConfigSchema.parse({ paths: { baseModel: snapshot } }),
    options: {
      baseModelRevision: "ce38b6ab8b252b4b8ee7165b4605e93191cafd73",
    },
  });
  assert.equal(
    launch.env.TT_BASE_MODEL,
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
  );
  assert.equal(launch.env.TT_MODEL_SOURCE, snapshot);
  assert.equal(
    launch.env.TT_BASE_MODEL_REVISION,
    "ce38b6ab8b252b4b8ee7165b4605e93191cafd73",
  );
});

test("serving rejects unsafe artifacts, base-model mismatches, and invalid network bounds", () => {
  const config = localRunnerConfigSchema.parse({
    paths: { baseModel: "/tmp/qwen-snapshot" },
  });
  assert.throws(
    () => buildLocalModelServerLaunch({
      model: { ...model, artifact_uri: "s3://bucket/model" },
      config,
    }),
    /local file artifact/,
  );
  assert.throws(
    () => buildLocalModelServerLaunch({
      model,
      config,
      options: { baseModelArtifactUri: "file:///tmp/different-snapshot" },
    }),
    /does not match the base model recorded/,
  );
  assert.throws(
    () => buildLocalModelServerLaunch({ model, config, options: { port: 70_000 } }),
    /port must be between/,
  );
  assert.throws(
    () => buildLocalModelServerLaunch({ model, config, options: { maxConcurrentRequests: 9 } }),
    /maxConcurrentRequests must be between/,
  );
  assert.throws(
    () => buildLocalModelServerLaunch({ model, config, options: { host: "0.0.0.0" } }),
    /--allow-remote/,
  );
  assert.throws(
    () => buildLocalBaseModelServerLaunch({
      baseModel: "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
      config: localRunnerConfigSchema.parse({}),
      options: { baseModelRevision: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" },
    }),
    /must use certified revision/,
  );
});

test("serving verifies that a configured local snapshot matches the selected model", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-serve-model-"));
  try {
    const snapshot = join(root, "qwen-snapshot");
    await mkdir(snapshot);
    await Promise.all([
      writeFile(join(snapshot, "config.json"), JSON.stringify({
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
      })),
      writeFile(join(snapshot, "tokenizer_config.json"), "{}"),
      writeFile(join(snapshot, "tokenizer.json"), "{}"),
      writeFile(join(snapshot, "model.safetensors"), "weights"),
    ]);
    const launch = buildLocalBaseModelServerLaunch({
      baseModel: "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
      config: localRunnerConfigSchema.parse({ paths: { baseModel: snapshot } }),
      options: {
        baseModelRevision: "ce38b6ab8b252b4b8ee7165b4605e93191cafd73",
      },
    });
    await assert.rejects(
      verifyLocalModelServerLaunch(launch),
      /does not match requested base model/,
    );
    await assert.rejects(
      serveLocalModel({
        ...launch,
        command: join(root, "must-not-spawn"),
        commandArgs: [],
      }),
      /does not match requested base model/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("an explicitly remote bind requires and forwards only the selected bearer token", () => {
  const previous = process.env.TT_TEST_SERVE_KEY;
  process.env.TT_TEST_SERVE_KEY = "local-test-token";
  try {
    const launch = buildLocalModelServerLaunch({
      model,
      config: localRunnerConfigSchema.parse({}),
      options: {
        host: "0.0.0.0",
        allowRemote: true,
        apiKeyEnv: "TT_TEST_SERVE_KEY",
      },
    });
    assert.equal(launch.env.TT_API_KEY, "local-test-token");
    assert.equal(launch.env.TT_TEST_SERVE_KEY, undefined);
    assert.equal(launch.url, "http://0.0.0.0:8000");
  } finally {
    if (previous === undefined) delete process.env.TT_TEST_SERVE_KEY;
    else process.env.TT_TEST_SERVE_KEY = previous;
  }
});
