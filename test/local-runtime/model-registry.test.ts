import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import {
  NEMOTRON_BF16_REVISION,
  assertCertifiedBaseModelConfig,
  assertUsableModelArtifact,
  canonicalizeTrainingModel,
  defaultBaseModelRevision,
  resolveTrainingModel,
  TRAINING_MODELS,
} from "../../src/local-runtime/model-registry.js";

test("registry certifies the native text-only Qwen and Nemotron training paths", () => {
  assert.deepEqual(TRAINING_MODELS.map((model) => model.id), [
    "Qwen/Qwen3.5-2B",
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
  ]);
  assert.equal(canonicalizeTrainingModel(" qwen/QWEN3.5-2b "), "Qwen/Qwen3.5-2B");
  assert.deepEqual(resolveTrainingModel("Qwen/Qwen3.5-2B"), {
    id: "Qwen/Qwen3.5-2B",
    family: "qwen3_5",
    defaultLearningRate: 0.00001,
    defaultPerDeviceBatchSize: 1,
    defaultGradientAccumulationSteps: 8,
    defaultLoraRank: 16,
    defaultLoraAlpha: 32,
    defaultLoraDropout: 0.05,
    defaultMaxSeqLength: 2048,
    loraTargetModules: "all-linear",
    gradientCheckpointing: false,
  });
  assert.equal(
    canonicalizeTrainingModel(" NVIDIA/nvidia-nemotron-3.5-lightning-30b-a3b-bf16 "),
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
  );
  assert.deepEqual(resolveTrainingModel("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"), {
    id: "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    family: "nemotron_h",
    defaultRevision: NEMOTRON_BF16_REVISION,
    defaultLearningRate: 0.00001,
    defaultPerDeviceBatchSize: 1,
    defaultGradientAccumulationSteps: 8,
    defaultLoraRank: 16,
    defaultLoraAlpha: 32,
    defaultLoraDropout: 0.05,
    defaultMaxSeqLength: 1024,
    loraTargetModules: ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj"],
    gradientCheckpointing: true,
  });
});

test("certified config rejects a larger same-family Qwen snapshot", () => {
  const config = {
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
  assert.doesNotThrow(() => assertCertifiedBaseModelConfig(config));
  assert.throws(
    () => assertCertifiedBaseModelConfig({
      ...config,
      text_config: { ...config.text_config, hidden_size: 2560 },
    }),
    /certified Qwen\/Qwen3\.5-2B architecture/,
  );
});

test("certified config accepts only the released Nemotron 3.5 Lightning BF16 architecture", () => {
  const config = {
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
  };
  assert.doesNotThrow(() => assertCertifiedBaseModelConfig(config));
  assert.throws(
    () => assertCertifiedBaseModelConfig({ ...config, n_routed_experts: 64 }),
    /Nemotron-3\.5-Lightning-30B-A3B-BF16 architecture/,
  );
});

test("Nemotron is bound to the reviewed immutable Hugging Face revision", () => {
  assert.equal(
    resolveTrainingModel("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16").defaultRevision,
    NEMOTRON_BF16_REVISION,
  );
  assert.equal(
    defaultBaseModelRevision("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"),
    "ce38b6ab8b252b4b8ee7165b4605e93191cafd73",
  );
  // Qwen remains unpinned for backward compatibility.
  assert.equal(defaultBaseModelRevision("Qwen/Qwen3.5-2B"), undefined);
});

test("certified config rejects a same-family Qwen snapshot paired with a Nemotron request", () => {
  const qwen = {
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
  // Valid Qwen snapshot, but requested model is Nemotron -> rejected.
  assert.throws(
    () => assertCertifiedBaseModelConfig(
      qwen,
      "Local base-model config",
      "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    ),
    /does not match requested base model/,
  );
});

test("certified config rejects a Nemotron snapshot paired with a Qwen request", () => {
  const nemotron = {
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
  };
  assert.doesNotThrow(() => assertCertifiedBaseModelConfig(nemotron));
  // Valid Nemotron snapshot, but requested model is Qwen -> rejected.
  assert.throws(
    () => assertCertifiedBaseModelConfig(
      nemotron,
      "Local base-model config",
      "Qwen/Qwen3.5-2B",
    ),
    /does not match requested base model/,
  );
  // Matching request accepted.
  assert.doesNotThrow(() => assertCertifiedBaseModelConfig(
    nemotron,
    "Local base-model config",
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
  ));
});

test("optimizer state cannot masquerade as a LoRA adapter", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-model-artifact-"));
  try {
    const model = join(root, "model");
    await mkdir(model);
    await writeFile(join(model, "optimizer.pt"), "optimizer state");
    await writeFile(join(model, "adapter_config.json"), "{}");
    await assert.rejects(
      assertUsableModelArtifact(model),
      /contains no non-empty adapter_model weights/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("a LoRA artifact requires both adapter weights and configuration", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-model-artifact-"));
  try {
    const model = join(root, "model");
    await mkdir(model);
    await writeFile(join(model, "adapter_model.safetensors"), "weights");
    await assert.rejects(
      assertUsableModelArtifact(model),
      /adapter weights but no non-empty adapter_config\.json/,
    );

    await writeFile(join(model, "adapter_config.json"), "{}");
    const inspection = await assertUsableModelArtifact(model);
    assert.equal(inspection.adapter_weight_file_count, 1);
    assert.equal(inspection.adapter_weight_bytes, 7);
    assert.equal(inspection.has_adapter_config, true);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
