import { lstat, readdir } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { verifyTarGzipArchive } from "./artifacts.js";

/**
 * Immutable Hugging Face revision reviewed and certified for Nemotron
 * 3.5 Lightning 30B-A3B-BF16 local fine-tuning. Prefetch, training,
 * evaluation, and serving reject any different revision.
 */
export const NEMOTRON_BF16_REVISION =
  "ce38b6ab8b252b4b8ee7165b4605e93191cafd73";

/**
 * How the bundled text-only runtime loads a training model. Causal-LM
 * checkpoints load through AutoModelForCausalLM; multimodal checkpoints whose
 * text tower is not registered in the causal-LM auto mapping load through
 * AutoModelForImageTextToText instead.
 */
export type ModelLoader = "causal_lm" | "image_text_to_text";

export interface TrainingModel {
  id: string;
  family: string;
  /** Immutable Hugging Face revision this training path is bound to, if any. */
  defaultRevision?: string;
  modelLoader: ModelLoader;
  defaultLearningRate: number;
  defaultPerDeviceBatchSize: number;
  defaultGradientAccumulationSteps: number;
  defaultLoraRank: number;
  defaultLoraAlpha: number;
  defaultLoraDropout: number;
  defaultMaxSeqLength: number;
  loraTargetModules: "all-linear" | string[];
  gradientCheckpointing: boolean;
}

export const TRAINING_MODELS: TrainingModel[] = [
  {
    id: "Qwen/Qwen3.5-2B",
    family: "qwen3_5",
    modelLoader: "causal_lm",
    defaultLearningRate: 0.00001,
    defaultPerDeviceBatchSize: 1,
    defaultGradientAccumulationSteps: 8,
    defaultLoraRank: 16,
    defaultLoraAlpha: 32,
    defaultLoraDropout: 0.05,
    defaultMaxSeqLength: 2048,
    loraTargetModules: "all-linear",
    gradientCheckpointing: false,
  },
  {
    id: "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    family: "nemotron_h",
    defaultRevision: NEMOTRON_BF16_REVISION,
    modelLoader: "causal_lm",
    defaultLearningRate: 0.00001,
    defaultPerDeviceBatchSize: 1,
    defaultGradientAccumulationSteps: 8,
    defaultLoraRank: 16,
    defaultLoraAlpha: 32,
    defaultLoraDropout: 0.05,
    defaultMaxSeqLength: 1024,
    // Avoid one LoRA pair for every routed expert matrix. Adapt the shared
    // attention (q/k/v/o) and Mamba input projections instead, keeping the
    // adapter bounded. out_proj/conv1d are rejected by PEFT for Mamba models.
    loraTargetModules: ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj"],
    gradientCheckpointing: true,
  },
  {
    id: "meta-models/Muse-Glimmer-30B",
    family: "muse_glimmer",
    // Muse Glimmer is a vision-language model whose text tower is not exposed
    // through the causal-LM auto mapping, so the bundled runtime loads the full
    // conditional-generation checkpoint (vision tower frozen and unused) via
    // AutoModelForImageTextToText for text-only SFT.
    modelLoader: "image_text_to_text",
    defaultLearningRate: 0.00001,
    defaultPerDeviceBatchSize: 1,
    defaultGradientAccumulationSteps: 8,
    defaultLoraRank: 16,
    defaultLoraAlpha: 32,
    defaultLoraDropout: 0.05,
    defaultMaxSeqLength: 2048,
    loraTargetModules: "all-linear",
    gradientCheckpointing: true,
  },
];

export function resolveTrainingModel(modelId: string): TrainingModel {
  const normalized = modelId.trim().toLowerCase();
  const model = TRAINING_MODELS.find((candidate) =>
    candidate.id.toLowerCase() === normalized
  );
  if (!model) {
    throw new Error(
      `Unsupported base model "${modelId}". Supported models: ${TRAINING_MODELS.map((item) => item.id).join(", ")}`,
    );
  }
  return model;
}

export function canonicalizeTrainingModel(modelId: string): string {
  return resolveTrainingModel(modelId).id;
}

/** Resolve and enforce the immutable revision contract for a training model. */
export function resolveRequestedBaseModelRevision(
  modelId: string,
  requestedRevision?: string,
): string | undefined {
  const model = resolveTrainingModel(modelId);
  if (
    model.defaultRevision
    && requestedRevision
    && requestedRevision !== model.defaultRevision
  ) {
    throw new Error(
      `${model.id} must use certified revision ${model.defaultRevision}; got ${requestedRevision}.`,
    );
  }
  return requestedRevision ?? model.defaultRevision;
}

/** Resolve the immutable revision a training model is bound to, if any. */
export function defaultBaseModelRevision(modelId: string): string | undefined {
  return resolveRequestedBaseModelRevision(modelId);
}

/** Resolve how the bundled text-only runtime must load a training model. */
export function resolveModelLoader(modelId: string): ModelLoader {
  return resolveTrainingModel(modelId).modelLoader;
}

const CERTIFIED_QWEN_TEXT_CONFIG = {
  model_type: "qwen3_5_text",
  hidden_size: 2048,
  num_hidden_layers: 24,
  num_attention_heads: 8,
  num_key_value_heads: 2,
  intermediate_size: 6144,
  vocab_size: 248320,
} as const;

const CERTIFIED_MUSE_GLIMMER_TEXT_CONFIG = {
  model_type: "muse_glimmer_text",
  hidden_size: 6656,
  num_hidden_layers: 52,
  num_attention_heads: 32,
  num_key_value_heads: 2,
  intermediate_size: 19968,
  vocab_size: 202048,
} as const;

const CERTIFIED_NEMOTRON_CONFIG = {
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
} as const;

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function hasArchitecture(config: Record<string, unknown>, architecture: string): boolean {
  const architectures = config.architectures;
  return Array.isArray(architectures) && architectures.includes(architecture);
}

/**
 * Reject a same-family snapshot that is not the certified text tower. Qwen3.5
 * and Muse Glimmer both expose the text tower through a repo-level config with
 * a `text_config` sub-object, so they share this check.
 */
function assertCertifiedTextTowerConfig(
  config: Record<string, unknown>,
  label: string,
  certifiedModelId: string,
  architecture: string,
  certifiedTextConfig: Readonly<Record<string, unknown>>,
): void {
  if (!hasArchitecture(config, architecture)) {
    throw new Error(`${label} is not the certified ${certifiedModelId} architecture.`);
  }
  const textConfig = config.text_config;
  if (!isRecord(textConfig)) {
    throw new Error(`${label} is not the certified ${certifiedModelId} architecture.`);
  }
  for (const [key, expected] of Object.entries(certifiedTextConfig)) {
    if (textConfig[key] !== expected) {
      throw new Error(
        `${label} is not the certified ${certifiedModelId} architecture: `
        + `text_config.${key} must be ${JSON.stringify(expected)}.`,
      );
    }
  }
}

/** Reject a same-family snapshot that is not the certified architecture. */
export function assertCertifiedBaseModelConfig(
  value: unknown,
  label = "base-model config.json",
  expectedModelId?: string,
): void {
  if (!isRecord(value)) {
    throw new Error(`${label} must contain a JSON object.`);
  }
  const config = value;
  const expectedFamily = expectedModelId
    ? resolveTrainingModel(expectedModelId).family
    : undefined;

  if (config.model_type === "qwen3_5") {
    if (expectedFamily && expectedFamily !== "qwen3_5") {
      throw new Error(`${label} does not match requested base model ${expectedModelId}.`);
    }
    assertCertifiedTextTowerConfig(
      config,
      label,
      "Qwen/Qwen3.5-2B",
      "Qwen3_5ForConditionalGeneration",
      CERTIFIED_QWEN_TEXT_CONFIG,
    );
    return;
  }
  if (config.model_type === "muse_glimmer") {
    if (expectedFamily && expectedFamily !== "muse_glimmer") {
      throw new Error(`${label} does not match requested base model ${expectedModelId}.`);
    }
    assertCertifiedTextTowerConfig(
      config,
      label,
      "meta-models/Muse-Glimmer-30B",
      "MuseGlimmerForConditionalGeneration",
      CERTIFIED_MUSE_GLIMMER_TEXT_CONFIG,
    );
    return;
  }
  if (
    config.model_type === "nemotron_h"
    && hasArchitecture(config, "NemotronHForCausalLM")
  ) {
    if (expectedFamily && expectedFamily !== "nemotron_h") {
      throw new Error(`${label} does not match requested base model ${expectedModelId}.`);
    }
    for (const [key, expected] of Object.entries(CERTIFIED_NEMOTRON_CONFIG)) {
      if (config[key] !== expected) {
        throw new Error(
          `${label} is not the certified nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 architecture: `
          + `${key} must be ${JSON.stringify(expected)}.`,
        );
      }
    }
    return;
  }
  throw new Error(
    `${label} is not a certified TT Local base-model architecture.`,
  );
}

export interface ModelArtifactInspection {
  uri: string;
  path: string;
  kind: "file" | "directory";
  adapter_weight_file_count: number;
  adapter_weight_bytes: number;
  has_adapter_config: boolean;
}

export function localModelArtifactPath(uri: string): string {
  if (uri.startsWith("file://")) return fileURLToPath(new URL(uri));
  if (/^[a-z][a-z0-9+.-]*:/i.test(uri)) {
    throw new Error(`Model artifact must be a local path or file URI: ${uri}`);
  }
  return resolve(uri);
}

function adapterWeightName(name: string): boolean {
  const lower = name.toLowerCase();
  return lower === "adapter_model.safetensors" || lower === "adapter_model.bin";
}

async function inspectAdapterDirectory(path: string): Promise<{
  count: number;
  bytes: number;
  hasConfig: boolean;
}> {
  let count = 0;
  let bytes = 0;
  let hasConfig = false;
  for (const entry of await readdir(path, { withFileTypes: true })) {
    const child = resolve(path, entry.name);
    if (entry.isSymbolicLink()) {
      throw new Error(`Model artifact must not contain symbolic links: ${child}`);
    }
    if (entry.isDirectory()) {
      const nested = await inspectAdapterDirectory(child);
      count += nested.count;
      bytes += nested.bytes;
      hasConfig ||= nested.hasConfig;
      continue;
    }
    if (!entry.isFile()) continue;
    const metadata = await lstat(child);
    if (adapterWeightName(entry.name) && metadata.size > 0) {
      count += 1;
      bytes += metadata.size;
    }
    if (entry.name.toLowerCase() === "adapter_config.json" && metadata.size > 0) {
      hasConfig = true;
    }
  }
  return { count, bytes, hasConfig };
}

export async function inspectModelArtifact(uri: string): Promise<ModelArtifactInspection> {
  const path = localModelArtifactPath(uri);
  const metadata = await lstat(path).catch(() => null);
  if (!metadata) throw new Error(`Model artifact does not exist: ${path}`);
  if (metadata.isSymbolicLink()) {
    throw new Error(`Model artifact must not be a symbolic link: ${path}`);
  }
  if (metadata.isDirectory()) {
    const adapter = await inspectAdapterDirectory(path);
    return {
      uri,
      path,
      kind: "directory",
      adapter_weight_file_count: adapter.count,
      adapter_weight_bytes: adapter.bytes,
      has_adapter_config: adapter.hasConfig,
    };
  }
  if (!metadata.isFile() || !path.toLowerCase().endsWith(".tar.gz")) {
    throw new Error(
      `Model artifact must be a PEFT adapter directory or .tar.gz archive: ${path}`,
    );
  }
  const archive = await verifyTarGzipArchive(path);
  return {
    uri,
    path,
    kind: "file",
    adapter_weight_file_count: archive.adapter_weight_entries,
    adapter_weight_bytes: archive.adapter_weight_bytes,
    has_adapter_config: archive.adapter_config_entries > 0,
  };
}

export async function assertUsableModelArtifact(
  uri: string,
): Promise<ModelArtifactInspection> {
  const inspection = await inspectModelArtifact(uri);
  if (
    inspection.adapter_weight_file_count === 0
    || inspection.adapter_weight_bytes === 0
  ) {
    throw new Error(
      `PEFT model artifact ${inspection.path} contains no non-empty adapter_model weights.`,
    );
  }
  if (!inspection.has_adapter_config) {
    throw new Error(
      `PEFT model artifact ${inspection.path} contains adapter weights but no non-empty adapter_config.json.`,
    );
  }
  return inspection;
}
