/** Bytes in one gibibyte. */
export const GIB = 1024 ** 3;

export function gib(n: number): number {
  return Math.round(n * GIB);
}

export const FOUNDATION_DEFAULT_DEPTH = 2;
export const FOUNDATION_MAX_DEPTH = 64;
export const FOUNDATION_DEFAULT_VOCAB = 256;
export const FOUNDATION_DEFAULT_SEQUENCE = 64;
export const FOUNDATION_DEFAULT_BATCH = 2;

/**
 * Conservative disk/VRAM bands for certified adapter workloads.
 * Headroom is intentional; these are not generic Hugging Face size charts.
 */
export interface AdapterMemoryProfile {
  id: string;
  /** Hugging Face cache / snapshot size. */
  diskBytes: number;
  /** Weights + KV/overhead for eval or serve. */
  inferenceBytes: number;
  /** LoRA SFT at the model's default seq/batch. */
  trainBytes: number;
  sparkClass: boolean;
}

export const ADAPTER_MEMORY_PROFILES: readonly AdapterMemoryProfile[] = [
  {
    id: "Qwen/Qwen3.5-2B",
    diskBytes: gib(8),
    inferenceBytes: gib(8),
    trainBytes: gib(16),
    sparkClass: false,
  },
  {
    id: "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    diskBytes: gib(80),
    inferenceBytes: gib(80),
    trainBytes: gib(96),
    sparkClass: true,
  },
  {
    id: "meta-models/Muse-Glimmer-30B",
    diskBytes: gib(80),
    inferenceBytes: gib(80),
    trainBytes: gib(96),
    sparkClass: true,
  },
];

export function adapterMemoryProfile(modelId: string): AdapterMemoryProfile | undefined {
  const normalized = modelId.trim().toLowerCase();
  return ADAPTER_MEMORY_PROFILES.find((profile) => profile.id.toLowerCase() === normalized);
}

/** Same width derivation as training/foundation/src/model.py. */
export function foundationWidth(depth: number): number {
  return 32 * depth;
}

/**
 * Approximate parameter count for the bundled FoundationGPT
 * (tied embeddings, width = 32 * depth).
 */
export function foundationParamCount(
  depth: number,
  vocabSize = FOUNDATION_DEFAULT_VOCAB,
  sequenceLength = FOUNDATION_DEFAULT_SEQUENCE,
): number {
  const width = foundationWidth(depth);
  return 12 * depth * width * width + vocabSize * width + sequenceLength * width;
}

/**
 * Conservative training-byte estimate: 16 bytes/param (weights + Adam)
 * plus activation slack at default batch/sequence.
 */
export function foundationTrainBytes(
  depth: number,
  options: {
    vocabSize?: number;
    sequenceLength?: number;
    batchSize?: number;
  } = {},
): number {
  const vocabSize = options.vocabSize ?? FOUNDATION_DEFAULT_VOCAB;
  const sequenceLength = options.sequenceLength ?? FOUNDATION_DEFAULT_SEQUENCE;
  const batchSize = options.batchSize ?? FOUNDATION_DEFAULT_BATCH;
  const params = foundationParamCount(depth, vocabSize, sequenceLength);
  const width = foundationWidth(depth);
  const activationSlack = batchSize * sequenceLength * width * depth * 64;
  return 16 * params + activationSlack;
}

export function foundationInferenceBytes(
  depth: number,
  options: { vocabSize?: number; sequenceLength?: number } = {},
): number {
  const params = foundationParamCount(
    depth,
    options.vocabSize ?? FOUNDATION_DEFAULT_VOCAB,
    options.sequenceLength ?? FOUNDATION_DEFAULT_SEQUENCE,
  );
  return 4 * params + gib(0.25);
}

export function isUnifiedGpuName(name: string | undefined): boolean {
  if (!name) return false;
  return /spark|gb10/i.test(name);
}
