export const SUPPORTED_BASE_MODELS = [
  "google/gemma-4-E2B-it",
  "google/gemma-4-26B-A4B-it",
  "Qwen/Qwen3.5-2B",
] as const;

export type SupportedBaseModel = (typeof SUPPORTED_BASE_MODELS)[number];

const BASE_MODEL_ALIASES = new Map<string, SupportedBaseModel>();

for (const model of SUPPORTED_BASE_MODELS) {
  BASE_MODEL_ALIASES.set(model.toLowerCase(), model);
}

for (const [alias, model] of [
  ["google/gemma-4-E2B", "google/gemma-4-E2B-it"],
  ["google/gemma-4-e2b", "google/gemma-4-E2B-it"],
  ["google/gemma-4-e2b-it", "google/gemma-4-E2B-it"],
  ["google/gemma-4-26b-a4b-it", "google/gemma-4-26B-A4B-it"],
  ["google/gemma-4-26B-A4B", "google/gemma-4-26B-A4B-it"],
  ["google/gemma-4-26b-a4b", "google/gemma-4-26B-A4B-it"],
  ["qwen/qwen3.5-2b", "Qwen/Qwen3.5-2B"],
  ["Qwen/Qwen3.5-2B-Base", "Qwen/Qwen3.5-2B"],
  ["qwen/qwen3.5-2b-base", "Qwen/Qwen3.5-2B"],
] as const) {
  BASE_MODEL_ALIASES.set(alias.toLowerCase(), model);
}

export class UnsupportedBaseModelError extends Error {
  constructor(model: unknown) {
    const supported = SUPPORTED_BASE_MODELS.join(", ");
    super(
      `Unsupported base_model "${String(model)}". Supported base models: ${supported}`,
    );
    this.name = "UnsupportedBaseModelError";
  }
}

export function canonicalizeBaseModel(model: unknown): SupportedBaseModel {
  if (typeof model !== "string" || !model.trim()) {
    throw new UnsupportedBaseModelError(model);
  }

  const canonical = BASE_MODEL_ALIASES.get(model.trim().toLowerCase());
  if (!canonical) {
    throw new UnsupportedBaseModelError(model);
  }

  return canonical;
}

export function canonicalizeSpecBaseModel<T extends Record<string, unknown>>(body: T): T {
  if ("base_model" in body && body.base_model !== undefined) {
    return { ...body, base_model: canonicalizeBaseModel(body.base_model) };
  }

  return body;
}
