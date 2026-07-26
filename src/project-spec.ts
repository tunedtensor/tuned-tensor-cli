import { canonicalizeSpecBaseModel } from "./base-models.js";
import type { LocalSpec } from "./eval/types.js";

export const CLOUD_SPEC_KEYS = [
  "name",
  "description",
  "base_model",
  "system_prompt",
  "guidelines",
  "constraints",
  "examples",
  "eval_cases",
] as const;

export const LOCAL_ONLY_SPEC_KEYS = [
  "hyperparameters",
  "dataset_prebuilt",
] as const;

export const LOCAL_SPEC_KEYS = [
  "id",
  "name",
  "description",
  "system_prompt",
  "guidelines",
  "constraints",
  "base_model",
  "examples",
  ...LOCAL_ONLY_SPEC_KEYS,
] as const;

const CLOUD_SPEC_KEY_SET = new Set<string>(CLOUD_SPEC_KEYS);
const LOCAL_ONLY_SPEC_KEY_SET = new Set<string>(LOCAL_ONLY_SPEC_KEYS);
const LOCAL_SPEC_KEY_SET = new Set<string>(LOCAL_SPEC_KEYS);
const KNOWN_PROJECT_SPEC_KEY_SET = new Set<string>([
  ...CLOUD_SPEC_KEYS,
  ...LOCAL_SPEC_KEYS,
]);

export interface ProjectSpec extends LocalSpec {
  hyperparameters?: Record<string, unknown>;
  dataset_prebuilt?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface ProjectSpecProjection {
  body: Record<string, unknown>;
  droppedKeys: string[];
}

function projectSpec(
  raw: Record<string, unknown>,
  supportedKeys: ReadonlySet<string>,
): ProjectSpecProjection {
  const body: Record<string, unknown> = {};
  const droppedKeys: string[] = [];

  for (const [key, value] of Object.entries(raw)) {
    if (supportedKeys.has(key)) {
      body[key] = value;
    } else {
      droppedKeys.push(key);
    }
  }

  return {
    body: canonicalizeSpecBaseModel(body),
    droppedKeys,
  };
}

/**
 * Project a shared tunedtensor.json onto the hosted API contract.
 *
 * This is intentionally an allow-list: local runner configuration and unknown
 * project metadata must remain in the project file without being sent to the
 * cloud API.
 */
export function projectCloudSpec(
  raw: Record<string, unknown>,
): ProjectSpecProjection {
  return projectSpec(raw, CLOUD_SPEC_KEY_SET);
}

/**
 * Project a shared tunedtensor.json onto the TT Local behavior-spec contract.
 *
 * Cloud-only executable eval cases are omitted because TT Local uses its own
 * held-out base-versus-tuned evaluation contract.
 */
export function projectLocalSpec(
  raw: Record<string, unknown>,
): ProjectSpecProjection {
  return projectSpec(raw, LOCAL_SPEC_KEY_SET);
}

export function hasLocalOnlySpecFields(
  raw: Record<string, unknown>,
): boolean {
  return Object.keys(raw).some((key) => LOCAL_ONLY_SPEC_KEY_SET.has(key));
}

export function unknownProjectSpecKeys(
  raw: Record<string, unknown>,
): string[] {
  return Object.keys(raw).filter(
    (key) => !KNOWN_PROJECT_SPEC_KEY_SET.has(key),
  );
}
