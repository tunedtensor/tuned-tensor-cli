/**
 * Typed decision checkpoints the bundled `training/decision` runtime can
 * evaluate and fine-tune. A decision model answers one typed question about an
 * input (choice, yes/no or ordinal score) in a single encoder forward pass and
 * returns a probability per option instead of generating text.
 *
 * Each entry is pinned to a reviewed Hugging Face commit; add a model only with
 * its own end-to-end fine-tune and evaluation run.
 */
export interface DecisionModel {
  id: string;
  family: "laya";
  revision: string;
  parameters: string;
  license: string;
  defaultLearningRate: number;
  defaultBatchSize: number;
  defaultEpochs: number;
}

/** Laya 0.3.22 `PINNED_REVISIONS` entry for the English checkpoint. */
export const LAYA_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851";

export const DECISION_MODELS: DecisionModel[] = [
  {
    id: "convaiinnovations/laya",
    family: "laya",
    revision: LAYA_REVISION,
    parameters: "421M (ModernBERT-large encoder + typed decision head)",
    license: "Apache-2.0",
    defaultLearningRate: 0.00002,
    defaultBatchSize: 8,
    defaultEpochs: 3,
  },
];

export const DEFAULT_DECISION_MODEL = DECISION_MODELS[0]!.id;

export function resolveDecisionModel(modelId: string): DecisionModel {
  const normalized = modelId.trim().toLowerCase();
  const model = DECISION_MODELS.find((candidate) => candidate.id.toLowerCase() === normalized);
  if (!model) {
    throw new Error(
      `Unsupported decision model "${modelId}". Supported decision models: ${DECISION_MODELS.map((item) => item.id).join(", ")}`,
    );
  }
  return model;
}

export function isDecisionModel(modelId: string): boolean {
  return DECISION_MODELS.some((candidate) => candidate.id.toLowerCase() === modelId.trim().toLowerCase());
}

export function canonicalizeDecisionModel(modelId: string): string {
  return resolveDecisionModel(modelId).id;
}
