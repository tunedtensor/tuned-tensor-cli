import {
  canonicalFoundationPipeline,
  canonicalPipeline as canonicalPortablePipeline,
  parsePipeline,
  pipelineDocumentSchema,
  type PipelineArtifactReference,
  type PipelineDocumentInput,
  type PipelineStep,
  type PipelineTarget,
  type Pipeline as NormalizedPortablePipeline,
} from "@tuned-tensor/pipeline-contract";

// The portable Pipeline v1 contract lives in `@tuned-tensor/pipeline-contract`.
// This module re-exports it and adds CLI-only execution planning (step
// selection and cross-target artifact transfers).
export {
  PIPELINE_DOCUMENT_VERSION,
  PIPELINE_EVALUATORS,
  PIPELINE_MAX_STEPS,
  PIPELINE_MAX_TRAIN_STEPS,
  PIPELINE_STEP_CAPABILITIES,
  PIPELINE_STEP_USES,
  PIPELINE_TARGETS,
  canonicalFoundationPipeline,
  canonicalJson,
  isFoundationPipeline,
  parsePipeline,
  pipelineDocumentSchema,
  pipelineHash,
} from "@tuned-tensor/pipeline-contract";
export type {
  ComparePipelineStepInput,
  EvaluatePipelineStepInput,
  PipelineArtifactKind,
  PipelineArtifactReference,
  PipelineDocumentInput,
  PipelineStep,
  PipelineStepInput,
  PipelineStepUse,
  PipelineTarget,
  TrainPipelineStepInput,
} from "@tuned-tensor/pipeline-contract";

/** A user-authored Pipeline v1 document, before normalization. */
export type Pipeline = PipelineDocumentInput;
/** A normalized Pipeline v1 document: targets resolved, evaluator defaulted. */
export type NormalizedPipeline = NormalizedPortablePipeline;

export interface ResolvedTransfer {
  from: string;
  from_target: PipelineTarget;
  to_target: PipelineTarget;
}

export type ResolvedPipelineStep = PipelineStep & { transfers: ResolvedTransfer[] };

export interface ExecutionPlan {
  version: 1;
  name?: string;
  steps: ResolvedPipelineStep[];
}

function stepReferences(step: PipelineStep): PipelineArtifactReference[] {
  if (step.uses === "evaluate") return step.with.model === "base" ? [] : [step.with.model];
  if (step.uses === "compare") return [step.with.before, step.with.after];
  if (step.uses === "pretrain") return [step.with.tokenizer];
  if (step.uses === "finetune" || step.uses === "rl") return [step.with.model];
  return [];
}

/** Hyperparameters copied from a foundation `tunedtensor.json` into a Pipeline v1 DAG. */
export interface FoundationPipelineHyperparameters {
  vocab_size: number;
  max_chars: number;
  depth: number;
  pretrain_steps: number;
  finetune_steps: number;
  rl_steps: number;
  batch_size: number;
  sequence_length: number;
  nproc_per_node: number;
}

/** Build the local foundation DAG from spec hyperparameters, including optional RL. */
export function pipelineFromFoundationHyperparameters(
  name: string,
  hp: FoundationPipelineHyperparameters,
): Pipeline {
  const lastTrainId = hp.rl_steps > 0 ? "rl" : "sft";
  const steps: Pipeline["steps"] = [
    {
      id: "tokenize",
      uses: "tokenize",
      target: "local",
      with: { vocabSize: hp.vocab_size, maxChars: hp.max_chars },
    },
    {
      id: "pretrain",
      uses: "pretrain",
      target: "local",
      with: {
        tokenizer: { from: "tokenize.tokenizer" },
        depth: hp.depth,
        steps: hp.pretrain_steps,
        batchSize: hp.batch_size,
        sequenceLength: hp.sequence_length,
        nprocPerNode: hp.nproc_per_node,
      },
    },
    {
      id: "bpb",
      uses: "evaluate",
      target: "local",
      with: { model: { from: "pretrain.model" }, evaluator: "bpb" },
    },
    {
      id: "sft",
      uses: "finetune",
      target: "local",
      with: {
        model: { from: "pretrain.model" },
        steps: hp.finetune_steps,
        batchSize: hp.batch_size,
      },
    },
    {
      id: "chat",
      uses: "evaluate",
      target: "local",
      with: { model: { from: "sft.model" }, evaluator: "chat" },
    },
  ];
  if (hp.rl_steps > 0) {
    steps.push(
      {
        id: "rl",
        uses: "rl",
        target: "local",
        with: { model: { from: "sft.model" }, steps: hp.rl_steps },
      },
      {
        id: "chat_rl",
        uses: "evaluate",
        target: "local",
        with: { model: { from: "rl.model" }, evaluator: "chat" },
      },
    );
  }
  steps.push({
    id: "infer",
    uses: "evaluate",
    target: "local",
    with: { model: { from: `${lastTrainId}.model` }, evaluator: "inference" },
  });
  return {
    version: 1,
    name,
    runtime: { engine: "foundation" },
    steps,
  };
}

function producerId(reference: PipelineArtifactReference): string {
  return reference.from.slice(0, reference.from.lastIndexOf("."));
}

/** Validate an ordered, v1 pipeline without reading, writing, or executing anything. */
export function validatePipeline(pipeline: unknown): string[] {
  const structural = pipelineDocumentSchema.safeParse(pipeline);
  if (!structural.success) {
    return structural.error.issues.map((issue) => {
      const path = issue.path.map(String).join(".");
      return path ? `${path}: ${issue.message}` : issue.message;
    });
  }
  try {
    parsePipeline(pipeline);
    return [];
  } catch (error) {
    return [error instanceof Error ? error.message : String(error)];
  }
}

/** The named compatibility recipe retained by both local and cloud executors. */
export function canonicalPipeline(target: PipelineTarget = "local"): NormalizedPipeline {
  return canonicalPortablePipeline(target);
}

/** Resolve a valid pipeline into an ordered plan with explicit cross-target transfers. */
export function createExecutionPlan(pipeline: unknown, selection: { only?: string[]; skip?: string[] } = {}): ExecutionPlan {
  const errors = validatePipeline(pipeline);
  if (errors.length) throw new Error(`Invalid pipeline:\n- ${errors.join("\n- ")}`);
  const normalized = parsePipeline(pipeline);
  const allIds = new Set(normalized.steps.map((step) => step.id));
  for (const id of [...(selection.only ?? []), ...(selection.skip ?? [])]) {
    if (!allIds.has(id)) throw new Error(`Unknown pipeline step: ${id}.`);
  }
  const only = selection.only?.length ? new Set(selection.only) : undefined;
  const skip = new Set(selection.skip ?? []);
  const steps = normalized.steps.filter((step) => (!only || only.has(step.id)) && !skip.has(step.id));
  if (!steps.length) throw new Error("Selection leaves no pipeline steps to run.");
  const retained = new Map(steps.map((step) => [step.id, step]));
  const resolved = steps.map((step): ResolvedPipelineStep => {
    const transfers: ResolvedTransfer[] = [];
    for (const reference of stepReferences(step)) {
      const producer = retained.get(producerId(reference));
      if (!producer) throw new Error(`Step ${step.id} has a dependency on ${producerId(reference)}, which was omitted by --only or --skip.`);
      if (producer.target !== step.target) transfers.push({ from: reference.from, from_target: producer.target, to_target: step.target });
    }
    return { ...step, transfers };
  });
  return { version: 1, ...(normalized.name ? { name: normalized.name } : {}), steps: resolved };
}
