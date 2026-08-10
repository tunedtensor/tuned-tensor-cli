export const PIPELINE_VERSION = 1 as const;
export const PIPELINE_TARGETS = ["local", "cloud"] as const;
export const PIPELINE_USES = ["train", "evaluate", "compare"] as const;

export type PipelineTarget = (typeof PIPELINE_TARGETS)[number];
export type PipelineUse = (typeof PIPELINE_USES)[number];
export type PipelineRef = { from: string };

export interface PipelineStep {
  id: string;
  uses: PipelineUse;
  target?: PipelineTarget;
  with?: Record<string, unknown>;
}

export interface Pipeline {
  version: number;
  name?: string;
  target?: PipelineTarget;
  steps: PipelineStep[];
}

export interface ResolvedTransfer {
  from: string;
  from_target: PipelineTarget;
  to_target: PipelineTarget;
}

export interface ResolvedPipelineStep extends PipelineStep {
  target: PipelineTarget;
  transfers: ResolvedTransfer[];
}

export interface ExecutionPlan {
  version: 1;
  name?: string;
  steps: ResolvedPipelineStep[];
}

const OUTPUTS: Record<PipelineUse, readonly string[]> = {
  train: ["model"],
  evaluate: ["report"],
  compare: ["comparison"],
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function ref(value: unknown): PipelineRef | undefined {
  return isRecord(value) && typeof value.from === "string" ? { from: value.from } : undefined;
}

function parseRef(value: PipelineRef): { stepId: string; output: string } | undefined {
  const match = /^([A-Za-z][A-Za-z0-9_-]*)\.([A-Za-z][A-Za-z0-9_-]*)$/.exec(value.from);
  return match ? { stepId: match[1], output: match[2] } : undefined;
}

function references(step: PipelineStep): Array<{ field: string; value: unknown; expected: string }> {
  if (step.uses === "evaluate") return [{ field: "model", value: step.with?.model, expected: "model" }];
  if (step.uses === "compare") return [
    { field: "before", value: step.with?.before, expected: "report" },
    { field: "after", value: step.with?.after, expected: "report" },
  ];
  return [];
}

function resolvedTarget(step: PipelineStep, pipeline: Pipeline): PipelineTarget | undefined {
  return step.target ?? pipeline.target;
}

/** Validate an ordered, v1 pipeline without reading, writing, or executing anything. */
export function validatePipeline(pipeline: unknown): string[] {
  const errors: string[] = [];
  if (!isRecord(pipeline)) return ["Pipeline must be an object."];
  if (pipeline.version !== PIPELINE_VERSION) errors.push("Pipeline version must be 1.");
  if (pipeline.target !== undefined && !PIPELINE_TARGETS.includes(pipeline.target as PipelineTarget)) {
    errors.push("Pipeline target must be local or cloud.");
  }
  const typedPipeline = pipeline as unknown as Pipeline;
  if (!Array.isArray(pipeline.steps) || pipeline.steps.length === 0) return [...errors, "Pipeline must contain at least one step."];

  const seen = new Map<string, PipelineStep>();
  let trainCount = 0;
  for (const rawStep of pipeline.steps) {
    if (!isRecord(rawStep)) { errors.push("Each pipeline step must be an object."); continue; }
    const step = rawStep as unknown as PipelineStep;
    if (typeof step.id !== "string" || !/^[A-Za-z][A-Za-z0-9_-]*$/.test(step.id)) {
      errors.push("Step id must start with a letter and contain only letters, numbers, _ or -.");
      continue;
    }
    if (seen.has(step.id)) errors.push(`Duplicate step id: ${step.id}.`);
    if (!(PIPELINE_USES as readonly string[]).includes(step.uses)) errors.push(`Step ${step.id} uses an unsupported component: ${String(step.uses)}.`);
    if (step.uses === "train" && ++trainCount > 1) errors.push("Pipeline v1 supports at most one train step because local model artifacts are run-scoped.");
    if (step.uses === "evaluate" && step.with?.evaluator !== undefined && step.with.evaluator !== "behavior") {
      errors.push(`Step ${step.id} evaluator must be behavior.`);
    }
    if (step.target !== undefined && !PIPELINE_TARGETS.includes(step.target)) errors.push(`Step ${step.id} target must be local or cloud.`);
    if (!resolvedTarget(step, typedPipeline)) errors.push(`Step ${step.id} needs a target or a pipeline default target.`);

    for (const input of references(step)) {
      if (step.uses === "evaluate" && input.value === "base") continue;
      const source = ref(input.value);
      if (!source) {
        errors.push(`Step ${step.id} ${input.field} must be ${step.uses === "evaluate" ? "base or " : ""}a { from: \"step.output\" } reference.`);
        continue;
      }
      const parsed = parseRef(source);
      if (!parsed) { errors.push(`Step ${step.id} has an invalid reference: ${source.from}.`); continue; }
      const producer = seen.get(parsed.stepId);
      if (!producer) {
        const existsLater = (pipeline.steps as unknown[]).some((candidate) => isRecord(candidate) && candidate.id === parsed.stepId);
        errors.push(`Step ${step.id} has a ${existsLater ? "forward" : "missing"} reference to ${parsed.stepId}.`);
      } else if (!OUTPUTS[producer.uses]?.includes(parsed.output)) {
        errors.push(`Step ${parsed.stepId} does not produce ${parsed.output} for step ${step.id}; it produces ${OUTPUTS[producer.uses]?.join(", ") ?? "no outputs"}.`);
      } else if (parsed.output !== input.expected) {
        errors.push(`Step ${step.id} ${input.field} requires a ${input.expected} reference, not ${source.from}.`);
      }
    }
    seen.set(step.id, step);
  }
  return errors;
}

export function canonicalPipeline(target: PipelineTarget = "local"): Pipeline {
  return {
    version: PIPELINE_VERSION,
    name: `default-${target}`,
    target,
    steps: [
      { id: "baseline", uses: "evaluate", target, with: { model: "base", evaluator: "behavior" } },
      { id: "train", uses: "train", target },
      { id: "candidate", uses: "evaluate", target, with: { model: { from: "train.model" }, evaluator: "behavior" } },
      { id: "compare", uses: "compare", target, with: { before: { from: "baseline.report" }, after: { from: "candidate.report" } } },
    ],
  };
}

export function createExecutionPlan(pipeline: Pipeline, selection: { only?: string[]; skip?: string[] } = {}): ExecutionPlan {
  const errors = validatePipeline(pipeline);
  if (errors.length) throw new Error(`Invalid pipeline:\n- ${errors.join("\n- ")}`);
  const allIds = new Set(pipeline.steps.map((step) => step.id));
  for (const id of [...(selection.only ?? []), ...(selection.skip ?? [])]) {
    if (!allIds.has(id)) throw new Error(`Unknown pipeline step: ${id}.`);
  }
  const only = selection.only?.length ? new Set(selection.only) : undefined;
  const skip = new Set(selection.skip ?? []);
  const steps = pipeline.steps.filter((step) => (!only || only.has(step.id)) && !skip.has(step.id));
  if (!steps.length) throw new Error("Selection leaves no pipeline steps to run.");
  const retained = new Map(steps.map((step) => [step.id, step]));
  const resolved = steps.map((step) => {
    const target = resolvedTarget(step, pipeline)!;
    const transfers: ResolvedTransfer[] = [];
    for (const input of references(step)) {
      const source = ref(input.value);
      if (!source) continue;
      const parsed = parseRef(source)!;
      const producer = retained.get(parsed.stepId);
      if (!producer) throw new Error(`Step ${step.id} has a dependency on ${parsed.stepId}, which was omitted by --only or --skip.`);
      const fromTarget = resolvedTarget(producer, pipeline)!;
      if (fromTarget !== target) transfers.push({ from: source.from, from_target: fromTarget, to_target: target });
    }
    const normalized = step.uses === "evaluate"
      ? { ...step, with: { ...step.with, evaluator: step.with?.evaluator ?? "behavior" } }
      : step;
    return { ...normalized, target, transfers };
  });
  return { version: PIPELINE_VERSION, ...(pipeline.name ? { name: pipeline.name } : {}), steps: resolved };
}
