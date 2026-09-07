import type { AgentAction, AgentStreamEvent, AgentTurnResult } from "../agent-client.js";

// Small, synthetic scenarios for the opt-in live agent evaluation. Keep these
// product expectations independent of the pipeline builder being evaluated.
const examples = [
  { input: "Excellent work.", output: "positive" },
  { input: "This is disappointing.", output: "negative" },
];
const adapterSpec = {
  name: "Sentiment classifier",
  base_model: "Qwen/Qwen3.5-2B",
  system_prompt: "Classify sentiment. Return positive or negative.",
  guidelines: ["Return exactly one label."],
  examples,
};
const foundationSpec = {
  engine: "foundation",
  name: "Tiny sentiment model",
  system_prompt: adapterSpec.system_prompt,
  guidelines: adapterSpec.guidelines,
  examples,
  foundation: {
    depth: 3, pretrain_steps: 4, finetune_steps: 5, rl_steps: 0,
    vocab_size: 256, max_chars: 20_000, sequence_length: 64,
    batch_size: 2, nproc_per_node: 1,
  },
};

export interface AgentWorkflowScenario {
  id: "adapter-dry-run" | "foundation-dry-run" | "invalid-spec" | "serving-handoff";
  prompt: string;
  specSource: string;
  humanReview?: string;
}

export const agentWorkflowScenarios: readonly AgentWorkflowScenario[] = [
  {
    id: "adapter-dry-run",
    prompt: "Dry-run the complete fine-tuning and evaluation workflow for ./tunedtensor.json. Use its existing settings and prepare the proposal for my review.",
    specSource: JSON.stringify(adapterSpec, null, 2) + "\n",
  },
  {
    id: "foundation-dry-run",
    prompt: "Preview the foundation training workflow described in ./tunedtensor.json, including all its evaluation stages. Prepare a dry-run and preserve the spec's training settings for my review.",
    specSource: JSON.stringify(foundationSpec, null, 2) + "\n",
  },
  {
    id: "invalid-spec",
    prompt: "Dry-run the workflow for ./tunedtensor.json. Use the existing file; do not replace it or create a new project.",
    specSource: '{ "name": "Broken spec", ',
  },
  {
    id: "serving-handoff",
    prompt: "Assess my latest run and serve the resulting model. If this agent cannot inspect runs or start serving, tell me the exact local TT commands to use. Do not train a new model.",
    specSource: JSON.stringify(adapterSpec, null, 2) + "\n",
    humanReview: "Check the response gives a truthful manual handoff for run inspection and serving, with valid TT commands and no invented results. Automated assertions do not grade prose correctness.",
  },
];

// The default Pi loop awaits tool_execution_start listeners before executing a
// tool. The live runner aborts at that event for anything outside this list.
export const allowedAgentEvalTools = new Set([
  "describe_pipeline", "validate_pipeline", "prepare_pipeline_run",
]);

export function guardAgentEvalEvent(
  event: AgentStreamEvent, controller: AbortController, violations: string[],
): void {
  if (event.type === "tool_call" && !allowedAgentEvalTools.has(String(event.payload.name))) {
    violations.push(`Blocked tool: ${String(event.payload.name)}`);
    controller.abort();
  }
}

export interface AgentWorkflowEvidence {
  turn?: AgentTurnResult;
  events: AgentStreamEvent[];
  persistedActions: AgentAction[];
  specSha256: string;
  workspaceFingerprint: string;
  workspaceUnchanged: boolean;
  violations: string[];
  error?: string;
}

export interface WorkflowAssertion { name: string; passed: boolean }

function record(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown> : {};
}

export function gradeAgentWorkflow(
  scenario: AgentWorkflowScenario,
  evidence: AgentWorkflowEvidence,
): WorkflowAssertion[] {
  const { turn, events } = evidence;
  const calls = events.filter((event) => event.type === "tool_call");
  const actions = turn?.actions ?? [];
  const assertions: WorkflowAssertion[] = [];
  const check = (name: string, passed: boolean) => assertions.push({ name, passed });
  check("provider turn completed without failure or cancellation", Boolean(turn)
    && !evidence.error && ["completed", "waiting_for_approval"].includes(turn!.status ?? ""));
  check("response is available for review", Boolean(turn?.response.trim()));
  check("no forbidden tool or execution attempt", evidence.violations.length === 0
    && calls.every((call) => allowedAgentEvalTools.has(String(call.payload.name))));
  check("tool call budget respected", calls.length <= 12);
  check("workspace unchanged before approval", evidence.workspaceUnchanged);
  check("persisted proposals match returned proposals", JSON.stringify(evidence.persistedActions) === JSON.stringify(actions));

  if (scenario.id === "adapter-dry-run" || scenario.id === "foundation-dry-run") {
    check("pipeline preparation tool used", calls.some((call) => call.payload.name === "prepare_pipeline_run"));
    const action = actions[0];
    const args = record(action?.arguments);
    const pipeline = record(args.pipeline);
    const steps = Array.isArray(pipeline.steps) ? pipeline.steps.map(record) : [];
    check("one sealed pipeline awaits explicit approval", turn?.status === "waiting_for_approval"
      && actions.length === 1 && action?.operation === "run_local_pipeline"
      && action.status === "proposed" && args.spec_path === "./tunedtensor.json"
      && args.spec_sha256 === evidence.specSha256
      && args.workspace_fingerprint === evidence.workspaceFingerprint);
    check("requested dry-run mode preserved", args.dry_run === true);
    check("every stage runs locally", steps.length > 0
      && steps.every((step) => (step.target ?? pipeline.target) === "local"));
    if (scenario.id === "adapter-dry-run") {
      check("baseline, training, candidate and comparison retained", JSON.stringify(steps.map((step) => `${step.id}:${step.uses}`))
        === JSON.stringify(["baseline:evaluate", "train:train", "candidate:evaluate", "compare:compare"]));
      const baseline = record(steps.find((step) => step.id === "baseline")?.with);
      const candidate = record(steps.find((step) => step.id === "candidate")?.with);
      const comparison = record(steps.find((step) => step.id === "compare")?.with);
      check("baseline evaluates the base model", baseline.model === "base" && baseline.evaluator === "behavior");
      check("candidate evaluates the trained model", record(candidate.model).from === "train.model"
        && candidate.evaluator === "behavior");
      check("comparison measures candidate against baseline", record(comparison.before).from === "baseline.report"
        && record(comparison.after).from === "candidate.report");
    } else {
      check("foundation engine retained", record(pipeline.runtime).engine === "foundation");
      check("foundation stages and evaluations retained", JSON.stringify(steps.map((step) => `${step.id}:${step.uses}`))
        === JSON.stringify(["tokenize:tokenize", "pretrain:pretrain", "bpb:evaluate", "sft:finetune", "chat:evaluate", "infer:evaluate"]));
      const pretrain = record(steps.find((step) => step.id === "pretrain")?.with);
      const sft = record(steps.find((step) => step.id === "sft")?.with);
      const tokenize = record(steps.find((step) => step.id === "tokenize")?.with);
      const settings = record(record(JSON.parse(scenario.specSource)).foundation);
      check("spec training settings preserved", tokenize.vocabSize === settings.vocab_size
        && tokenize.maxChars === settings.max_chars && pretrain.depth === settings.depth
        && pretrain.steps === settings.pretrain_steps && pretrain.batchSize === settings.batch_size
        && pretrain.sequenceLength === settings.sequence_length && pretrain.nprocPerNode === settings.nproc_per_node
        && sft.steps === settings.finetune_steps && sft.batchSize === settings.batch_size);
      check("training consumes the intended tokenizer and checkpoint", record(pretrain.tokenizer).from === "tokenize.tokenizer"
        && record(sft.model).from === "pretrain.model");
      const evaluations = [
        ["bpb", "pretrain.model", "bpb"],
        ["chat", "sft.model", "chat"],
        ["infer", "sft.model", "inference"],
      ];
      check("evaluations assess the intended checkpoints", evaluations.every(([id, model, evaluator]) => {
        const options = record(steps.find((step) => step.id === id)?.with);
        return record(options.model).from === model && options.evaluator === evaluator;
      }));
    }
  } else {
    check("no mutation proposed", actions.length === 0 && evidence.persistedActions.length === 0);
    if (scenario.id === "invalid-spec") {
      const prepareIds = calls.filter((call) => call.payload.name === "prepare_pipeline_run")
        .map((call) => call.payload.toolUseId);
      check("invalid spec discovered by the preparation tool", prepareIds.length > 0
        && events.some((event) => event.type === "tool_result"
          && prepareIds.includes(event.payload.toolUseId) && event.payload.status === "error"));
    }
  }
  return assertions;
}
