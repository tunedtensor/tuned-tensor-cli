import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { StreamFn } from "@earendil-works/pi-agent-core";
import { describe, expect, it, vi } from "vitest";
import type { AgentAction } from "../../agent-client.js";
import { LocalAgentStore } from "../../agent-store.js";
import { createLocalAgentClient } from "../../local-agent-client.js";
import { prepareLocalPipelineAction } from "../../local-pipeline-action.js";
import * as hardware from "../../local-runtime/hardware.js";
import {
  agentWorkflowScenarios, gradeAgentWorkflow, guardAgentEvalEvent,
  type AgentWorkflowEvidence, type AgentWorkflowScenario,
} from "../../eval/agent-workflows.js";

function scenario(id: AgentWorkflowScenario["id"]): AgentWorkflowScenario {
  return agentWorkflowScenarios.find((entry) => entry.id === id)!;
}

function preparedEvidence(): AgentWorkflowEvidence {
  const action: AgentAction = {
    id: "action", title: "Dry-run", summary: "Prepared", risk: "medium",
    operation: "run_local_pipeline", status: "proposed",
    arguments: {
      spec_path: "./tunedtensor.json", spec_sha256: "sealed-spec", dry_run: true,
      workspace_fingerprint: "sealed-workspace",
      pipeline: { version: 1, target: "local", steps: [
        { id: "baseline", uses: "evaluate", with: { model: "base", evaluator: "behavior" } },
        { id: "train", uses: "train" },
        { id: "candidate", uses: "evaluate", with: { model: { from: "train.model" }, evaluator: "behavior" } },
        { id: "compare", uses: "compare", with: { before: { from: "baseline.report" }, after: { from: "candidate.report" } } },
      ] },
    },
  };
  return {
    turn: { threadId: "thread", turnId: "turn", status: "waiting_for_approval",
      response: "The dry-run is prepared for review.", actions: [action] },
    events: [{ type: "tool_call", payload: { name: "prepare_pipeline_run", toolUseId: "tool" } }],
    persistedActions: [action], specSha256: "sealed-spec", workspaceFingerprint: "sealed-workspace",
    workspaceUnchanged: true, violations: [],
  };
}

function failedChecks(id: AgentWorkflowScenario["id"], evidence: AgentWorkflowEvidence): string[] {
  return gradeAgentWorkflow(scenario(id), evidence).filter((check) => !check.passed).map((check) => check.name);
}

async function realPreparedEvidence(
  id: "adapter-dry-run" | "foundation-dry-run",
  change?: { step: string; parameters: Record<string, unknown> },
): Promise<AgentWorkflowEvidence> {
  const workspace = await mkdtemp(join(tmpdir(), "tt-live-eval-preparation-"));
  try {
    await writeFile(join(workspace, "tunedtensor.json"), scenario(id).specSource);
    const input = { workspaceRoot: workspace, dryRun: true };
    let prepared = await prepareLocalPipelineAction(input);
    if (change) {
      const step = prepared.pipeline.steps.find((entry) => entry.id === change.step)!;
      if (!("with" in step)) throw new Error(`Step ${change.step} has no parameters`);
      Object.assign(step.with!, change.parameters);
      // These altered pipelines still satisfy the production tool's contract;
      // the eval must reject their wrong workflow semantics independently.
      prepared = await prepareLocalPipelineAction({ ...input, pipeline: prepared.pipeline });
    }
    const evidence = preparedEvidence();
    evidence.specSha256 = prepared.specSha256;
    evidence.workspaceFingerprint = prepared.workspaceFingerprint;
    evidence.turn!.actions[0]!.arguments = {
      pipeline: prepared.pipeline, spec_path: prepared.specPath, spec_sha256: prepared.specSha256,
      workspace_fingerprint: prepared.workspaceFingerprint,
      config_path: prepared.configPath ?? null, config_sha256: prepared.configSha256 ?? null,
      dry_run: prepared.dryRun,
    };
    return evidence;
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
}

describe("live agent workflow grading", () => {
  it("accepts a sealed canonical dry-run proposal without requiring exact prose", () => {
    expect(failedChecks("adapter-dry-run", preparedEvidence())).toEqual([]);
  });

  it.each(["adapter-dry-run", "foundation-dry-run"] as const)(
    "accepts the actual prepared pipeline for %s",
    async (id) => {
      expect(failedChecks(id, await realPreparedEvidence(id))).toEqual([]);
    },
  );

  it.each([
    { step: "candidate", parameters: { model: "base" }, expected: "candidate evaluates the trained model" },
    { step: "compare", parameters: { before: { from: "candidate.report" }, after: { from: "baseline.report" } },
      expected: "comparison measures candidate against baseline" },
  ])("rejects valid adapter pipelines with wrong $step inputs", async ({ step, parameters, expected }) => {
    const evidence = await realPreparedEvidence("adapter-dry-run", { step, parameters });
    expect(failedChecks("adapter-dry-run", evidence)).toContain(expected);
  });

  it.each([
    { step: "bpb", parameters: { evaluator: "chat" } },
    { step: "chat", parameters: { model: { from: "pretrain.model" } } },
    { step: "chat", parameters: { evaluator: "inference" } },
    { step: "infer", parameters: { model: { from: "pretrain.model" } } },
    { step: "infer", parameters: { evaluator: "chat" } },
  ])("rejects valid foundation pipelines with wrong $step evaluation", async (change) => {
    const evidence = await realPreparedEvidence("foundation-dry-run", change);
    expect(failedChecks("foundation-dry-run", evidence)).toContain("evaluations assess the intended checkpoints");
  });

  it.each([
    { step: "tokenize", parameters: { vocabSize: 512 } },
    { step: "tokenize", parameters: { maxChars: 5_000_000 } },
    { step: "pretrain", parameters: { depth: 6 } },
    { step: "pretrain", parameters: { steps: 40 } },
    { step: "pretrain", parameters: { batchSize: 4 } },
    { step: "pretrain", parameters: { sequenceLength: 128 } },
    { step: "pretrain", parameters: { nprocPerNode: 8 } },
    { step: "sft", parameters: { steps: 50 } },
    { step: "sft", parameters: { batchSize: 4 } },
  ])("rejects valid foundation pipelines that change $step settings: $parameters", async (change) => {
    const evidence = await realPreparedEvidence("foundation-dry-run", change);
    expect(failedChecks("foundation-dry-run", evidence)).toContain("spec training settings preserved");
  });

  it("rejects execution mode changes and skipped evaluation stages", () => {
    const evidence = preparedEvidence();
    const args = evidence.turn!.actions[0]!.arguments as { dry_run: boolean; pipeline: { steps: unknown[] } };
    args.dry_run = false;
    args.pipeline.steps.splice(0, 1);
    expect(failedChecks("adapter-dry-run", evidence)).toEqual(expect.arrayContaining([
      "requested dry-run mode preserved", "baseline, training, candidate and comparison retained",
    ]));
  });

  it("rejects stale seals, completed actions and proposals missing from storage", () => {
    const evidence = preparedEvidence();
    evidence.specSha256 = "different-spec";
    evidence.turn!.actions[0]!.status = "completed";
    evidence.persistedActions = [];
    expect(failedChecks("adapter-dry-run", evidence)).toEqual(expect.arrayContaining([
      "one sealed pipeline awaits explicit approval", "persisted proposals match returned proposals",
    ]));
  });

  it("rejects forbidden tools, workspace writes, timeouts and provider failures", () => {
    const evidence = preparedEvidence();
    evidence.events.push({ type: "tool_call", payload: { name: "search_hugging_face" } });
    evidence.workspaceUnchanged = false;
    evidence.violations.push("Scenario deadline exceeded");
    evidence.turn!.status = "failed";
    expect(failedChecks("adapter-dry-run", evidence)).toEqual(expect.arrayContaining([
      "provider turn completed without failure or cancellation",
      "no forbidden tool or execution attempt", "workspace unchanged before approval",
    ]));
  });

  it("requires evidence the invalid spec was actually checked", () => {
    const evidence = preparedEvidence();
    evidence.turn!.status = "completed";
    evidence.turn!.actions = [];
    evidence.persistedActions = [];
    expect(failedChecks("invalid-spec", evidence)).toContain("invalid spec discovered by the preparation tool");
    evidence.events.push({ type: "tool_result", payload: { toolUseId: "tool", status: "error" } });
    expect(failedChecks("invalid-spec", evidence)).toEqual([]);
  });

  it("does not accept an unrelated tool failure as invalid-spec evidence", () => {
    const evidence = preparedEvidence();
    evidence.turn!.status = "completed";
    evidence.turn!.actions = [];
    evidence.persistedActions = [];
    evidence.events.push({ type: "tool_result", payload: { toolUseId: "unrelated", status: "error" } });
    expect(failedChecks("invalid-spec", evidence)).toContain("invalid spec discovered by the preparation tool");
  });

  it("checks foundation stages and the spec's bounded training settings", () => {
    const evidence = preparedEvidence();
    const args = evidence.turn!.actions[0]!.arguments as Record<string, unknown>;
    args.dry_run = true;
    args.pipeline = { version: 1, target: "local", runtime: { engine: "foundation" }, steps: [
      { id: "tokenize", uses: "tokenize", with: { vocabSize: 256, maxChars: 20_000 } },
      { id: "pretrain", uses: "pretrain", with: { tokenizer: { from: "tokenize.tokenizer" }, depth: 3, steps: 4,
        batchSize: 2, sequenceLength: 64, nprocPerNode: 1 } },
      { id: "bpb", uses: "evaluate", with: { model: { from: "pretrain.model" }, evaluator: "bpb" } },
      { id: "sft", uses: "finetune", with: { model: { from: "pretrain.model" }, steps: 5, batchSize: 2 } },
      { id: "chat", uses: "evaluate", with: { model: { from: "sft.model" }, evaluator: "chat" } },
      { id: "infer", uses: "evaluate", with: { model: { from: "sft.model" }, evaluator: "inference" } },
    ] };
    expect(failedChecks("foundation-dry-run", evidence)).toEqual([]);
    ((args.pipeline as { steps: { with?: Record<string, number> }[] }).steps[1]!.with!).depth = 20;
    expect(failedChecks("foundation-dry-run", evidence)).toContain("spec training settings preserved");
  });

  it("flags serving prose for review and never equates it with actual serving", () => {
    const evidence = preparedEvidence();
    evidence.turn!.status = "completed";
    evidence.turn!.actions = [];
    evidence.events = [];
    evidence.persistedActions = [];
    expect(failedChecks("serving-handoff", evidence)).toEqual([]);
    expect(scenario("serving-handoff").humanReview).toMatch(/no invented results/);
  });

  it.each([
    { name: "search_hugging_face", args: { kind: "model", query: "Qwen" } },
    { name: "examine_hardware", args: { full: true } },
  ])("aborts the real Pi loop before $name can dispatch", async ({ name, args }) => {
    const root = await mkdtemp(join(tmpdir(), "tt-live-eval-guard-"));
    vi.stubEnv("TUNED_TENSOR_HOME", join(root, "home"));
    const fetchSpy = vi.fn(async () => { throw new Error("External network must not be called"); });
    vi.stubGlobal("fetch", fetchSpy);
    const hardwareSpy = vi.spyOn(hardware, "assessHardware")
      .mockRejectedValue(new Error("Hardware probe must not be called"));
    const forbidden = vi.fn(async (): Promise<never> => { throw new Error("Unexpected API call"); });
    type Message = Awaited<ReturnType<Awaited<ReturnType<StreamFn>>["result"]>>;
    const message: Message = {
      role: "assistant", api: "openai-completions", provider: "fixture", model: "scripted",
      timestamp: 0, stopReason: "toolUse",
      content: [{ type: "toolCall", id: "blocked-call", name, arguments: args }],
      usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
    };
    const streamSimple = vi.fn((_model: unknown, _context: unknown, options?: { signal?: AbortSignal }) => {
      // Match real provider cancellation. Pi asks the stream boundary once
      // more after the aborted batch; no second provider request is needed.
      if (options?.signal?.aborted) throw new Error("Aborted before provider request");
      if (streamSimple.mock.calls.length > 1) throw new Error("Unexpected un-aborted continuation");
      return {
        async *[Symbol.asyncIterator]() {
          yield { type: "start", partial: message };
          yield { type: "done", reason: "toolUse", message };
        },
        result: async () => message,
      };
    });
    try {
      const client = createLocalAgentClient({
        store: new LocalAgentStore(join(root, "agent")), workspaceRoot: root,
        selection: { provider: "fixture", model: "scripted", thinking: "off" },
        modelRuntime: {
          getProviders: () => [{ id: "fixture" }],
          getModels: () => [{ id: "scripted", provider: "fixture" }],
          getModel: () => ({ id: "scripted", provider: "fixture" }),
          hasConfiguredAuth: () => true, streamSimple,
        },
        toolApi: { get: forbidden, postRead: forbidden, propose: forbidden },
        mutationApi: { get: forbidden, post: forbidden, put: forbidden },
      });
      const thread = await client.createThread();
      const controller = new AbortController();
      const violations: string[] = [];
      const turn = await client.runTurn(thread.id, "Exercise the evaluation guard.",
        (event) => guardAgentEvalEvent(event, controller, violations), controller.signal);
      expect(turn.status).toBe("cancelled");
      expect(violations).toEqual([`Blocked tool: ${name}`]);
      expect(fetchSpy).not.toHaveBeenCalled();
      expect(hardwareSpy).not.toHaveBeenCalled();
      expect(forbidden).not.toHaveBeenCalled();
      expect(streamSimple).toHaveBeenCalledTimes(2);
      expect(streamSimple.mock.calls[1]![2]?.signal?.aborted).toBe(true);
    } finally {
      vi.unstubAllEnvs();
      vi.unstubAllGlobals();
      await rm(root, { recursive: true, force: true });
    }
  });
});
