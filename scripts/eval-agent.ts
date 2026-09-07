#!/usr/bin/env node
/**
 * Opt-in, paid evaluation of the real TT agent. Run via `node --import tsx`.
 * It prepares proposals only: approvals, training, downloads and HF search are
 * blocked. Temporary projects and threads are deleted after every invocation.
 * createPiModelRuntime uses the normal saved credentials/custom model config;
 * its usual config-header normalization and OAuth refresh behavior still apply.
 */
import { createHash } from "node:crypto";
import { mkdtemp, mkdir, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { createPiModelRuntime, readStoredProviderSecrets } from "../src/agent-model.js";
import { LocalAgentStore } from "../src/agent-store.js";
import { getAgentSelection, getApiKey } from "../src/config.js";
import { createLocalAgentClient } from "../src/local-agent-client.js";
import { fingerprintWorkspace } from "../src/local-spec-workspace.js";
import type { AgentAction, AgentStreamEvent, AgentTurnResult } from "../src/agent-client.js";
import {
  agentWorkflowScenarios, gradeAgentWorkflow, guardAgentEvalEvent,
  type AgentWorkflowScenario,
} from "../src/eval/agent-workflows.js";

const { values } = parseArgs({ options: {
  help: { type: "boolean", short: "h" },
  list: { type: "boolean" },
  scenario: { type: "string" },
  "timeout-seconds": { type: "string", default: "90" },
  "max-output-tokens": { type: "string", default: "2048" },
} });

if (values.help) {
  process.stdout.write(`Usage: node --import tsx scripts/eval-agent.ts [options]

Runs paid model calls using the provider/model/thinking configured in TT.
Uses synthetic temporary projects; never approves or executes proposals.
JSON lines report assertions and redacted evidence. Exit 1 on any failed check.
The serving handoff additionally requires human review of the response.

  --list                       List cases without calling a provider
  --scenario <id>               Run one case (default: all)
  --timeout-seconds <1..300>     Per-case deadline (default: 90)
  --max-output-tokens <256..8192> Per-response token cap (default: 2048)
  -h, --help                   Show help without calling a provider

Each case permits at most 8 model requests and 12 tool calls.
No model quality, CUDA training or actual inference is measured here.
`);
} else if (values.list) {
  process.stdout.write(JSON.stringify(agentWorkflowScenarios.map(({ id, humanReview }) => ({ id, humanReview }))) + "\n");
} else {
  await main().catch(() => {
    // Provider/config errors can contain credentials. Keep initialization
    // errors generic; per-case diagnostics pass through the store's redactor.
    process.stdout.write(JSON.stringify({ type: "error", automated_passed: false, outcome: "failed",
      error: "Could not initialize the live evaluation. Check TT's saved provider/model configuration and authentication, and the supplied options." }) + "\n");
    process.exitCode = 1;
  });
}

function integerOption(value: string, min: number, max: number): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < min || parsed > max) throw new Error("Invalid numeric option");
  return parsed;
}

async function main(): Promise<void> {
  const timeoutMs = integerOption(values["timeout-seconds"]!, 1, 300) * 1000;
  const maxTokens = integerOption(values["max-output-tokens"]!, 256, 8192);
  const scenarios = values.scenario
    ? agentWorkflowScenarios.filter((scenario) => scenario.id === values.scenario)
    : agentWorkflowScenarios;
  if (!scenarios.length) throw new Error("Unknown scenario");
  const selection = getAgentSelection();
  if (!selection) throw new Error("Configure an agent first");
  const runtime = await createPiModelRuntime();
  const secrets = [
    ...readStoredProviderSecrets(), getApiKey(),
    ...Object.entries(process.env)
      .filter(([key, value]) => /API_?KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL/i.test(key) && (value?.length ?? 0) >= 6)
      .map(([, value]) => value),
  ].filter((value): value is string => Boolean(value));
  const root = await mkdtemp(join(tmpdir(), "tt-live-agent-eval-"));
  const redactor = new LocalAgentStore(join(root, "redaction-only"), {
    secretValues: secrets, secretValueProvider: readStoredProviderSecrets,
  });
  const emit = (value: unknown) => process.stdout.write(JSON.stringify(redactor.redact(value)) + "\n");
  let failed = 0;

  try {
    for (const scenario of scenarios) {
      const report = await runScenario(scenario);
      emit(report);
      if (!report.automated_passed) failed += 1;
    }
    emit({ type: "summary", ...selection, total: scenarios.length, failed,
      automated_passed: failed === 0,
      outcome: failed ? "failed" : scenarios.some((scenario) => scenario.humanReview) ? "manual_review_required" : "passed",
      human_review_required: scenarios.some((scenario) => scenario.humanReview),
      limits: { timeout_ms_per_case: timeoutMs, max_output_tokens_per_response: maxTokens,
        max_model_requests_per_case: 8, max_tool_calls_per_case: 12 } });
    if (failed) process.exitCode = 1;
  } finally {
    await rm(root, { recursive: true, force: true });
  }

  async function runScenario(scenario: AgentWorkflowScenario) {
    const started = performance.now();
    const workspace = join(root, scenario.id, "workspace");
    await mkdir(workspace, { recursive: true });
    const specPath = join(workspace, "tunedtensor.json");
    await writeFile(specPath, scenario.specSource, { mode: 0o600 });
    const workspaceFingerprint = await fingerprintWorkspace(workspace);
    const store = new LocalAgentStore(join(root, scenario.id, "agent"), {
      secretValues: secrets, secretValueProvider: readStoredProviderSecrets,
    });
    const controller = new AbortController();
    const events: AgentStreamEvent[] = [];
    const violations: string[] = [];
    let requests = 0;
    let turn: AgentTurnResult | undefined;
    let error: string | undefined;
    let persistedActions: AgentAction[] = [];
    const blocked = async () => {
      violations.push("Attempted an API call or pipeline execution");
      controller.abort();
      throw new Error("Live evaluation never executes work");
    };
    const timeout = setTimeout(() => {
      violations.push("Scenario deadline exceeded");
      controller.abort();
    }, timeoutMs);
    // A provider that ignores cancellation must not leave the evaluation
    // running indefinitely. Emit an explicit failure and clean up before exit.
    const hardTimeout = setTimeout(() => {
      emit({ type: "scenario", scenario: scenario.id, ...selection, automated_passed: false, outcome: "failed",
        error: "Provider did not stop within five seconds of the deadline" });
      rmSync(root, { recursive: true, force: true });
      process.exit(1);
    }, timeoutMs + 5_000);
    try {
      const client = createLocalAgentClient({
        store, workspaceRoot: workspace, selection: selection!,
        modelRuntime: {
          getProviders: runtime.getProviders.bind(runtime),
          getModels: runtime.getModels.bind(runtime),
          getModel: runtime.getModel.bind(runtime),
          hasConfiguredAuth: runtime.hasConfiguredAuth.bind(runtime),
          registerOpenRouterModel: runtime.registerOpenRouterModel?.bind(runtime),
          streamSimple: (...args: Parameters<typeof runtime.streamSimple>) => {
            // Pi may request another response after an aborted tool batch.
            // Stop at this boundary before even opening a provider request.
            if (controller.signal.aborted || args[2]?.signal?.aborted) throw new Error("Evaluation cancelled");
            requests += 1;
            if (requests > 8) {
              violations.push("Model request budget exceeded");
              controller.abort();
              throw new Error("Model request budget exceeded");
            }
            return runtime.streamSimple(args[0], args[1], { ...args[2], maxTokens,
              signal: args[2]?.signal ? AbortSignal.any([args[2].signal, controller.signal]) : controller.signal });
          },
        },
        toolApi: { get: blocked, postRead: blocked, propose: async (action) => action },
        mutationApi: { get: blocked, post: blocked, put: blocked },
        runPipelineCommand: blocked,
      });
      const thread = await client.createThread();
      try {
        turn = await client.runTurn(thread.id, scenario.prompt, (event) => {
          // Abort synchronously: Pi awaits this listener before executing the
          // tool, and checks the aborted signal before dispatch (pinned 0.84.1).
          guardAgentEvalEvent(event, controller, violations);
          if (!["text_delta", "reasoning_delta"].includes(event.type)) events.push(event);
        }, controller.signal);
      } catch (cause) {
        error = cause instanceof Error ? cause.message : String(cause);
      }
      persistedActions = (await store.load(thread.id)).actions;
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
    } finally {
      clearTimeout(timeout);
      clearTimeout(hardTimeout);
    }
    const workspaceUnchanged = JSON.stringify(await readdir(workspace)) === JSON.stringify(["tunedtensor.json"])
      && await readFile(specPath, "utf8") === scenario.specSource;
    const evidence = { turn, events, persistedActions, workspaceUnchanged, workspaceFingerprint, violations, error,
      specSha256: createHash("sha256").update(scenario.specSource).digest("hex") };
    const assertions = gradeAgentWorkflow(scenario, evidence);
    const automatedPassed = assertions.every((assertion) => assertion.passed);
    return { type: "scenario", scenario: scenario.id, ...selection,
      automated_passed: automatedPassed,
      outcome: !automatedPassed ? "failed" : scenario.humanReview ? "manual_review_required" : "passed", assertions,
      duration_ms: Math.round(performance.now() - started), model_requests: requests,
      tool_calls: events.filter((event) => event.type === "tool_call").length,
      human_review: scenario.humanReview ?? null,
      prompt: scenario.prompt, response: turn?.response ?? "", evidence };
  }
}
