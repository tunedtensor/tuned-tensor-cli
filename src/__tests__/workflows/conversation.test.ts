import { execFile } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import type { StreamFn } from "@earendil-works/pi-agent-core";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TunedTensorAgentSession } from "../../agent.js";
import { LocalAgentStore } from "../../agent-store.js";
import { createLocalAgentClient } from "../../local-agent-client.js";
import type { LocalPipelineCommandRunner } from "../../local-pipeline-action.js";

const execFileAsync = promisify(execFile);
type Context = Parameters<StreamFn>[1];
type Message = Awaited<ReturnType<Awaited<ReturnType<StreamFn>>["result"]>>;
type Reply = string | { tool: string; args: Record<string, unknown> } | { error: string };

const spec = {
  name: "Feedback labels",
  base_model: "Qwen/Qwen3.5-2B",
  system_prompt: "Classify feedback as positive or negative.",
  guidelines: ["Return one lowercase label."],
  examples: [
    { input: "I love it", output: "positive" },
    { input: "It broke", output: "negative" },
  ],
};

let root: string;
beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "tt-conversation-workflow-"));
  vi.stubEnv("TUNED_TENSOR_HOME", join(root, "home"));
});
afterEach(() => {
  vi.unstubAllEnvs();
  rmSync(root, { recursive: true, force: true });
});

// Only the provider stream is scripted. Pi still validates arguments, executes
// the real tools, feeds their results back to the model, and enforces its budget.
function conversation(replies: Reply[] | ((context: Context, call: number) => Reply), runCommand?: LocalPipelineCommandRunner) {
  const contexts: Context[] = [];
  const streamSimple = vi.fn((_model: unknown, context: Context) => {
    const call = contexts.length;
    contexts.push({ ...context, messages: structuredClone(context.messages) });
    const reply = typeof replies === "function" ? replies(context, call) : replies[call];
    if (reply === undefined) throw new Error(`Unexpected model call ${call + 1}`);
    const message: Message = {
      role: "assistant",
      api: "openai-completions",
      provider: "fixture",
      model: "scripted",
      timestamp: Date.now(),
      usage: {
        input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
      content: typeof reply === "string"
        ? [{ type: "text", text: reply }]
        : "tool" in reply
          ? [{ type: "toolCall", id: `call-${call}`, name: reply.tool, arguments: reply.args }]
          : [],
      stopReason: typeof reply === "string" ? "stop" : "tool" in reply ? "toolUse" : "error",
      ...(typeof reply !== "string" && "error" in reply ? { errorMessage: reply.error } : {}),
    };
    return {
      async *[Symbol.asyncIterator]() {
        yield { type: "start", partial: message };
        if (typeof reply === "string") {
          yield { type: "text_delta", contentIndex: 0, delta: reply, partial: message };
        }
        yield message.stopReason === "error"
          ? { type: "error", reason: "error", error: message }
          : { type: "done", reason: message.stopReason, message };
      },
      result: async () => message,
    };
  });
  const forbidden = vi.fn(async (): Promise<never> => { throw new Error("Unexpected cloud API call"); });
  const store = new LocalAgentStore(join(root, "agent"));
  const client = createLocalAgentClient({
    store,
    workspaceRoot: root,
    selection: { provider: "fixture", model: "scripted", thinking: "off" },
    modelRuntime: {
      getProviders: () => [{ id: "fixture" }],
      getModels: () => [{ id: "scripted", provider: "fixture" }],
      getModel: () => ({ id: "scripted", provider: "fixture" }),
      hasConfiguredAuth: () => true,
      streamSimple,
    },
    toolApi: { get: forbidden, postRead: forbidden, propose: async (action) => action },
    mutationApi: { get: forbidden, post: forbidden, put: forbidden },
    runPipelineCommand: runCommand,
  });
  const output: string[] = [];
  const errors: string[] = [];
  const newSession = () => new TunedTensorAgentSession({
    client,
    io: { write: (text) => output.push(text), writeError: (text) => errors.push(text), clear: () => {} },
  });
  return { client, store, session: newSession(), newSession, output, errors, contexts, streamSimple, forbidden };
}

describe("conversation workflow contracts (scripted provider, real agent and tools)", () => {
  it("creates a reviewed project, resumes the conversation, and approves the same plan through the real CLI", async () => {
    const plans: Array<{ dry_run: boolean; steps: Array<{ id: string; uses: string }> }> = [];
    const runCommand = vi.fn<LocalPipelineCommandRunner>(async (args, options) => {
      const { stdout } = await execFileAsync(process.execPath, [
        "--import", import.meta.resolve("tsx"), "--input-type=module", "--eval",
        `import { createProgram } from ${JSON.stringify(new URL("../../cli.ts", import.meta.url).href)};
         await createProgram("workflow-test").parseAsync(process.argv.slice(1), { from: "user" });`,
        "--", "--json", ...args,
      ], { cwd: options.cwd, env: process.env, signal: options.signal, timeout: 15_000 });
      plans.push(JSON.parse(stdout));
      return { exitCode: 0, signal: null };
    });
    const c = conversation([
      { tool: "prepare_create_local_spec", args: { directory: "feedback", spec } },
      "The project is ready for review. Use /approve to create it.",
      { tool: "prepare_pipeline_run", args: { spec_path: "feedback/tunedtensor.json" } },
      "The dry-run plan is ready for /approve.",
    ], runCommand);

    await c.session.handleLine("Create a feedback classifier project with positive and negative labels.");
    expect(c.errors).toEqual([]);
    expect(existsSync(join(root, "feedback"))).toBe(false);
    expect(c.session.snapshot().pendingActions).toHaveLength(1);
    await c.session.handleLine("/approve");
    expect(JSON.parse(readFileSync(join(root, "feedback/tunedtensor.json"), "utf8"))).toMatchObject(spec);

    // A fresh UI session reads durable conversation state, including tool results.
    const resumed = c.newSession();
    const threadId = c.session.snapshot().thread!.id;
    await resumed.handleLine(`/resume ${threadId}`);
    await resumed.handleLine("Preview fine-tuning and compare it with the base model; do not train yet.");
    expect(runCommand).not.toHaveBeenCalled();
    expect(resumed.snapshot().pendingActions[0]).toMatchObject({
      operation: "run_local_pipeline", arguments: { dry_run: true, spec_path: "./feedback/tunedtensor.json" },
    });
    expect(c.contexts[2].messages.some((message) => message.role === "toolResult")).toBe(true);
    await resumed.handleLine("/approve");
    expect(runCommand).toHaveBeenCalledTimes(1);
    expect(plans).toEqual([expect.objectContaining({
      dry_run: true,
      steps: [
        expect.objectContaining({ id: "baseline", uses: "evaluate" }),
        expect.objectContaining({ id: "train", uses: "train" }),
        expect.objectContaining({ id: "candidate", uses: "evaluate" }),
        expect.objectContaining({ id: "compare", uses: "compare" }),
      ],
    })]);
    expect((await c.store.load(threadId)).actions.map((action) => action.status)).toEqual(["completed", "completed"]);
    expect(resumed.snapshot().pendingActions).toEqual([]);
    expect(c.errors).toEqual([]);
    await resumed.handleLine("/approve");
    expect(runCommand).toHaveBeenCalledTimes(1);
    expect(c.streamSimple).toHaveBeenCalledTimes(4); // Approvals never call the model.
    expect(c.errors.join("")).toContain("There is no pending action");
    expect(c.forbidden).not.toHaveBeenCalled();
  }, 20_000);

  it.each(["reject", "spec changed", "config changed"])("does not execute a prepared run after %s", async (reason) => {
    writeFileSync(join(root, "tunedtensor.json"), JSON.stringify(spec));
    const runCommand = vi.fn<LocalPipelineCommandRunner>();
    const c = conversation([
      { tool: "prepare_pipeline_run", args: {} },
      "Ready for review.",
    ], runCommand);
    await c.session.handleLine("Fine-tune the current project.");
    expect(c.session.snapshot().pendingActions).toHaveLength(1);
    if (reason === "spec changed") {
      writeFileSync(join(root, "tunedtensor.json"), JSON.stringify({ ...spec, system_prompt: "Changed task" }));
    } else if (reason === "config changed") {
      writeFileSync(join(root, "local-runner.json"), JSON.stringify({ dryRun: true }));
    }
    await c.session.handleLine(reason === "reject" ? "/reject" : "/approve");
    expect(runCommand).not.toHaveBeenCalled();
    const state = await c.store.load(c.session.snapshot().thread!.id);
    expect(state.actions[0].status).toBe(reason === "reject" ? "rejected" : "failed");
    expect(c.session.snapshot().pendingActions).toEqual([]);
    if (reason !== "reject") expect(c.errors.join("")).toMatch(/changed.*prepare it again/);
  });

  it("refuses a persisted proposal changed to request real training", async () => {
    writeFileSync(join(root, "tunedtensor.json"), JSON.stringify(spec));
    const runCommand = vi.fn<LocalPipelineCommandRunner>();
    const c = conversation([
      { tool: "prepare_pipeline_run", args: {} }, "The dry-run is ready for review.",
    ], runCommand);
    await c.session.handleLine("Preview fine-tuning this project.");
    const state = await c.store.load(c.session.snapshot().thread!.id);
    expect(state.actions[0]).toMatchObject({ arguments: { dry_run: true } });
    (state.actions[0].arguments as Record<string, unknown>).dry_run = false;
    await c.store.save(state);
    await c.session.handleLine("/approve");
    expect(runCommand).not.toHaveBeenCalled();
    expect(c.errors.join("")).toMatch(/dry-run only|explicit direct/);
    expect((await c.store.load(state.thread.id)).actions[0].status).toBe("failed");
    expect(c.session.snapshot().pendingActions).toEqual([]);
  });

  it("records a dry-run execution failure without claiming completion or retrying the action", async () => {
    writeFileSync(join(root, "tunedtensor.json"), JSON.stringify(spec));
    const runCommand = vi.fn<LocalPipelineCommandRunner>(async () => ({ exitCode: 1, signal: null }));
    const c = conversation([
      { tool: "prepare_pipeline_run", args: {} }, "Ready for review.",
    ], runCommand);
    await c.session.handleLine("Fine-tune this project.");
    await c.session.handleLine("/approve");
    expect((await c.store.load(c.session.snapshot().thread!.id)).actions[0].status).toBe("failed");
    expect(c.errors.join("")).toContain("Approved pipeline dry-run exited with code 1");
    expect(c.session.snapshot().pendingActions).toEqual([]);
    await c.session.handleLine("/approve");
    expect(runCommand).toHaveBeenCalledTimes(1);
    expect(c.streamSimple).toHaveBeenCalledTimes(2);
  });

  it("surfaces provider failure as a failed turn with no executable action", async () => {
    const c = conversation([{ error: "Provider unavailable" }]);
    const thread = await c.client.createThread();
    const errors: unknown[] = [];
    const result = await c.client.runTurn(thread.id, "Fine-tune this project.", (event) => {
      if (event.type === "error") errors.push(event.payload.message);
    });
    expect(errors).toEqual(["Provider unavailable"]);
    expect(result).toMatchObject({ status: "failed", actions: [] });
    expect((await c.store.load(thread.id)).actions).toEqual([]);
    expect(c.streamSimple).toHaveBeenCalledTimes(1);
  });

  it("stops a model that repeatedly calls tools at the production turn budget", async () => {
    const c = conversation(() => ({ tool: "describe_pipeline", args: { engine: "adapter" } }));
    const thread = await c.client.createThread();
    const statuses: unknown[] = [];
    await c.client.runTurn(thread.id, "Explain the adapter workflow.", (event) => {
      if (event.type === "tool_result") statuses.push(event.payload.status);
    });
    expect(statuses.filter((status) => status === "success")).toHaveLength(12);
    expect(statuses.at(-1)).toBe("error");
    expect(c.streamSimple).toHaveBeenCalledTimes(13);
    expect((await c.store.load(thread.id)).actions).toEqual([]);
  });
});
