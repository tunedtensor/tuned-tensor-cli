import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { LocalAgentStore } from "../agent-store.js";
import {
  createLocalAgentClient,
  type LocalPiAgent,
  type LocalPiAgentOptions,
} from "../local-agent-client.js";
import type { AgentToolApi } from "../agent-tools.js";
import { prepareLocalSpecProject } from "../local-spec-workspace.js";
import { prepareLocalPipelineAction } from "../local-pipeline-action.js";
import { createCloudActionContext } from "../agent-approval.js";
import type { CloudActionContext } from "../agent-client.js";

const TT_SECRET = "tt_never_send_this_to_pi";
const CLOUD_CONTEXT = createCloudActionContext("https://account-a.example", TT_SECRET);
let root: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "tt-local-agent-"));
});
afterEach(() => rmSync(root, { recursive: true, force: true }));

describe("local agent conversation client", () => {
  it.each(["create_spec", "update_spec"] as const)("binds persisted %s approvals to the original account and API origin", async (operation) => {
    const mutationApi = {
      get: vi.fn(async (path: string) => path === "/version"
        ? { data: { capabilities: ["local_agent_spec_mutation_guards_v1"] } }
        : { data: { updated_at: "2026-09-06T00:00:00.000Z" } }),
      post: vi.fn(async () => ({ data: { id: "created-spec" } })),
      put: vi.fn(async () => ({ data: { id: "updated-spec" } })),
    };
    const clientFor = (cloudContext: CloudActionContext | undefined) => createLocalAgentClient({
      store: new LocalAgentStore(root),
      workspaceRoot: root,
      selection: { provider: "openai", model: "gpt-5.2", thinking: "off" },
      modelRuntime: {
        getProviders: () => [{ id: "openai" }],
        getModels: () => [{ id: "gpt-5.2", provider: "openai", reasoning: true }],
        getModel: () => ({ id: "gpt-5.2", provider: "openai", reasoning: true }),
        hasConfiguredAuth: () => true,
      },
      cloudEnabled: Boolean(cloudContext),
      cloudContext,
      toolApi: { get: mutationApi.get, postRead: vi.fn(), propose: async (action) => action },
      mutationApi,
      createAgent: (options) => ({
        state: { messages: [] },
        subscribe: () => () => {},
        abort: () => {},
        prompt: async () => {
          const tool = options.tools.find((candidate) => candidate.name === `prepare_${operation}`)!;
          await tool.execute("prepare-cloud-spec", operation === "create_spec"
            ? { spec: { name: "Support", examples: [{ input: "Hi", output: "Hello" }] } }
            : { spec_id: "11111111-1111-4111-8111-111111111111", changes: { name: "Support" } });
        },
      }),
    });
    const original = clientFor(CLOUD_CONTEXT);
    const thread = await original.createThread();
    const turn = await original.runTurn(thread.id, "Prepare the cloud spec", () => {});
    const action = turn.actions[0]!;
    expect(action.cloud_context).toEqual(CLOUD_CONTEXT);
    expect(action.preview).toEqual({ api_origin: CLOUD_CONTEXT.origin });
    expect(readFileSync(join(root, "threads", `${thread.id}.json`), "utf8")).not.toContain(TT_SECRET);
    mutationApi.get.mockClear();

    for (const changed of [
      createCloudActionContext("https://account-b.example", TT_SECRET),
      createCloudActionContext(CLOUD_CONTEXT.origin, "tt_different_account_token"),
      undefined,
    ]) {
      await expect(clientFor(changed).approveAction(action.id, () => {})).rejects.toThrow(/current TT account and API origin/);
      expect(mutationApi.get).not.toHaveBeenCalled();
      expect(mutationApi.post).not.toHaveBeenCalled();
      expect(mutationApi.put).not.toHaveBeenCalled();
      expect((await new LocalAgentStore(root).load(thread.id)).actions[0]?.status).toBe("proposed");
    }

    // Legacy/unbound proposals must not gain the current account implicitly.
    const legacyStore = new LocalAgentStore(root);
    const legacyState = await legacyStore.load(thread.id);
    delete legacyState.actions[0]!.cloud_context;
    await legacyStore.save(legacyState);
    await expect(clientFor(CLOUD_CONTEXT).approveAction(action.id, () => {})).rejects.toThrow(/current TT account and API origin/);
    expect(mutationApi.get).not.toHaveBeenCalled();
    legacyState.actions[0]!.cloud_context = CLOUD_CONTEXT;
    await legacyStore.save(legacyState);

    const result = await clientFor(createCloudActionContext(`${CLOUD_CONTEXT.origin}/`, TT_SECRET))
      .approveAction(action.id, () => {});
    expect(result.status).toBe("completed");
    expect(operation === "create_spec" ? mutationApi.post : mutationApi.put).toHaveBeenCalledTimes(1);
    expect((await new LocalAgentStore(root).load(thread.id)).actions[0]?.status).toBe("completed");
  });

  it("maps agent streaming events to the existing UI seam and resumes locally", async () => {
    const created: LocalPiAgentOptions[] = [];
    const prompts: string[] = [];
    const createAgent = vi.fn((options: LocalPiAgentOptions): LocalPiAgent => {
      created.push(options);
      const listeners: Array<(event: any) => void | Promise<void>> = [];
      const state = { messages: [...options.messages] };
      return {
        state,
        subscribe(listener) { listeners.push(listener); return () => {}; },
        async prompt(prompt) {
          prompts.push(prompt);
          state.messages.push({ role: "user", content: prompt });
          if (prompt === "Fail") {
            for (const listener of listeners) await listener({
              type: "turn_end",
              message: { role: "assistant", errorMessage: "provider unavailable" },
              toolResults: [],
            });
            return;
          }
          for (const listener of listeners) await listener({
            type: "message_update",
            assistantMessageEvent: { type: "thinking_delta", delta: "Locally thinking" },
          });
          for (const listener of listeners) await listener({
            type: "message_update",
            assistantMessageEvent: { type: "text_delta", delta: "Local answer" },
          });
          for (const listener of listeners) await listener({
            type: "tool_execution_start", toolCallId: "t1", toolName: "validate_pipeline", args: {},
          });
          for (const listener of listeners) await listener({
            type: "tool_execution_end", toolCallId: "t1", toolName: "validate_pipeline",
            result: { details: {} }, isError: false,
          });
          state.messages.push({ role: "assistant", content: [{ type: "text", text: "Local answer" }] });
        },
        abort: vi.fn(),
      };
    });
    const toolApi: AgentToolApi = {
      get: vi.fn(), postRead: vi.fn(), propose: vi.fn(async (action) => action),
    };
    const client = createLocalAgentClient({
      store: new LocalAgentStore(root, { secretValues: [TT_SECRET] }),
      workspaceRoot: root,
      selection: { provider: "openai", model: "gpt-5.2", thinking: "high" },
      modelRuntime: {
        getProviders: () => [{ id: "openai" }],
        getModels: () => [{ id: "gpt-5.2", provider: "openai", reasoning: true }],
        getModel: () => ({ id: "gpt-5.2", provider: "openai", reasoning: true }),
        hasConfiguredAuth: () => true,
      },
      createAgent,
      toolApi,
      mutationApi: { get: vi.fn(), post: vi.fn(), put: vi.fn() },
    });

    const thread = await client.createThread();
    const events: string[] = [];
    const turn = await client.runTurn(thread.id, "Inspect specs", (event) => events.push(event.type));
    expect(events).toEqual([
      "turn_started", "reasoning_delta", "text_delta", "tool_call", "tool_result", "final",
    ]);
    expect(turn.response).toBe("Local answer");
    expect((await client.listThreads())[0]?.id).toBe(thread.id);
    expect((await client.getThread(thread.id)).thread.title).toBe("Inspect specs");
    expect(created[0]?.messages).toEqual([]);
    expect(created[0]?.tools.map((candidate) => candidate.name)).toEqual([
      "examine_hardware",
      "describe_pipeline",
      "search_hugging_face",
      "inspect_training_source",
      "validate_pipeline",
      "prepare_create_local_spec",
      "prepare_pipeline_run",
    ]);
    expect(created[0]?.systemPrompt).toMatch(/sealed local pipeline dry-run/i);
    expect(created[0]?.systemPrompt).toContain("prepare_pipeline_run");
    expect(created[0]?.systemPrompt).toMatch(/approved pipeline actions are dry-runs only/i);
    expect(created[0]?.systemPrompt).toMatch(/real training requires an explicit direct.*tt pipeline run/is);
    expect(created[0]?.systemPrompt).toMatch(/account and cloud tools are unavailable/i);
    expect(created[0]?.systemPrompt).toMatch(/adapter.*foundation.*describe_pipeline/is);
    expect(created[0]?.systemPrompt).toContain("search_hugging_face");
    expect(created[0]?.systemPrompt).toMatch(/models or datasets/i);
    expect(created[0]?.systemPrompt).toMatch(/Hugging Face.*never include secrets or private data/is);
    expect(created[0]?.systemPrompt).toContain("inspect_training_source");
    expect(created[0]?.systemPrompt).toMatch(/observed behavior.*inferred rationale/is);
    expect(created[0]?.systemPrompt).toContain("examine_hardware");
    expect(created[0]?.systemPrompt).toMatch(/examine this host.*GPU/is);
    expect(JSON.stringify(created[0])).not.toContain(TT_SECRET);

    await client.runTurn(thread.id, "Continue", () => {});
    expect(created[1]?.messages.length).toBeGreaterThan(0);
    await client.runTurn(thread.id, `please use ${TT_SECRET}`, () => {});
    expect(prompts.at(-1)).toBe("please use [REDACTED]");

    const failureEvents: string[] = [];
    const failed = await client.runTurn(thread.id, "Fail", (event) =>
      failureEvents.push(event.type)
    );
    expect(failed.status).toBe("failed");
    expect(failureEvents).toEqual(["turn_started", "error", "final"]);
  });

  it("propagates post-dispatch uncertainty as a non-retryable outcome", async () => {
    const store = new LocalAgentStore(root);
    const threadId = "251f122f-dd8e-4894-a0ab-99965e976e29";
    const actionId = "ed8e4bca-ab1c-4c9f-8b65-9f7997f76670";
    await store.save({
      thread: {
        id: threadId,
        title: "Create spec",
        status: "active",
        last_message_at: null,
        created_at: "2026-08-09T10:00:00.000Z",
        updated_at: "2026-08-09T10:00:00.000Z",
      },
      messages: [],
      actions: [{
        id: actionId,
        operation: "create_spec",
        title: "Create spec",
        summary: "Create Support",
        risk: "Creates a remote spec",
        status: "proposed",
        cloud_context: CLOUD_CONTEXT,
        arguments: { spec: { name: "Support", examples: [{ input: "a", output: "b" }] } },
      }],
    });
    const client = createLocalAgentClient({
      store,
      workspaceRoot: root,
      selection: { provider: "openai", model: "gpt-5.2", thinking: "high" },
      modelRuntime: {
        getProviders: () => [{ id: "openai" }],
        getModels: () => [{ id: "gpt-5.2", provider: "openai", reasoning: true }],
        getModel: () => ({ id: "gpt-5.2", provider: "openai", reasoning: true }),
        hasConfiguredAuth: () => true,
      },
      createAgent: vi.fn(),
      cloudEnabled: true,
      cloudContext: CLOUD_CONTEXT,
      toolApi: { get: vi.fn(), postRead: vi.fn(), propose: vi.fn() },
      mutationApi: {
        get: vi.fn(async () => ({ data: { capabilities: ["local_agent_spec_mutation_guards_v1"] } })),
        post: vi.fn(async () => { throw new Error("response lost"); }),
        put: vi.fn(),
      },
    });
    const events: Array<{ type: string; payload: Record<string, unknown> }> = [];

    await expect(client.approveAction(actionId, (event) => events.push(event))).rejects.toThrow(
      /outcome is unknown/i,
    );
    expect(events).toContainEqual(expect.objectContaining({
      type: "action_result",
      payload: expect.objectContaining({ status: "outcome_unknown" }),
    }));
    expect((await store.load(threadId)).actions[0]?.status).toBe("outcome_unknown");
  });

  it("keeps a pipeline approval claim available after a wrong-workspace preflight", async () => {
    const wrongWorkspace = mkdtempSync(join(tmpdir(), "tt-local-agent-wrong-workspace-"));
    const store = new LocalAgentStore(root);
    const threadId = "4d8667fe-b1c9-4c4a-8b56-9e1333ce1f66";
    const actionId = "22f2ad59-c7ea-4530-bdde-793c65f34c17";
    const tamperedActionId = "e83966e0-d205-4814-a714-f5262b79de5f";
    const spec = JSON.stringify({
      name: "Sentiment",
      base_model: "Qwen/Qwen3.5-2B",
      system_prompt: "Classify sentiment.",
      guidelines: ["Return one label."],
      examples: [
        { input: "Great", output: "positive" },
        { input: "Awful", output: "negative" },
      ],
    });
    writeFileSync(join(root, "tunedtensor.json"), spec);
    writeFileSync(join(wrongWorkspace, "tunedtensor.json"), spec);
    const prepared = await prepareLocalPipelineAction({ workspaceRoot: root });
    await store.save({
      thread: {
        id: threadId,
        title: "Preview pipeline",
        status: "active",
        last_message_at: null,
        created_at: "2026-08-31T10:00:00.000Z",
        updated_at: "2026-08-31T10:00:00.000Z",
      },
      messages: [],
      actions: [{
        id: actionId,
        operation: "run_local_pipeline",
        title: "Dry-run local pipeline",
        summary: "Preview the adapter pipeline.",
        risk: "medium",
        status: "proposed",
        arguments: {
          pipeline: prepared.pipeline,
          spec_path: prepared.specPath,
          spec_sha256: prepared.specSha256,
          workspace_fingerprint: prepared.workspaceFingerprint,
          dry_run: true,
        },
      }, {
        id: tamperedActionId,
        operation: "run_local_pipeline",
        title: "Tampered real pipeline",
        summary: "A persisted action that must not dispatch.",
        risk: "high",
        status: "proposed",
        arguments: {
          pipeline: prepared.pipeline,
          spec_path: prepared.specPath,
          spec_sha256: prepared.specSha256,
          workspace_fingerprint: prepared.workspaceFingerprint,
          dry_run: false,
        },
      }],
    });
    const runPipelineCommand = vi.fn(async () => ({ exitCode: 0, signal: null }));
    const client = createLocalAgentClient({
      store,
      workspaceRoot: root,
      selection: { provider: "openai", model: "gpt-5.2", thinking: "high" },
      modelRuntime: {
        getProviders: () => [{ id: "openai" }],
        getModels: () => [{ id: "gpt-5.2", provider: "openai", reasoning: true }],
        getModel: () => ({ id: "gpt-5.2", provider: "openai", reasoning: true }),
        hasConfiguredAuth: () => true,
      },
      createAgent: vi.fn(),
      toolApi: { get: vi.fn(), postRead: vi.fn(), propose: vi.fn() },
      mutationApi: { get: vi.fn(), post: vi.fn(), put: vi.fn() },
      runPipelineCommand,
    });

    try {
      await expect(client.approveAction(actionId, vi.fn(), undefined, {
        workspaceRoot: wrongWorkspace,
      })).rejects.toThrow(/workspace changed after.*prepared/i);
      expect((await store.load(threadId)).actions[0]?.status).toBe("proposed");
      await expect(client.approveAction(tamperedActionId, vi.fn(), undefined, {
        workspaceRoot: root,
      })).rejects.toThrow(/real pipeline execution requires.*direct tt pipeline run/i);
      expect((await store.load(threadId)).actions[1]?.status).toBe("failed");
      expect(runPipelineCommand).not.toHaveBeenCalled();
      await expect(client.approveAction(actionId, vi.fn(), undefined, {
        workspaceRoot: root,
      })).resolves.toMatchObject({ status: "completed" });
      expect(runPipelineCommand).toHaveBeenCalledTimes(1);
    } finally {
      rmSync(wrongWorkspace, { recursive: true, force: true });
    }
  });

  it("keeps a local spec claim available after a wrong-workspace preflight", async () => {
    const store = new LocalAgentStore(root);
    const activeWorkspace = join(root, "active-workspace");
    mkdirSync(activeWorkspace);
    const threadId = "251f122f-dd8e-4894-a0ab-99965e976e29";
    const actionId = "ed8e4bca-ab1c-4c9f-8b65-9f7997f76670";
    const spec = {
      name: "Sentiment classifier",
      base_model: "Qwen/Qwen3.5-2B" as const,
      system_prompt: "Classify sentiment and return only the label.",
      guidelines: ["Return positive, neutral, or negative."],
      examples: [
        { input: "Excellent work.", output: "positive" },
        { input: "This is disappointing.", output: "negative" },
      ],
    };
    const prepared = await prepareLocalSpecProject(activeWorkspace, "sentiment-demo", spec);
    await store.save({
      thread: {
        id: threadId,
        title: "Create local spec",
        status: "active",
        last_message_at: null,
        created_at: "2026-08-09T10:00:00.000Z",
        updated_at: "2026-08-09T10:00:00.000Z",
      },
      messages: [],
      actions: [{
        id: actionId,
        operation: "create_local_spec",
        title: "Create local spec",
        summary: "Create ./sentiment-demo/tunedtensor.json",
        risk: "Creates local files",
        status: "proposed",
        arguments: {
          directory: "sentiment-demo",
          spec,
          workspace_fingerprint: prepared.workspaceFingerprint,
        },
      }],
    });
    const mutationApi = { get: vi.fn(), post: vi.fn(), put: vi.fn() };
    const client = createLocalAgentClient({
      store,
      workspaceRoot: root,
      selection: { provider: "openai", model: "gpt-5.2", thinking: "high" },
      modelRuntime: {
        getProviders: () => [{ id: "openai" }],
        getModels: () => [{ id: "gpt-5.2", provider: "openai", reasoning: true }],
        getModel: () => ({ id: "gpt-5.2", provider: "openai", reasoning: true }),
        hasConfiguredAuth: () => true,
      },
      createAgent: vi.fn(),
      toolApi: { get: vi.fn(), postRead: vi.fn(), propose: vi.fn() },
      mutationApi,
    });

    mkdirSync(join(root, "sentiment-demo"));
    await expect(client.approveAction(actionId, () => {}, undefined, {
      mode: "local",
      workspaceRoot: root,
    })).rejects.toThrow(/workspace changed after.*prepared/i);
    expect((await store.load(threadId)).actions[0]?.status).toBe("proposed");

    await expect(client.approveAction(actionId, () => {}, undefined, {
      mode: "local",
      workspaceRoot: activeWorkspace,
    })).resolves.toMatchObject({
      status: "completed",
    });
    expect(JSON.parse(readFileSync(
      join(activeWorkspace, "sentiment-demo", "tunedtensor.json"),
      "utf8",
    )))
      .toEqual(spec);
    expect((await store.load(threadId)).actions[0]?.status).toBe("completed");
    expect(mutationApi.get).not.toHaveBeenCalled();
    expect(mutationApi.post).not.toHaveBeenCalled();
    expect(mutationApi.put).not.toHaveBeenCalled();
  });
});
