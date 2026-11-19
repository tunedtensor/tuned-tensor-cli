import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { LocalAgentStore } from "../agent-store.js";
import {
  createLocalAgentClient,
  type LocalPiAgent,
  type LocalPiAgentOptions,
  toolErrorSummary,
} from "../local-agent-client.js";
import type { AgentToolApi } from "../agent-tools.js";
import { prepareLocalSpecProject } from "../local-spec-workspace.js";
import { prepareLocalPipelineAction } from "../local-pipeline-action.js";
import type { AgentLocalServer } from "../agent-local-server.js";

const PROVIDER_SECRET = "sk_never_send_this_to_pi";
let root: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "tt-local-agent-"));
});
afterEach(() => rmSync(root, { recursive: true, force: true }));

describe("local agent conversation client", () => {
  it.each(["create_spec", "update_spec", "start_label_job", "download_label_job"])(
    "fails a persisted %s proposal from the retired hosted service without dispatch",
    async (operation) => {
      const store = new LocalAgentStore(root);
      const threadId = "251f122f-dd8e-4894-a0ab-99965e976e29";
      const actionId = "ed8e4bca-ab1c-4c9f-8b65-9f7997f76670";
      await store.save({
        thread: {
          id: threadId,
          title: "Legacy proposal",
          status: "active",
          last_message_at: null,
          created_at: "2026-08-09T10:00:00.000Z",
          updated_at: "2026-08-09T10:00:00.000Z",
        },
        messages: [],
        actions: [{
          id: actionId,
          operation,
          title: "Legacy",
          summary: "Prepared by an earlier tt release",
          risk: "Hosted account change",
          status: "proposed",
          arguments: { spec: { name: "Support", examples: [{ input: "a", output: "b" }] } },
        }],
      });
      const fetchSpy = vi.spyOn(globalThis, "fetch");
      const client = createLocalAgentClient({
        store,
        workspaceRoot: root,
        selection: { provider: "openai", model: "gpt-5.2", thinking: "off" },
        modelRuntime: {
          getProviders: () => [{ id: "openai" }],
          getModels: () => [{ id: "gpt-5.2", provider: "openai", reasoning: true }],
          getModel: () => ({ id: "gpt-5.2", provider: "openai", reasoning: true }),
          hasConfiguredAuth: () => true,
        },
        createAgent: vi.fn(),
      });
      try {
        await expect(client.approveAction(actionId, () => {})).rejects.toThrow(/unsupported prepared operation/i);
        expect((await store.load(threadId)).actions[0]?.status).toBe("failed");
        expect(fetchSpy).not.toHaveBeenCalled();
      } finally {
        fetchSpy.mockRestore();
      }
    },
  );

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
          if (prompt === "Length") {
            for (const listener of listeners) await listener({ type: "turn_end", message: { role: "assistant", content: [], stopReason: "length" } });
            return;
          }
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
          if (prompt === "Separated") {
            for (const listener of listeners) await listener({ type: "message_update", assistantMessageEvent: { type: "text_delta", delta: "Result after tool" } });
          }
          state.messages.push({ role: "assistant", content: [{ type: "text", text: "Local answer" }] });
        },
        abort: vi.fn(),
      };
    });
    const toolApi: AgentToolApi = {
      propose: vi.fn(async (action) => action),
    };
    const client = createLocalAgentClient({
      store: new LocalAgentStore(root, { secretValues: [PROVIDER_SECRET] }),
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
        "inspect_dataset_file",
        "prepare_dataset_split",
      "list_local_datasets",
      "inspect_local_dataset",
      "inspect_hugging_face_dataset",
      "inspect_training_source",
      "validate_pipeline",
      "get_local_spec",
      "prepare_update_local_spec",
      "prepare_create_local_spec",
      "prepare_pipeline_run",
      "prepare_download",
            "get_local_service", "prepare_serve_local_model", "prepare_stop_local_service",
        "list_local_runs", "inspect_local_run", "prepare_activate_local_model", "prepare_rollback_local_model", "assess_local_dataset",
]);
    expect(created[0]?.systemPrompt).toMatch(/sealed local pipeline dry-run/i);
    expect(created[0]?.systemPrompt).toContain("prepare_pipeline_run");
    expect(created[0]?.systemPrompt).toMatch(/approved dry-runs never train/i);
    expect(created[0]?.systemPrompt).toMatch(/after a dry-run of the same spec completed.*prepare_background_run/is);
    expect(created[0]?.systemPrompt).toMatch(/tt pipeline run --spec tunedtensor\.json/);
    expect(created[0]?.systemPrompt).toMatch(/no Tuned Tensor account, token or hosted service exists/i);
    expect(created[0]?.systemPrompt).not.toMatch(/tt cloud|tt auth|TT access token|TT credits|labeling job/i);
    expect(created[0]?.systemPrompt).toMatch(/adapter.*foundation.*describe_pipeline/is);
    expect(created[0]?.systemPrompt).toContain("search_hugging_face");
    expect(created[0]?.systemPrompt).toMatch(/models or datasets/i);
    expect(created[0]?.systemPrompt).toMatch(/Hugging Face.*never include secrets or private data/is);
    expect(created[0]?.systemPrompt).toContain('call prepare_download with kind "dataset"');
    expect(created[0]?.systemPrompt).toContain("writes, previews, runs, cancellations and downloads require the user's approval");
    expect(created[0]?.systemPrompt).toContain("inspect_training_source");
    expect(created[0]?.systemPrompt).toMatch(/observed behavior.*inferred rationale/is);
    expect(created[0]?.systemPrompt).toContain("examine_hardware");
    expect(created[0]?.systemPrompt).toContain("User-owned AWS GPU training uses");
    expect(created[0]?.systemPrompt).not.toContain("Cloud execution is explicit through");
    expect(created[0]?.systemPrompt).toContain("do not use a missing local GPU to reject AWS training");
    expect(created[0]?.systemPrompt).toMatch(/examine this host.*GPU/is);
    expect(JSON.stringify(created[0])).not.toContain(PROVIDER_SECRET);

    await client.runTurn(thread.id, "Continue", () => {});
    expect(created[1]?.messages.length).toBeGreaterThan(0);
    await client.runTurn(thread.id, `please use ${PROVIDER_SECRET}`, () => {});
    expect(prompts.at(-1)).toBe("please use [REDACTED]");

    const failureEvents: string[] = [];
    const failed = await client.runTurn(thread.id, "Fail", (event) =>
      failureEvents.push(event.type)
    );
    expect(failed.status).toBe("failed");
    expect(failureEvents).toEqual(["turn_started", "error", "final"]);
    const lengthEvents: unknown[] = [];
    const limited = await client.runTurn(thread.id, "Length", event => lengthEvents.push(event));
    expect(limited.status).toBe("failed");
    expect(lengthEvents).toContainEqual({ type: "error", payload: { message: expect.stringContaining("model reached its response or context limit") } });
    const separatedText: string[] = [];
    const separateThread = await client.createThread();
    const separated = await client.runTurn(separateThread.id, "Separated", event => {
      if (event.type === "text_delta") separatedText.push(String(event.payload.delta));
    });
    expect(separated.response).toBe("Local answer\n\nResult after tool");
    expect(separatedText.join("")).toBe(separated.response);
  });

  it("propagates post-dispatch uncertainty as a non-retryable outcome", async () => {
    const store = new LocalAgentStore(root);
    const threadId = "251f122f-dd8e-4894-a0ab-99965e976e29";
    const actionId = "ed8e4bca-ab1c-4c9f-8b65-9f7997f76670";
    const serverId = "11111111-1111-4111-8111-111111111111";
    await store.save({
      thread: {
        id: threadId,
        title: "Stop server",
        status: "active",
        last_message_at: null,
        created_at: "2026-08-09T10:00:00.000Z",
        updated_at: "2026-08-09T10:00:00.000Z",
      },
      messages: [],
      actions: [{
        id: actionId,
        operation: "stop_local_service",
        title: "Stop local model server",
        summary: "Release the GPU used by this session's server.",
        risk: "Stops a local process",
        status: "proposed",
        arguments: { server_id: serverId },
      }],
    });
    const localServer = {
      status: vi.fn(async () => ({ server_id: serverId })),
      stop: vi.fn(async () => { throw new Error("process did not confirm exit"); }),
    } as unknown as AgentLocalServer;
    const client = createLocalAgentClient({
      store,
      localServer,
      workspaceRoot: root,
      selection: { provider: "openai", model: "gpt-5.2", thinking: "high" },
      modelRuntime: {
        getProviders: () => [{ id: "openai" }],
        getModels: () => [{ id: "gpt-5.2", provider: "openai", reasoning: true }],
        getModel: () => ({ id: "gpt-5.2", provider: "openai", reasoning: true }),
        hasConfiguredAuth: () => true,
      },
      createAgent: vi.fn(),
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
    const prepared = await prepareLocalSpecProject(activeWorkspace, spec);
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
        summary: "Create ./tunedtensor.json",
        risk: "Creates local files",
        status: "proposed",
        arguments: {
          spec,
          workspace_fingerprint: prepared.workspaceFingerprint,
        },
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
    });

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
      join(activeWorkspace, "tunedtensor.json"),
      "utf8",
    )))
      .toEqual({ id: expect.any(String), ...spec });
    expect((await store.load(threadId)).actions[0]?.status).toBe("completed");
  });
});

describe("toolErrorSummary", () => {
  it("returns the first bounded line of a failed tool's text", () => {
    expect(toolErrorSummary({
      content: [{ type: "text", text: "Hugging Face search failed: could not reach huggingface.co (ENOTFOUND).\nstack" }],
    })).toBe("Hugging Face search failed: could not reach huggingface.co (ENOTFOUND).");
    expect(toolErrorSummary({ content: [{ type: "text", text: "x".repeat(300) }] })).toHaveLength(200);
    expect(toolErrorSummary({ content: [] })).toBeUndefined();
    expect(toolErrorSummary(undefined)).toBeUndefined();
  });
});
