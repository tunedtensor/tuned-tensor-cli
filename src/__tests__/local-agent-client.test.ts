import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { LocalAgentStore } from "../agent-store.js";
import {
  createLocalAgentClient,
  type LocalPiAgent,
  type LocalPiAgentOptions,
} from "../local-agent-client.js";
import type { AgentToolApi } from "../agent-tools.js";

const TT_SECRET = "tt_never_send_this_to_pi";
let root: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "tt-local-agent-"));
});
afterEach(() => rmSync(root, { recursive: true, force: true }));

describe("local Pi conversation client", () => {
  it("maps Pi streaming events to the existing UI seam and resumes locally", async () => {
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
            type: "tool_execution_start", toolCallId: "t1", toolName: "list_specs", args: {},
          });
          for (const listener of listeners) await listener({
            type: "tool_execution_end", toolCallId: "t1", toolName: "list_specs",
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
        arguments: { spec: { name: "Support", examples: [{ input: "a", output: "b" }] } },
      }],
    });
    const client = createLocalAgentClient({
      store,
      selection: { provider: "openai", model: "gpt-5.2", thinking: "high" },
      modelRuntime: {
        getProviders: () => [{ id: "openai" }],
        getModels: () => [{ id: "gpt-5.2", provider: "openai", reasoning: true }],
        getModel: () => ({ id: "gpt-5.2", provider: "openai", reasoning: true }),
        hasConfiguredAuth: () => true,
      },
      createAgent: vi.fn(),
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
});
