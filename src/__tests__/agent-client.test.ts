import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  consumeAgentEventStream,
  createAgentClient,
  createAgentEventDecoder,
  type AgentStreamEvent,
} from "../agent-client.js";

const API_KEY = `tt_${"a".repeat(48)}`;

beforeEach(() => {
  process.env.TUNED_TENSOR_API_KEY = API_KEY;
  process.env.TUNED_TENSOR_URL = "https://test.tunedtensor.com";
});

afterEach(() => {
  delete process.env.TUNED_TENSOR_API_KEY;
  delete process.env.TUNED_TENSOR_URL;
  vi.restoreAllMocks();
});

describe("agent event decoding", () => {
  it("decodes chunked and multiline server-sent events", () => {
    const events: AgentStreamEvent[] = [];
    const decoder = createAgentEventDecoder((event) => events.push(event));
    decoder.push('data: {"type":"text_delta","delta":"hel');
    decoder.push('lo"}\n\nevent: final\ndata: {"status":');
    decoder.push('"completed"}\n\n');
    decoder.finish();

    expect(events).toEqual([
      { type: "text_delta", payload: { type: "text_delta", delta: "hello" } },
      { type: "final", payload: { status: "completed" } },
    ]);
  });

  it("consumes a web response body", async () => {
    const events: AgentStreamEvent[] = [];
    const response = new Response(
      'data: {"type":"text_delta","delta":"hi"}\n\ndata: {"type":"final","status":"completed"}\n\n',
      { headers: { "Content-Type": "text/event-stream" } },
    );
    await consumeAgentEventStream(response, (event) => events.push(event));
    expect(events.map((event) => event.type)).toEqual(["text_delta", "final"]);
  });
});

describe("agent client", () => {
  it("loads resumable approval state with a conversation", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({
        data: {
          thread: {
            id: "thread-1",
            title: "Train support model",
            status: "active",
            last_message_at: "2026-07-30T10:00:00Z",
            created_at: "2026-07-30T10:00:00Z",
            updated_at: "2026-07-30T10:00:00Z",
          },
          actions: [{
            id: "action-1",
            title: "Start training run",
            summary: "Estimated cost £2.00",
            risk: "high",
            status: "proposed",
          }],
        },
      }), { headers: { "Content-Type": "application/json" } }),
    );

    const detail = await createAgentClient().getThread("thread-1");
    expect(detail.thread.id).toBe("thread-1");
    expect(detail.actions).toEqual([{
      id: "action-1",
      title: "Start training run",
      summary: "Estimated cost £2.00",
      risk: "high",
      status: "proposed",
      operation: undefined,
      thread_id: undefined,
      turn_id: undefined,
      arguments: undefined,
      preview: undefined,
      method: undefined,
      path: undefined,
    }]);
  });

  it("creates a conversation and streams an authenticated turn", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        new Response(JSON.stringify({
          data: {
            id: "thread-1",
            title: "New conversation",
            status: "active",
            last_message_at: null,
            created_at: "2026-07-30T10:00:00Z",
            updated_at: "2026-07-30T10:00:00Z",
          },
        }), { headers: { "Content-Type": "application/json" } }),
      )
      .mockResolvedValueOnce(
        new Response(
          [
            'data: {"type":"turn_started","thread_id":"thread-1","turn_id":"turn-1"}',
            'data: {"type":"reasoning_delta","delta":"Checking the account context."}',
            'data: {"type":"text_delta","delta":"Hello"}',
            'data: {"type":"final","thread_id":"thread-1","turn_id":"turn-1","status":"completed"}',
            "",
          ].join("\n\n"),
          { headers: { "Content-Type": "text/event-stream" } },
        ),
      );

    const client = createAgentClient();
    const thread = await client.createThread();
    const seen: string[] = [];
    const result = await client.runTurn(
      thread.id,
      "Hello",
      (event) => seen.push(event.type),
    );

    expect(result).toEqual({
      threadId: "thread-1",
      turnId: "turn-1",
      status: "completed",
      response: "Hello",
      actions: [],
    });
    expect(seen).toEqual([
      "turn_started",
      "reasoning_delta",
      "text_delta",
      "final",
    ]);
    const [, turnInit] = fetchSpy.mock.calls[1]!;
    expect(turnInit?.headers).toMatchObject({
      Authorization: `Bearer ${API_KEY}`,
      Accept: "text/event-stream",
      "Content-Type": "application/json",
    });
    expect(turnInit?.body).toBe(JSON.stringify({ prompt: "Hello" }));
  });

  it("rejects streams that end without a final event", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response('data: {"type":"text_delta","delta":"partial"}\n\n'),
    );
    const client = createAgentClient();
    await expect(
      client.runTurn("thread-1", "hello", () => {}),
    ).rejects.toThrow(/before its final status/);
  });
});
