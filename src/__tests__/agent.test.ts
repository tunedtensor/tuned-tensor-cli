import { describe, expect, it, vi } from "vitest";
import {
  TunedTensorAgentSession,
  type AgentSessionIO,
} from "../agent.js";
import type {
  AgentConversationClient,
  AgentStreamEvent,
} from "../agent-client.js";

const thread = {
  id: "251f122f-dd8e-4894-a0ab-99965e976e29",
  title: "New conversation",
  status: "active",
  last_message_at: "2026-07-30T10:00:00Z",
  created_at: "2026-07-30T10:00:00Z",
  updated_at: "2026-07-30T10:00:00Z",
};

function testIO() {
  const stdout: string[] = [];
  const stderr: string[] = [];
  const io: AgentSessionIO = {
    write: (text) => stdout.push(text),
    writeError: (text) => stderr.push(text),
    clear: vi.fn(),
  };
  return { io, stdout, stderr };
}

function fakeClient(): AgentConversationClient {
  return {
    createThread: vi.fn(async () => thread),
    listThreads: vi.fn(async () => [thread]),
    getThread: vi.fn(async () => ({ thread, actions: [] })),
    runTurn: vi.fn(async (_threadId, _prompt, onEvent) => {
      const events: AgentStreamEvent[] = [
        {
          type: "turn_started",
          payload: { thread_id: thread.id, turn_id: "turn-1" },
        },
        {
          type: "reasoning_delta",
          payload: { delta: "I should inspect the available context first." },
        },
        { type: "text_delta", payload: { delta: "Here is the answer." } },
        {
          type: "approval_required",
          payload: {
            action: {
              id: "action-123",
              title: "Create behaviour spec",
              summary: "Create the reviewed Support spec.",
              risk: "medium",
              operation: "create_spec",
              arguments: { spec: { name: "Support" } },
              method: "POST",
              path: "/api/v1/behavior-specs",
            },
          },
        },
        { type: "final", payload: { status: "waiting_for_approval" } },
      ];
      for (const event of events) onEvent(event);
      return {
        threadId: thread.id,
        turnId: "turn-1",
        status: "waiting_for_approval",
        response: "Here is the answer.",
        actions: [{
          id: "action-123",
          title: "Create behaviour spec",
          summary: "Create the reviewed Support spec.",
          risk: "medium",
        }],
      };
    }),
    approveAction: vi.fn(async (_actionId, onEvent) => {
      onEvent({ type: "action_started", payload: {} });
      onEvent({ type: "action_result", payload: { status: "completed" } });
      onEvent({ type: "final", payload: { status: "completed" } });
      return {
        threadId: thread.id,
        turnId: "turn-1",
        status: "completed",
        response: "",
        actions: [],
      };
    }),
    rejectAction: vi.fn(async () => {}),
  };
}

describe("TunedTensorAgentSession", () => {
  it("keeps a conversation in the shell and exposes approval controls", async () => {
    const client = fakeClient();
    const { io, stdout, stderr } = testIO();
    const session = new TunedTensorAgentSession({ client, io });

    await session.handleLine("What should I train next?");
    expect(client.createThread).toHaveBeenCalledTimes(1);
    expect(client.runTurn).toHaveBeenCalledWith(
      thread.id,
      "What should I train next?",
      expect.any(Function),
      expect.any(AbortSignal),
    );
    expect(stdout.join("")).toContain("Here is the answer.");
    expect(stdout.join("")).toContain(
      "I should inspect the available context first.",
    );
    expect(stdout.join("")).toContain("Approval required");
    expect(stdout.join("")).toContain("will not run without approval");
    expect(stdout.join("")).toContain("\"name\": \"Support\"");
    expect(stdout.join("")).toContain(
      "\"request\": \"POST /api/v1/behavior-specs\"",
    );
    expect(session.snapshot().pendingActions).toHaveLength(1);

    await session.handleLine("/approve");
    expect(client.approveAction).toHaveBeenCalledWith(
      "action-123",
      expect.any(Function),
      expect.any(AbortSignal),
    );
    expect(session.snapshot().pendingActions).toHaveLength(0);
    expect(stderr).toEqual([]);
  });

  it("keeps a prepared local action pending until it is approved in local mode", async () => {
    const client = fakeClient();
    client.runTurn = vi.fn(async (_threadId, _prompt, onEvent) => {
      const action = {
        id: "local-action-123",
        title: "Create local Tuned Tensor spec",
        summary: "Create ./support/tunedtensor.json.",
        risk: "medium",
        operation: "create_local_spec",
        arguments: {},
      };
      onEvent({ type: "approval_required", payload: { action } });
      onEvent({ type: "final", payload: { status: "waiting_for_approval" } });
      return {
        threadId: thread.id,
        turnId: "turn-local",
        status: "waiting_for_approval",
        response: "",
        actions: [action],
      };
    });
    const { io } = testIO();
    const session = new TunedTensorAgentSession({ client, io });
    const localContext = { mode: "local" as const, workspaceRoot: "/tmp/workspace" };

    await session.handleLine("create a local support spec", localContext);
    await session.handleLine("/approve", localContext);

    expect(client.approveAction).toHaveBeenCalledTimes(1);
    expect(session.snapshot().pendingActions).toHaveLength(0);
  });

  it("removes outcome-unknown actions and shows non-retryable recovery guidance", async () => {
    const client = fakeClient();
    client.approveAction = vi.fn(async (_actionId, onEvent) => {
      onEvent({ type: "action_result", payload: { status: "outcome_unknown", error: "lost response" } });
      throw new Error("The mutation outcome is unknown and this action cannot be retried automatically.");
    });
    const { io, stderr } = testIO();
    const session = new TunedTensorAgentSession({ client, io });

    await session.handleLine("prepare a spec");
    await session.handleLine("/approve");

    expect(session.snapshot().pendingActions).toHaveLength(0);
    expect(stderr.join(" ")).toMatch(/outcome is unknown.*cannot be retried.*inspect the remote spec/i);
    expect(await session.handleLine("/approve")).toBe("continue");
    expect(client.approveAction).toHaveBeenCalledTimes(1);
  });

  it("does not abort an approved action while it is settling", async () => {
    const client = fakeClient();
    let finish!: () => void;
    let approvalSignal: AbortSignal | undefined;
    client.approveAction = vi.fn(async (_actionId, _onEvent, signal) => {
      approvalSignal = signal;
      await new Promise<void>((resolve) => { finish = resolve; });
      return {
        threadId: thread.id,
        turnId: "turn-1",
        status: "completed",
        response: "",
        actions: [],
      };
    });
    const { io, stderr } = testIO();
    const session = new TunedTensorAgentSession({ client, io });
    await session.handleLine("prepare a spec");

    const settling = session.handleLine("/approve");
    await Promise.resolve();
    expect(session.interrupt()).toBe(true);
    expect(approvalSignal?.aborted).toBe(false);
    expect(stderr.join(" ")).toMatch(/settling.*cannot be interrupted.*do not retry/i);
    finish();
    await settling;
    expect(session.snapshot().pendingActions).toHaveLength(0);
  });

  it("allows an approved local pipeline to be stopped with Ctrl-C", async () => {
    const client = fakeClient();
    const action = {
      id: "pipeline-action-123",
      title: "Run local pipeline",
      summary: "Run the adapter pipeline.",
      risk: "medium",
      operation: "run_local_pipeline",
      arguments: {},
    };
    client.runTurn = vi.fn(async (_threadId, _prompt, onEvent) => {
      onEvent({ type: "approval_required", payload: { action } });
      return {
        threadId: thread.id,
        turnId: "turn-pipeline",
        status: "waiting_for_approval",
        response: "",
        actions: [action],
      };
    });
    let approvalSignal: AbortSignal | undefined;
    let started!: () => void;
    const didStart = new Promise<void>((resolve) => { started = resolve; });
    client.approveAction = vi.fn(async (_actionId, _onEvent, signal) => {
      approvalSignal = signal;
      started();
      await new Promise<void>((resolve) => {
        signal?.addEventListener("abort", () => resolve(), { once: true });
      });
      return {
        threadId: thread.id,
        turnId: "turn-pipeline",
        status: "cancelled",
        response: "",
        actions: [],
      };
    });
    const { io, stdout } = testIO();
    const session = new TunedTensorAgentSession({ client, io });
    await session.handleLine("train the current spec");

    const settling = session.handleLine("/approve");
    await didStart;
    expect(session.interrupt()).toBe(true);
    expect(approvalSignal?.aborted).toBe(true);
    await settling;
    expect(stdout.join(" ")).toContain("Response stopped");
    expect(session.snapshot().pendingActions).toHaveLength(0);
  });

  it("reuses the active thread and supports new and resumed conversations", async () => {
    const client = fakeClient();
    const { io } = testIO();
    const session = new TunedTensorAgentSession({ client, io });

    await session.handleLine("first");
    await session.handleLine("second");
    expect(client.createThread).toHaveBeenCalledTimes(1);
    expect(client.runTurn).toHaveBeenCalledTimes(2);

    await session.handleLine("/new");
    await session.handleLine("third");
    expect(client.createThread).toHaveBeenCalledTimes(2);

    await session.handleLine(`/resume ${thread.id.slice(0, 8)}`);
    expect(client.listThreads).toHaveBeenCalled();
    expect(client.getThread).toHaveBeenCalledWith(thread.id);
    expect(session.snapshot().thread?.id).toBe(thread.id);
  });

  it("keeps reasoning, tool activity, and the final answer visually ordered", async () => {
    const client = fakeClient();
    client.runTurn = vi.fn(async (_threadId, _prompt, onEvent) => {
      const events: AgentStreamEvent[] = [
        { type: "reasoning_delta", payload: { delta: "Checking runs." } },
        {
          type: "tool_call",
          payload: { name: "list_runs", toolUseId: "tool-1", input: {} },
        },
        {
          type: "tool_result",
          payload: { toolUseId: "tool-1", status: "success" },
        },
        { type: "reasoning_delta", payload: { delta: "Comparing results." } },
        { type: "text_delta", payload: { delta: "The latest run improved." } },
      ];
      for (const event of events) onEvent(event);
      return {
        threadId: thread.id,
        turnId: "turn-1",
        status: "completed",
        response: "The latest run improved.",
        actions: [],
      };
    });
    const { io, stdout } = testIO();
    const session = new TunedTensorAgentSession({ client, io });

    await session.send("Compare my runs");

    const output = stdout.join("");
    const parts = [
      "Checking runs.",
      "list_runs",
      "Tool complete",
      "Comparing results.",
      "The latest run improved.",
    ];
    for (let index = 1; index < parts.length; index += 1) {
      expect(output.indexOf(parts[index - 1]!)).toBeLessThan(
        output.indexOf(parts[index]!),
      );
    }
    expect(output.split("tt  ")).toHaveLength(2);
  });

  it("strips terminal controls from streamed reasoning", async () => {
    const client = fakeClient();
    client.runTurn = vi.fn(async (_threadId, _prompt, onEvent) => {
      onEvent({
        type: "reasoning_delta",
        payload: { delta: "safe\u001b[2J reasoning\u0007" },
      });
      return {
        threadId: thread.id,
        turnId: "turn-1",
        status: "completed",
        response: "",
        actions: [],
      };
    });
    const { io, stdout } = testIO();
    const session = new TunedTensorAgentSession({ client, io });

    await session.send("think safely");

    const output = stdout.join("");
    expect(output).toContain("safe reasoning");
    expect(output).not.toContain("[2J");
    expect(output).not.toContain("\u0007");
  });

  it("renders streamed assistant Markdown without exposing syntax markers", async () => {
    const client = fakeClient();
    client.runTurn = vi.fn(async (_threadId, _prompt, onEvent) => {
      onEvent({ type: "text_delta", payload: { delta: "**Behaviour" } });
      onEvent({
        type: "text_delta",
        payload: { delta: " Specs**\n- List and inspect `specs`." },
      });
      return {
        threadId: thread.id,
        turnId: "turn-1",
        status: "completed",
        response: "**Behaviour Specs**\n- List and inspect `specs`.",
        actions: [],
      };
    });
    const { io, stdout } = testIO();
    const session = new TunedTensorAgentSession({ client, io });

    await session.send("What can you do?");

    const output = stdout.join("");
    expect(output).toContain("Behaviour Specs");
    expect(output).toContain("• List and inspect specs.");
    expect(output).not.toContain("**");
    expect(output).not.toContain("`specs`");
    expect(output.split("tt  ")).toHaveLength(2);
  });

  it("cancels an active response without ending the session", async () => {
    const client = fakeClient();
    let started!: () => void;
    const didStart = new Promise<void>((resolve) => {
      started = resolve;
    });
    client.runTurn = vi.fn(async (_threadId, _prompt, _onEvent, signal) => {
      started();
      return await new Promise<never>((_resolve, reject) => {
        signal?.addEventListener(
          "abort",
          () => reject(new Error("aborted")),
          { once: true },
        );
      });
    });
    const { io, stdout } = testIO();
    const session = new TunedTensorAgentSession({ client, io });

    const pending = session.send("keep thinking");
    await didStart;
    expect(session.interrupt()).toBe(true);
    await expect(pending).resolves.toBeNull();
    expect(stdout.join("")).toContain("Response stopped");
    expect(session.interrupt()).toBe(false);
  });

  it("shows cancellation when the agent resolves normally after abort", async () => {
    const client = fakeClient();
    let started!: () => void;
    const didStart = new Promise<void>((resolve) => { started = resolve; });
    client.runTurn = vi.fn(async (_threadId, _prompt, onEvent, signal) => {
      started();
      await new Promise<void>((resolve) => {
        signal?.addEventListener("abort", () => resolve(), { once: true });
      });
      onEvent({ type: "final", payload: { status: "cancelled" } });
      return {
        threadId: thread.id,
        turnId: "turn-aborted",
        status: "cancelled",
        response: "",
        actions: [],
      };
    });
    const { io, stdout } = testIO();
    const session = new TunedTensorAgentSession({ client, io });

    const pending = session.send("keep thinking");
    await didStart;
    expect(session.interrupt()).toBe(true);
    await expect(pending).resolves.toBeNull();
    expect(stdout.join("")).toContain("Response stopped");
  });
});
