import {
  get,
  post,
  postStream,
  type ClientOpts,
  type StreamClientOpts,
} from "./client.js";

export interface AgentThread {
  id: string;
  title: string;
  status: string;
  last_message_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface AgentAction {
  id: string;
  thread_id?: string;
  turn_id?: string;
  operation?: string;
  title: string;
  summary: string;
  risk: string;
  status?: string;
  arguments?: unknown;
  preview?: unknown;
  method?: string;
  path?: string;
}

export interface AgentThreadDetail {
  thread: AgentThread;
  actions: AgentAction[];
}

export interface AgentStreamEvent {
  type: string;
  payload: Record<string, unknown>;
}

export interface AgentTurnResult {
  threadId: string;
  turnId: string | null;
  status: string | null;
  response: string;
  actions: AgentAction[];
}

export interface AgentConversationClient {
  createThread(): Promise<AgentThread>;
  listThreads(): Promise<AgentThread[]>;
  getThread(id: string): Promise<AgentThreadDetail>;
  runTurn(
    threadId: string,
    prompt: string,
    onEvent: (event: AgentStreamEvent) => void,
    signal?: AbortSignal,
  ): Promise<AgentTurnResult>;
  approveAction(
    actionId: string,
    onEvent: (event: AgentStreamEvent) => void,
    signal?: AbortSignal,
  ): Promise<AgentTurnResult>;
  rejectAction(actionId: string): Promise<void>;
}

function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function stringValue(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function dataEnvelope<T>(value: unknown): T {
  const outer = record(value);
  return (outer && "data" in outer ? outer.data : value) as T;
}

function normalizeEvent(block: string): AgentStreamEvent | null {
  let eventName: string | null = null;
  const dataLines: string[] = [];

  for (const rawLine of block.split(/\r?\n/)) {
    if (!rawLine || rawLine.startsWith(":")) continue;
    const separator = rawLine.indexOf(":");
    const field = separator === -1 ? rawLine : rawLine.slice(0, separator);
    const rawValue = separator === -1 ? "" : rawLine.slice(separator + 1);
    const value = rawValue.startsWith(" ") ? rawValue.slice(1) : rawValue;
    if (field === "event") eventName = value;
    if (field === "data") dataLines.push(value);
  }

  const rawData = dataLines.join("\n");
  if (!eventName && !rawData) return null;
  if (rawData === "[DONE]") return { type: "final", payload: {} };

  let decoded: unknown = rawData;
  if (rawData) {
    try {
      decoded = JSON.parse(rawData) as unknown;
    } catch {
      decoded = rawData;
    }
  }

  const decodedRecord = record(decoded);
  const inferredType =
    stringValue(decodedRecord?.type) ?? stringValue(decodedRecord?.event);
  const nestedData = record(decodedRecord?.data);
  const payload =
    decodedRecord && nestedData && inferredType
      ? nestedData
      : decodedRecord ?? (rawData ? { text: rawData } : {});
  const type = (eventName ?? inferredType ?? "message")
    .toLowerCase()
    .replaceAll("-", "_");
  return { type, payload };
}

export function createAgentEventDecoder(
  onEvent: (event: AgentStreamEvent) => void,
): { push(chunk: string): void; finish(): void } {
  let buffer = "";

  const drain = (flush: boolean) => {
    while (true) {
      const boundary = buffer.match(/\r?\n\r?\n/);
      if (!boundary || boundary.index === undefined) break;
      const block = buffer.slice(0, boundary.index);
      buffer = buffer.slice(boundary.index + boundary[0].length);
      const event = normalizeEvent(block);
      if (event) onEvent(event);
    }
    if (flush && buffer) {
      const event = normalizeEvent(buffer);
      buffer = "";
      if (event) onEvent(event);
    }
  };

  return {
    push(chunk) {
      buffer += chunk;
      drain(false);
    },
    finish() {
      drain(true);
    },
  };
}

export async function consumeAgentEventStream(
  response: Response,
  onEvent: (event: AgentStreamEvent) => void,
): Promise<void> {
  if (!response.body) throw new Error("The agent returned an empty response.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const events = createAgentEventDecoder(onEvent);

  while (true) {
    const { value, done } = await reader.read();
    events.push(decoder.decode(value, { stream: !done }));
    if (done) break;
  }
  events.finish();
}

function normalizeAction(value: unknown): AgentAction | null {
  const input = record(value);
  const id = stringValue(input?.id);
  const title = stringValue(input?.title);
  if (!input || !id || !title) return null;
  return {
    id,
    title,
    summary: stringValue(input.summary) ?? "",
    risk: stringValue(input.risk) ?? "unknown",
    status: stringValue(input.status) ?? undefined,
    operation: stringValue(input.operation) ?? undefined,
    thread_id: stringValue(input.thread_id) ?? undefined,
    turn_id: stringValue(input.turn_id) ?? undefined,
    arguments: input.arguments,
    preview: input.preview,
    method: stringValue(input.method) ?? undefined,
    path: stringValue(input.path) ?? undefined,
  };
}

async function streamResult(
  response: Response,
  fallbackThreadId: string,
  onEvent: (event: AgentStreamEvent) => void,
): Promise<AgentTurnResult> {
  let threadId = fallbackThreadId;
  let turnId: string | null = null;
  let status: string | null = null;
  let assistantText = "";
  let sawFinal = false;
  const actions: AgentAction[] = [];

  await consumeAgentEventStream(response, (event) => {
    const payload = event.payload;
    if (event.type === "turn_started") {
      threadId =
        stringValue(payload.thread_id) ??
        stringValue(payload.threadId) ??
        threadId;
      turnId =
        stringValue(payload.turn_id) ??
        stringValue(payload.turnId) ??
        turnId;
    } else if (event.type === "text_delta") {
      assistantText +=
        stringValue(payload.delta) ??
        stringValue(payload.text) ??
        stringValue(payload.content) ??
        "";
    } else if (event.type === "approval_required") {
      const action = normalizeAction(payload.action ?? payload.approval ?? payload);
      if (action && !actions.some((candidate) => candidate.id === action.id)) {
        actions.push(action);
      }
    } else if (event.type === "final") {
      sawFinal = true;
      status = stringValue(payload.status) ?? status;
      threadId =
        stringValue(payload.thread_id) ??
        stringValue(payload.threadId) ??
        threadId;
      turnId =
        stringValue(payload.turn_id) ??
        stringValue(payload.turnId) ??
        turnId;
    }
    onEvent(event);
  });

  if (!sawFinal) {
    throw new Error("The agent response ended before its final status was confirmed.");
  }
  return { threadId, turnId, status, response: assistantText, actions };
}

export function createAgentClient(
  opts?: ClientOpts,
): AgentConversationClient {
  return {
    async createThread() {
      const response = await post<AgentThread>("/agent/threads", {}, opts);
      return dataEnvelope<AgentThread>(response);
    },
    async listThreads() {
      const response = await get<AgentThread[]>("/agent/threads", undefined, opts);
      return dataEnvelope<AgentThread[]>(response);
    },
    async getThread(id) {
      const response = await get<{
        thread: AgentThread;
        actions: AgentAction[];
      }>(`/agent/threads/${encodeURIComponent(id)}`, undefined, opts);
      const detail = dataEnvelope<{
        thread: AgentThread;
        actions: AgentAction[];
      }>(response);
      return {
        thread: detail.thread,
        actions: (detail.actions ?? [])
          .map((action) => normalizeAction(action))
          .filter((action): action is AgentAction => action !== null),
      };
    },
    async runTurn(threadId, prompt, onEvent, signal) {
      const streamOpts: StreamClientOpts = { ...opts, signal };
      const response = await postStream(
        `/agent/threads/${encodeURIComponent(threadId)}/turns`,
        { prompt },
        streamOpts,
      );
      return await streamResult(response, threadId, onEvent);
    },
    async approveAction(actionId, onEvent, signal) {
      const streamOpts: StreamClientOpts = { ...opts, signal };
      const response = await postStream(
        `/agent/actions/${encodeURIComponent(actionId)}/approve`,
        {},
        streamOpts,
      );
      return await streamResult(response, "", onEvent);
    },
    async rejectAction(actionId) {
      await post(`/agent/actions/${encodeURIComponent(actionId)}/reject`, {}, opts);
    },
  };
}
