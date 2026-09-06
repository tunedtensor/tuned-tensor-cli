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
  cloud_context?: CloudActionContext;
}

/** Non-secret binding of a local approval to one TT account credential and API origin. */
export interface CloudActionContext {
  origin: string;
  credential_fingerprint: string;
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

export interface AgentTurnContext {
  mode?: "cloud" | "local";
  workspaceRoot?: string;
}

/**
 * UI-facing conversation seam implemented by the local agent runtime.
 *
 * Threads and turn orchestration stay local. Managed model inference uses a
 * separate proxy transport and does not change this conversation contract.
 */
export interface AgentConversationClient {
  createThread(): Promise<AgentThread>;
  listThreads(): Promise<AgentThread[]>;
  getThread(id: string): Promise<AgentThreadDetail>;
  runTurn(
    threadId: string,
    prompt: string,
    onEvent: (event: AgentStreamEvent) => void,
    signal?: AbortSignal,
    context?: AgentTurnContext,
  ): Promise<AgentTurnResult>;
  approveAction(
    actionId: string,
    onEvent: (event: AgentStreamEvent) => void,
    signal?: AbortSignal,
    context?: AgentTurnContext,
  ): Promise<AgentTurnResult>;
  rejectAction(actionId: string): Promise<void>;
}
