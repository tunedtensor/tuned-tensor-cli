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
  /** Compact outcome of an approved action the model has not been told yet. */
  result?: unknown;
  result_reported?: boolean;
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
  mode?: "local";
  workspaceRoot?: string;
}

/**
 * UI-facing conversation seam implemented by the local agent runtime.
 *
 * Threads, turn orchestration and approvals stay on this machine. Model
 * inference uses the provider the user configured.
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
