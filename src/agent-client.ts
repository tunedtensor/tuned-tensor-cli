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

/**
 * UI-facing conversation seam implemented by the local Pi runtime.
 *
 * This contract intentionally has no transport assumptions: ordinary CLI
 * flows do not call hosted `/agent/*` runtime endpoints.
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
  ): Promise<AgentTurnResult>;
  approveAction(
    actionId: string,
    onEvent: (event: AgentStreamEvent) => void,
    signal?: AbortSignal,
  ): Promise<AgentTurnResult>;
  rejectAction(actionId: string): Promise<void>;
}
