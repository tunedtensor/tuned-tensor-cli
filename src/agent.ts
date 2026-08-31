import chalk from "chalk";
import { stripVTControlCharacters } from "node:util";
import {
  type AgentAction,
  type AgentConversationClient,
  type AgentStreamEvent,
  type AgentThread,
  type AgentThreadDetail,
  type AgentTurnContext,
  type AgentTurnResult,
} from "./agent-client.js";
import { tokenizeShellInput } from "./shell.js";
import { StreamingTerminalMarkdown } from "./terminal-markdown.js";

const accent = chalk.hex("#8B5CF6");
const successMark = (): string => chalk.green("✓");
const errorMark = (): string => chalk.red("✗");

export interface AgentSessionIO {
  write(text: string): void;
  writeError(text: string): void;
  clear(): void;
}

export interface AgentSessionOptions {
  client: AgentConversationClient;
  io: AgentSessionIO;
  thread?: AgentThread | null;
}

export type AgentLineAction = "continue" | "exit";

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function stringValue(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function sanitizeTerminalText(text: string): string {
  return stripVTControlCharacters(text)
    .replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f]/g, "");
}

function helpText(): string {
  return [
    accent.bold("Agent commands"),
    "",
    `  ${accent("/new".padEnd(22))} Start a new conversation.`,
    `  ${accent("/threads".padEnd(22))} List recent conversations.`,
    `  ${accent("/resume <id>".padEnd(22))} Resume a conversation by ID or prefix.`,
    `  ${accent("/approve [id]".padEnd(22))} Approve a proposed action.`,
    `  ${accent("/reject [id]".padEnd(22))} Reject a proposed action.`,
    `  ${accent("/status".padEnd(22))} Show the active conversation.`,
    `  ${accent("/clear".padEnd(22))} Clear the terminal.`,
    `  ${accent("/exit".padEnd(22))} Exit the agent.`,
    "",
    chalk.dim(
      "Everything else is sent to the laptop-local Tuned Tensor assistant; known TT commands still run directly in the shell.",
    ),
    "",
  ].join("\n");
}

function formatThread(thread: AgentThread, activeId?: string): string {
  const active = thread.id === activeId ? accent("●") : chalk.dim("○");
  const time = thread.last_message_at ?? thread.updated_at ?? thread.created_at;
  const date = new Date(time);
  const dateLabel = Number.isNaN(date.getTime())
    ? ""
    : date.toLocaleString([], {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
  return `${active} ${chalk.dim(thread.id.slice(0, 8))}  ${sanitizeTerminalText(thread.title)}  ${chalk.dim(dateLabel)}`;
}

function actionFromPayload(payload: Record<string, unknown>): AgentAction | null {
  const value = isRecord(payload.action)
    ? payload.action
    : isRecord(payload.approval)
      ? payload.approval
      : payload;
  const id = stringValue(value.id);
  const title = stringValue(value.title);
  if (!id || !title) return null;
  return {
    id,
    title,
    summary: stringValue(value.summary) ?? "",
    risk: stringValue(value.risk) ?? "unknown",
    status: stringValue(value.status) ?? "proposed",
    operation: stringValue(value.operation) ?? undefined,
    thread_id: stringValue(value.thread_id) ?? undefined,
    turn_id: stringValue(value.turn_id) ?? undefined,
    arguments: value.arguments,
    preview: value.preview,
    method: stringValue(value.method) ?? undefined,
    path: stringValue(value.path) ?? undefined,
  };
}

function formatJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export class TunedTensorAgentSession {
  private thread: AgentThread | null;
  private readonly pendingActions = new Map<string, AgentAction>();
  private activeRequest: { controller: AbortController; interruptible: boolean } | null = null;
  private responseStarted = false;
  private reasoningActive = false;
  private lineOpen = false;
  private readonly responseMarkdown = new StreamingTerminalMarkdown();

  constructor(private readonly options: AgentSessionOptions) {
    this.thread = options.thread ?? null;
  }

  get busy(): boolean {
    return this.activeRequest !== null;
  }

  snapshot(): {
    thread: AgentThread | null;
    pendingActions: AgentAction[];
  } {
    return {
      thread: this.thread,
      pendingActions: [...this.pendingActions.values()],
    };
  }

  interrupt(): boolean {
    if (!this.activeRequest) return false;
    if (!this.activeRequest.interruptible) {
      this.options.io.writeError(
        `${errorMark()} Approved action is settling and cannot be interrupted safely. Do not retry it.\n`,
      );
      return true;
    }
    this.activeRequest.controller.abort("user-stop");
    return true;
  }

  private endOpenLine(): void {
    if (this.lineOpen) this.options.io.write("\n");
    this.lineOpen = false;
    this.reasoningActive = false;
  }

  private writeRenderedResponse(rendered: string): void {
    if (!rendered) return;
    if (!this.responseStarted) {
      this.options.io.write(`\n${accent.bold("tt")}  `);
      this.responseStarted = true;
    }
    this.options.io.write(rendered);
    this.lineOpen = !rendered.endsWith("\n");
  }

  private flushResponse(): void {
    this.writeRenderedResponse(this.responseMarkdown.flush());
  }

  private renderAction(action: AgentAction): void {
    this.endOpenLine();
    this.options.io.write(
      `\n${chalk.yellow.bold("Approval required")}  ${chalk.dim(action.id.slice(0, 8))}\n`,
    );
    this.options.io.write(`${chalk.bold(sanitizeTerminalText(action.title))}  ${chalk.yellow(`[${sanitizeTerminalText(action.risk)} risk]`)}\n`);
    if (action.summary) this.options.io.write(`${sanitizeTerminalText(action.summary)}\n`);
    const request = [action.method, action.path].filter(Boolean).join(" ");
    const technical = {
      operation: action.operation,
      ...(request ? { request } : {}),
      arguments: action.arguments ?? null,
      preview: action.preview ?? null,
    };
    this.options.io.write(`${chalk.bold("Proposed action")}\n`);
    this.options.io.write(`${formatJson(technical)}\n`);
    this.options.io.write(
      chalk.dim("Use /approve or /reject. The action will not run without approval.") + "\n",
    );
  }

  private renderEvent(event: AgentStreamEvent): void {
    const payload = event.payload;
    if (event.type === "text_delta") {
      const delta =
        stringValue(payload.delta) ??
        stringValue(payload.text) ??
        stringValue(payload.content);
      if (!delta) return;
      if (this.reasoningActive) {
        this.endOpenLine();
        this.options.io.write("\n");
      }
      this.writeRenderedResponse(this.responseMarkdown.push(delta));
      return;
    }

    if (event.type === "reasoning_delta") {
      const delta =
        stringValue(payload.delta) ??
        stringValue(payload.text) ??
        stringValue(payload.content);
      if (!delta) return;
      const safeDelta = sanitizeTerminalText(delta);
      if (!safeDelta) return;
      if (!this.reasoningActive) {
        this.flushResponse();
        this.endOpenLine();
        this.options.io.write("\n");
        this.reasoningActive = true;
      }
      this.options.io.write(chalk.dim.italic(safeDelta));
      this.lineOpen = !safeDelta.endsWith("\n");
      return;
    }

    if (event.type === "tool_call") {
      this.flushResponse();
      this.endOpenLine();
      const name =
        stringValue(payload.name) ??
        stringValue(isRecord(payload.tool_call) ? payload.tool_call.name : null) ??
        "tool";
      this.options.io.write(chalk.dim(`  ○ ${name}\n`));
      return;
    }

    if (event.type === "tool_result") {
      this.flushResponse();
      this.endOpenLine();
      const failed = payload.status === "error" || Boolean(payload.error);
      this.options.io.write(
        chalk.dim(`  ${failed ? chalk.red("✗") : chalk.green("✓")} Tool ${failed ? "failed" : "complete"}\n`),
      );
      return;
    }

    if (event.type === "approval_required") {
      this.flushResponse();
      const action = actionFromPayload(payload);
      if (!action) return;
      this.pendingActions.set(action.id, action);
      this.renderAction(action);
      return;
    }

    if (event.type === "action_started") {
      this.flushResponse();
      this.endOpenLine();
      this.options.io.write(chalk.dim("  ○ Running approved action…\n"));
      return;
    }

    if (event.type === "action_result") {
      this.flushResponse();
      this.endOpenLine();
      if (payload.status === "outcome_unknown") {
        this.options.io.writeError(
          `${errorMark()} Approved action outcome is unknown. It cannot be retried; inspect the remote spec before preparing another action.\n`,
        );
        return;
      }
      const failed = payload.status === "failed" || Boolean(payload.error);
      this.options.io.write(
        `${failed ? errorMark() : successMark()} Approved action ${failed ? "failed" : "completed"}.\n`,
      );
      return;
    }

    if (event.type === "error") {
      this.flushResponse();
      this.endOpenLine();
      const message =
        stringValue(payload.message) ??
        stringValue(payload.error) ??
        "The agent could not complete this request.";
      this.options.io.writeError(`${errorMark()} ${sanitizeTerminalText(message)}\n`);
    }
  }

  private async ensureThread(): Promise<AgentThread> {
    if (!this.thread) this.thread = await this.options.client.createThread();
    return this.thread;
  }

  private async runStream(
    operation: (signal: AbortSignal) => Promise<AgentTurnResult>,
    interruptible = true,
  ): Promise<AgentTurnResult | null> {
    if (this.activeRequest) {
      this.options.io.writeError(`${errorMark()} A response is already running.\n`);
      return null;
    }
    const controller = new AbortController();
    this.activeRequest = { controller, interruptible };
    this.responseStarted = false;
    this.reasoningActive = false;
    this.lineOpen = false;
    this.responseMarkdown.reset();
    try {
      const result = await operation(controller.signal);
      this.flushResponse();
      this.endOpenLine();
      for (const action of result.actions) {
        this.pendingActions.set(action.id, action);
      }
      if (controller.signal.aborted || result.status === "cancelled") {
        this.options.io.write(chalk.dim("Response stopped.\n"));
        return null;
      }
      return result;
    } catch (error) {
      this.flushResponse();
      this.endOpenLine();
      if (controller.signal.aborted) {
        this.options.io.write(chalk.dim("Response stopped.\n"));
        return null;
      }
      throw error;
    } finally {
      this.activeRequest = null;
      this.responseStarted = false;
      this.reasoningActive = false;
      this.lineOpen = false;
      this.responseMarkdown.reset();
    }
  }

  async send(prompt: string, context?: AgentTurnContext): Promise<AgentTurnResult | null> {
    const normalized = prompt.trim();
    if (!normalized) return null;
    const thread = await this.ensureThread();
    return await this.runStream(async (signal) => {
      const onEvent = (event: AgentStreamEvent) => this.renderEvent(event);
      return context
        ? await this.options.client.runTurn(
            thread.id,
            normalized,
            onEvent,
            signal,
            context,
          )
        : await this.options.client.runTurn(
            thread.id,
            normalized,
            onEvent,
            signal,
          );
    });
  }

  private resolvePendingAction(idOrPrefix?: string): AgentAction {
    const candidates = [...this.pendingActions.values()].filter((action) =>
      idOrPrefix ? action.id.startsWith(idOrPrefix) : true
    );
    if (candidates.length === 0) {
      throw new Error(
        idOrPrefix
          ? `No pending action matches ${idOrPrefix}.`
          : "There is no pending action.",
      );
    }
    if (candidates.length > 1) {
      throw new Error("More than one action is pending; include its ID or prefix.");
    }
    return candidates[0]!;
  }

  private async approve(
    idOrPrefix?: string,
    context?: AgentTurnContext,
  ): Promise<void> {
    const action = this.resolvePendingAction(idOrPrefix);
    action.status = "executing";
    try {
      await this.runStream(async (signal) => {
        const onEvent = (event: AgentStreamEvent) => this.renderEvent(event);
        return context
          ? await this.options.client.approveAction(action.id, onEvent, signal, context)
          : await this.options.client.approveAction(action.id, onEvent, signal);
      }, action.operation === "run_local_pipeline");
    } finally {
      // Approval is one-way once requested. Preflight failures, completed
      // actions, and unknown outcomes are all non-retryable under this ID.
      this.pendingActions.delete(action.id);
    }
  }

  private async reject(idOrPrefix?: string): Promise<void> {
    const action = this.resolvePendingAction(idOrPrefix);
    await this.options.client.rejectAction(action.id);
    this.pendingActions.delete(action.id);
    this.options.io.write(`${successMark()} Action rejected.\n`);
  }

  private async resolveThread(idOrPrefix: string): Promise<AgentThreadDetail> {
    const id = idOrPrefix.trim();
    if (!id) throw new Error("Usage: /resume <conversation-id>");
    if (id.length >= 32) return await this.options.client.getThread(id);
    const matches = (await this.options.client.listThreads())
      .filter((thread) => thread.id.startsWith(id));
    if (matches.length === 0) throw new Error(`No conversation matches ${id}.`);
    if (matches.length > 1) {
      throw new Error(`Conversation prefix ${id} is ambiguous.`);
    }
    return await this.options.client.getThread(matches[0]!.id);
  }

  private async handleCommand(
    input: string,
    context?: AgentTurnContext,
  ): Promise<AgentLineAction> {
    const tokens = tokenizeShellInput(input);
    const command = tokens[0]!.slice(1).toLowerCase();
    const args = tokens.slice(1);

    switch (command) {
      case "help":
        this.options.io.write(helpText());
        return "continue";
      case "new":
        if (args.length > 0) throw new Error("/new does not accept arguments.");
        this.thread = null;
        this.pendingActions.clear();
        this.options.io.write(`${successMark()} New conversation ready.\n`);
        return "continue";
      case "threads": {
        if (args.length > 0) throw new Error("/threads does not accept arguments.");
        const threads = await this.options.client.listThreads();
        if (threads.length === 0) {
          this.options.io.write(chalk.dim("No conversations yet.\n"));
          return "continue";
        }
        this.options.io.write(
          `${threads.slice(0, 20).map((thread) => formatThread(thread, this.thread?.id)).join("\n")}\n`,
        );
        return "continue";
      }
      case "resume":
        if (args.length !== 1) throw new Error("Usage: /resume <conversation-id>");
        {
          const detail = await this.resolveThread(args[0]!);
          this.thread = detail.thread;
          this.pendingActions.clear();
          for (const action of detail.actions) {
            if (action.status === "proposed") {
              this.pendingActions.set(action.id, action);
            }
          }
        }
        this.options.io.write(
          `${successMark()} Resumed ${this.thread.title} (${this.thread.id.slice(0, 8)}).\n`,
        );
        return "continue";
      case "approve":
        if (args.length > 1) throw new Error("Usage: /approve [action-id]");
        await this.approve(args[0], context);
        return "continue";
      case "reject":
        if (args.length > 1) throw new Error("Usage: /reject [action-id]");
        await this.reject(args[0]);
        return "continue";
      case "status":
        if (args.length > 0) throw new Error("/status does not accept arguments.");
        this.options.io.write(
          this.thread
            ? `${formatThread(this.thread, this.thread.id)}\n`
            : chalk.dim("New conversation; send a message to create it.\n"),
        );
        this.options.io.write(
          chalk.dim(`${this.pendingActions.size} pending approval(s).\n`),
        );
        return "continue";
      case "clear":
        if (args.length > 0) throw new Error("/clear does not accept arguments.");
        this.options.io.clear();
        return "continue";
      case "exit":
        if (args.length > 0) throw new Error("/exit does not accept arguments.");
        return "exit";
      default:
        throw new Error(`Unknown agent command: /${command}. Use /help.`);
    }
  }

  async handleLine(input: string, context?: AgentTurnContext): Promise<AgentLineAction> {
    try {
      const normalized = input.trim();
      if (!normalized) return "continue";
      if (/^(exit|quit)$/i.test(normalized)) return "exit";
      if (normalized.startsWith("/")) return await this.handleCommand(normalized, context);
      await this.send(normalized, context);
      return "continue";
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.options.io.writeError(`${errorMark()} ${sanitizeTerminalText(message)}\n`);
      return "continue";
    }
  }
}
