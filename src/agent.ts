import chalk from "chalk";
import { mascotMark } from "./mascot.js";
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
import {
  sanitizeTerminalText,
  StreamingTerminalMarkdown,
  wrapTerminalLine,
} from "./terminal-markdown.js";

const accent = chalk.hex("#8B5CF6");
const successMark = (): string => chalk.green("✓");
const errorMark = (): string => chalk.red("✗");

export interface AgentSessionIO {
  write(text: string): void;
  writeError(text: string): void;
  clear(): void;
  /** Terminal width used to wrap answers; omit to leave wrapping to the terminal. */
  columns?(): number | undefined;
  /** True for an interactive terminal where status lines can be redrawn in place. */
  live?: boolean;
}

export type ApprovalDecision = "approve" | "reject" | "later";

/** Ask the user to decide on a proposal without leaving the conversation. */
export type ApprovalPrompter = (question: string) => Promise<ApprovalDecision>;

const ANSI_CLEAR_LINE = "\r\u001b[2K";
const GUTTER = "  ";

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

function helpText(): string {
  return [
    accent.bold("Agent commands"),
    "",
    `  ${accent("/new".padEnd(22))} Start a new conversation.`,
    `  ${accent("/threads".padEnd(22))} List recent conversations.`,
    `  ${accent("/resume <id>".padEnd(22))} Resume a conversation by ID or prefix.`,
    `  ${accent("/approve [id]".padEnd(22))} Approve an action left pending.`,
    `  ${accent("/reject [id]".padEnd(22))} Reject an action left pending.`,
    `  ${accent("/status".padEnd(22))} Show the active conversation.`,
    `  ${accent("/clear".padEnd(22))} Clear the terminal.`,
    `  ${accent("/exit".padEnd(22))} Exit the agent.`,
    "",
    chalk.dim(
      "Everything else is sent to the laptop-local Tuned Tensor assistant; known TT commands still run directly in the shell. Proposed changes ask for approval right after the answer.",
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

function riskLabel(risk: string): string {
  const safe = sanitizeTerminalText(risk);
  const color = safe === "low" ? chalk.green : safe === "high" ? chalk.red : chalk.yellow;
  return color(`${safe} risk`);
}

function diffStyle(line: string): (text: string) => string {
  if (line.startsWith("+++") || line.startsWith("---")) return chalk.dim;
  if (line.startsWith("+")) return chalk.green;
  if (line.startsWith("-")) return chalk.red;
  if (line.startsWith("@@")) return chalk.cyan;
  return (text) => text;
}

function approvalQuestion(action: AgentAction): string {
  switch (action.operation) {
    case "update_local_spec":
      return "Save this spec edit?";
    case "create_local_spec":
      return "Create this project?";
    case "run_local_pipeline":
      return "Run this dry-run preview?";
    case "create_spec":
    case "update_spec":
      return "Apply this change to your TT account?";
    default:
      return "Approve this action?";
  }
}

function completionMessage(action: AgentAction | null): string {
  switch (action?.operation) {
    case "update_local_spec":
      return "Spec edit saved.";
    case "create_local_spec":
      return "Project created.";
    case "run_local_pipeline":
      return "Dry-run preview finished.";
    default:
      return "Approved action completed.";
  }
}

function toolLabel(payload: Record<string, unknown>): string {
  const label = stringValue(payload.label);
  if (label) return sanitizeTerminalText(label);
  const name =
    stringValue(payload.name) ??
    stringValue(isRecord(payload.tool_call) ? payload.tool_call.name : null) ??
    "tool";
  const words = sanitizeTerminalText(name).replace(/_/g, " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

export class TunedTensorAgentSession {
  private thread: AgentThread | null;
  private readonly pendingActions = new Map<string, AgentAction>();
  private activeRequest: { controller: AbortController; interruptible: boolean } | null = null;
  private prompter: ApprovalPrompter | null = null;
  /** Proposals raised during the current turn, reviewed once the answer ends. */
  private turnProposals: string[] = [];
  private runningAction: AgentAction | null = null;
  /** The next answer line starts a new block and carries the ● marker. */
  private blockStart = true;
  private wroteInTurn = false;
  private lineOpen = false;
  private reasoningActive = false;
  /** A live status line is on screen without a trailing newline. */
  private transient: string | null = null;
  private readonly pendingToolLabels = new Map<string, string>();
  private pendingUnkeyedToolLabel: string | null = null;
  private readonly responseMarkdown = new StreamingTerminalMarkdown();

  constructor(private readonly options: AgentSessionOptions) {
    this.thread = options.thread ?? null;
  }

  get busy(): boolean {
    return this.activeRequest !== null;
  }

  /**
   * Enable inline approval. When set, tt asks for a decision right after an
   * answer that proposes a change instead of waiting for /approve.
   */
  setApprovalPrompter(prompter: ApprovalPrompter | null): void {
    this.prompter = prompter;
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
      this.writeErrorLine("Approved action is settling and cannot be interrupted safely. Do not retry it.");
      return true;
    }
    this.activeRequest.controller.abort("user-stop");
    return true;
  }

  private get io(): AgentSessionIO {
    return this.options.io;
  }

  private clearTransient(): void {
    if (this.transient === null) return;
    this.io.write(ANSI_CLEAR_LINE);
    this.transient = null;
  }

  private endOpenLine(): void {
    this.clearTransient();
    if (this.lineOpen) this.io.write("\n");
    this.lineOpen = false;
  }

  /** Write a status line in the answer gutter, e.g. "○ Thinking…". */
  private writeStatus(text: string, options: { transient?: boolean } = {}): void {
    this.flushResponse();
    this.endOpenLine();
    if (!this.blockStart) this.io.write("\n");
    this.blockStart = true;
    this.wroteInTurn = true;
    const line = `${GUTTER}${text}`;
    if (options.transient && this.io.live) {
      this.io.write(line);
      this.transient = line;
      return;
    }
    this.io.write(`${line}\n`);
  }

  private writeErrorLine(message: string): void {
    this.endOpenLine();
    this.io.writeError(`${GUTTER}${errorMark()} ${sanitizeTerminalText(message)}\n`);
  }

  private wrapWidth(): number | undefined {
    const columns = this.io.columns?.();
    return columns ? columns - GUTTER.length : undefined;
  }

  /** Write rendered answer text in the gutter, marking the start of each block. */
  private writeAnswer(rendered: string): void {
    if (!rendered) return;
    this.clearTransient();
    const width = this.wrapWidth();
    const lines = rendered.split("\n");
    lines.forEach((line, index) => {
      if (index > 0) {
        this.io.write("\n");
        this.lineOpen = false;
      }
      if (!line) return;
      const wrapped = width ? wrapTerminalLine(line, width) : [line];
      wrapped.forEach((part, partIndex) => {
        if (partIndex > 0) this.io.write("\n");
        if (!this.lineOpen || partIndex > 0) {
          const marker = this.blockStart ? `${accent("●")} ` : GUTTER;
          // Separate a new answer block from status lines above it.
          if (this.blockStart && this.wroteInTurn) this.io.write("\n");
          this.blockStart = false;
          this.wroteInTurn = true;
          this.io.write(marker);
        }
        this.io.write(part);
      });
      this.lineOpen = true;
    });
  }

  private flushResponse(): void {
    this.writeAnswer(this.responseMarkdown.flush());
  }

  private renderAction(action: AgentAction): void {
    this.flushResponse();
    this.endOpenLine();
    const bar = chalk.yellow("│");
    const out: string[] = [
      "",
      `${GUTTER}${chalk.yellow.bold("◆ Approval needed")}  ${chalk.dim(action.id.slice(0, 8))}`,
      `${GUTTER}${bar} ${chalk.bold(sanitizeTerminalText(action.title))} ${chalk.dim("·")} ${riskLabel(action.risk)}`,
    ];
    const width = this.wrapWidth();
    const push = (text: string, style: (line: string) => string = (line) => line) => {
      for (const raw of sanitizeTerminalText(text).split("\n")) {
        const parts = width ? wrapTerminalLine(raw, width - 2) : [raw];
        for (const part of parts) out.push(`${GUTTER}${bar} ${style(part)}`);
      }
    };
    if (action.summary) push(action.summary, chalk.dim);
    if (action.operation === "update_local_spec") {
      const preview = action.preview as { diff?: string; validation?: { warnings?: string[] } } | undefined;
      out.push(`${GUTTER}${bar}`);
      for (const line of sanitizeTerminalText(preview?.diff ?? "No diff available.").replace(/\n$/, "").split("\n")) {
        // Continuation rows keep the line's diff color and sit under its text.
        const style = diffStyle(line);
        const parts = width ? wrapTerminalLine(line, width - 4) : [line];
        parts.forEach((part, index) => {
          out.push(`${GUTTER}${bar} ${style(index === 0 ? part : `  ${part}`)}`);
        });
      }
      for (const warning of preview?.validation?.warnings ?? []) push(`! ${warning}`, chalk.yellow);
    } else {
      const request = [action.method, action.path].filter(Boolean).join(" ");
      const technical = {
        operation: action.operation,
        ...(request ? { request } : {}),
        arguments: action.arguments ?? null,
        preview: action.preview ?? null,
      };
      out.push(`${GUTTER}${bar}`);
      push(formatJson(technical), chalk.dim);
    }
    this.io.write(`${out.join("\n")}\n`);
  }

  private writePendingHint(): void {
    this.io.write(
      chalk.dim(`${GUTTER}Nothing changes until you decide: /approve applies it, /reject discards it.\n`),
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
      this.reasoningActive = false;
      this.writeAnswer(this.responseMarkdown.push(delta));
      return;
    }

    if (event.type === "reasoning_delta") {
      const delta =
        stringValue(payload.delta) ??
        stringValue(payload.text) ??
        stringValue(payload.content);
      if (!delta) return;
      // Raw reasoning is verbose and rarely useful to the user; show one
      // compact indicator per reasoning block instead of streaming it.
      if (this.reasoningActive) return;
      this.writeStatus(`${mascotMark()} ${chalk.dim("Thinking…")}`, { transient: true });
      this.reasoningActive = true;
      return;
    }

    if (event.type === "tool_call") {
      this.reasoningActive = false;
      const label = toolLabel(payload);
      this.writeStatus(chalk.dim(`○ ${label}…`), { transient: true });
      const id = stringValue(payload.toolUseId);
      if (id) this.pendingToolLabels.set(id, label);
      else this.pendingUnkeyedToolLabel = label;
      return;
    }

    if (event.type === "tool_result") {
      const failed = payload.status === "error" || Boolean(payload.error);
      const id = stringValue(payload.toolUseId);
      const label = (id && this.pendingToolLabels.get(id)) ?? this.pendingUnkeyedToolLabel ?? "Tool";
      if (id) this.pendingToolLabels.delete(id);
      else this.pendingUnkeyedToolLabel = null;
      this.writeStatus(
        `${failed ? errorMark() : successMark()} ${chalk.dim(failed ? `${label} failed` : label)}`,
      );
      return;
    }

    if (event.type === "approval_required") {
      const action = actionFromPayload(payload);
      if (!action) return;
      this.pendingActions.set(action.id, action);
      if (!this.turnProposals.includes(action.id)) this.turnProposals.push(action.id);
      return;
    }

    if (event.type === "action_started") {
      // Kept on screen: an approved pipeline preview streams child output below it.
      this.writeStatus(chalk.dim("○ Running approved action…"));
      return;
    }

    if (event.type === "action_result") {
      if (payload.status === "outcome_unknown") {
        this.writeErrorLine(
          "Approved action outcome is unknown. It cannot be retried; inspect the remote spec before preparing another action.",
        );
        return;
      }
      const failed = payload.status === "failed" || Boolean(payload.error);
      if (failed) {
        this.writeErrorLine("Approved action failed.");
        return;
      }
      this.writeStatus(`${successMark()} ${completionMessage(this.runningAction)}`);
      return;
    }

    if (event.type === "error") {
      this.flushResponse();
      const message =
        stringValue(payload.message) ??
        stringValue(payload.error) ??
        "The agent could not complete this request.";
      this.writeErrorLine(message);
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
      this.writeErrorLine("A response is already running.");
      return null;
    }
    const controller = new AbortController();
    this.activeRequest = { controller, interruptible };
    this.resetRenderState();
    try {
      const result = await operation(controller.signal);
      this.flushResponse();
      this.endOpenLine();
      for (const action of result.actions) {
        // Streamed proposals carry the full preview; keep that copy.
        if (!this.pendingActions.has(action.id)) this.pendingActions.set(action.id, action);
        if (action.status !== "executing" && !this.turnProposals.includes(action.id)) {
          this.turnProposals.push(action.id);
        }
      }
      if (controller.signal.aborted || result.status === "cancelled") {
        this.io.write(chalk.dim(`${GUTTER}Response stopped.\n`));
        return null;
      }
      return result;
    } catch (error) {
      this.flushResponse();
      this.endOpenLine();
      if (controller.signal.aborted) {
        this.io.write(chalk.dim(`${GUTTER}Response stopped.\n`));
        return null;
      }
      throw error;
    } finally {
      this.activeRequest = null;
      this.resetRenderState();
    }
  }

  private resetRenderState(): void {
    this.blockStart = true;
    this.wroteInTurn = false;
    this.reasoningActive = false;
    this.lineOpen = false;
    this.transient = null;
    this.pendingToolLabels.clear();
    this.pendingUnkeyedToolLabel = null;
    this.responseMarkdown.reset();
  }

  async send(prompt: string, context?: AgentTurnContext): Promise<AgentTurnResult | null> {
    const normalized = prompt.trim();
    if (!normalized) return null;
    const thread = await this.ensureThread();
    this.turnProposals = [];
    // Leave a gap between the submitted question and the answer.
    this.io.write("\n");
    let result: AgentTurnResult | null = null;
    let failure: { error: unknown } | null = null;
    try {
      result = await this.runStream(async (signal) => {
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
    } catch (error) {
      failure = { error };
    }
    // Show proposals after the answer so its explanation reads first.
    const proposals = this.takeTurnProposals();
    for (const action of proposals) this.renderAction(action);
    if (failure) {
      if (proposals.length > 0) this.writePendingHint();
      throw failure.error;
    }
    await this.reviewProposals(proposals, result !== null, context);
    return result;
  }

  private takeTurnProposals(): AgentAction[] {
    const actions = this.turnProposals
      .map((id) => this.pendingActions.get(id))
      .filter((action): action is AgentAction => action !== undefined);
    this.turnProposals = [];
    return actions;
  }

  /**
   * Ask for a decision on each new proposal. Approval stays a deterministic
   * user action: the model never answers this prompt.
   */
  private async reviewProposals(
    actions: AgentAction[],
    canPrompt: boolean,
    context?: AgentTurnContext,
  ): Promise<void> {
    if (actions.length === 0) return;
    if (!this.prompter || !canPrompt) {
      this.writePendingHint();
      return;
    }
    for (const action of actions) {
      if (!this.pendingActions.has(action.id)) continue;
      const label = actions.length > 1
        ? `${approvalQuestion(action)} ${chalk.dim(`(${action.title})`)}`
        : approvalQuestion(action);
      const decision = await this.prompter(`${GUTTER}${chalk.yellow("?")} ${chalk.bold(label)}`);
      try {
        if (decision === "approve") await this.approve(action.id, context);
        else if (decision === "reject") await this.reject(action.id);
        else {
          this.io.write(chalk.dim(`${GUTTER}Left pending. Use /approve or /reject when you're ready.\n`));
        }
      } catch (error) {
        this.writeErrorLine(error instanceof Error ? error.message : String(error));
      }
    }
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
    this.runningAction = action;
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
      this.runningAction = null;
    }
  }

  private async reject(idOrPrefix?: string): Promise<void> {
    const action = this.resolvePendingAction(idOrPrefix);
    await this.options.client.rejectAction(action.id);
    this.pendingActions.delete(action.id);
    this.io.write(`${GUTTER}${successMark()} Rejected. Nothing was changed.\n`);
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
      this.writeErrorLine(error instanceof Error ? error.message : String(error));
      return "continue";
    }
  }
}
