import chalk from "chalk";
import { stat } from "node:fs/promises";
import { homedir } from "node:os";
import { isAbsolute, resolve } from "node:path";
import { createInterface, type Interface } from "node:readline";
import { stripVTControlCharacters } from "node:util";
import {
  COMMAND_CATALOG,
  SLASH_COMMANDS,
  createCommandCompleter,
  groupedCatalog,
  type WorkflowMode,
} from "./command-catalog.js";
import {
  discoverShellContext,
  formatShellContext,
  formatShellStatus,
  type ShellContext,
  type TargetSource,
} from "./shell-context.js";

export type { WorkflowMode } from "./command-catalog.js";

export interface ShellAgent {
  busy: boolean;
  handleLine(
    input: string,
    context?: { mode: WorkflowMode; workspaceRoot: string },
  ): Promise<"continue" | "exit">;
  interrupt(): boolean;
}

const AGENT_SLASH_COMMANDS = new Set([
  "new",
  "threads",
  "resume",
  "approve",
  "reject",
]);

export class ShellParseError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ShellParseError";
  }
}

type QuoteMode = "none" | "single" | "double";

function operatorError(operator: string): ShellParseError {
  return new ShellParseError(
    `Shell operator ${JSON.stringify(operator)} is not supported. Run TT commands directly; pipes, redirects, and command substitution are disabled.`,
  );
}

/**
 * Parse a command line into argv without invoking a system shell.
 *
 * Quoted or backslash-escaped punctuation is treated as literal data. Shell
 * operators outside quotes are rejected rather than interpreted.
 */
export function tokenizeShellInput(input: string): string[] {
  if (input.includes("\0")) {
    throw new ShellParseError("Command input must not contain NUL bytes.");
  }
  if (/[\r\n]/.test(input)) {
    throw new ShellParseError("Enter one command at a time.");
  }
  if (input.trimStart().startsWith("!")) {
    throw new ShellParseError("Shell escapes are disabled; enter a TT command instead.");
  }

  const tokens: string[] = [];
  let token = "";
  let tokenStarted = false;
  let quote: QuoteMode = "none";

  const finishToken = () => {
    if (!tokenStarted) return;
    tokens.push(token);
    token = "";
    tokenStarted = false;
  };

  for (let index = 0; index < input.length; index += 1) {
    const character = input[index]!;

    if (quote === "single") {
      if (character === "'") {
        quote = "none";
      } else {
        token += character;
      }
      continue;
    }

    if (quote === "double") {
      if (character === "\"") {
        quote = "none";
        continue;
      }
      if (character === "\\") {
        const escaped = input[index + 1];
        if (escaped === undefined) {
          throw new ShellParseError("Command ends with an unfinished escape.");
        }
        token += escaped;
        tokenStarted = true;
        index += 1;
        continue;
      }
      if (character === "`") throw operatorError("`…`");
      if (character === "$" && input[index + 1] === "(") {
        throw operatorError("$(…)");
      }
      token += character;
      tokenStarted = true;
      continue;
    }

    if (/\s/.test(character)) {
      finishToken();
      continue;
    }
    if (character === "'") {
      quote = "single";
      tokenStarted = true;
      continue;
    }
    if (character === "\"") {
      quote = "double";
      tokenStarted = true;
      continue;
    }
    if (character === "\\") {
      const escaped = input[index + 1];
      if (escaped === undefined) {
        throw new ShellParseError("Command ends with an unfinished escape.");
      }
      token += escaped;
      tokenStarted = true;
      index += 1;
      continue;
    }
    if (character === "`") throw operatorError("`…`");
    if (character === "$" && input[index + 1] === "(") {
      throw operatorError("$(…)");
    }
    if (character === "|" || character === ">" || character === "<" || character === ";" || character === "&") {
      const doubled = input[index + 1] === character
        && (character === "|" || character === "&");
      throw operatorError(doubled ? character.repeat(2) : character);
    }

    token += character;
    tokenStarted = true;
  }

  if (quote !== "none") {
    throw new ShellParseError(
      `Command has an unterminated ${quote === "single" ? "single" : "double"} quote.`,
    );
  }
  finishToken();
  return tokens;
}

export interface RoutedShellCommand {
  target: WorkflowMode;
  args: string[];
}

export function routeShellCommand(
  input: string,
  activeMode: WorkflowMode,
): RoutedShellCommand | null {
  let args = tokenizeShellInput(input);
  if (args.length === 0) return null;
  if (args[0] === "tt") {
    args = args.slice(1);
    if (args.length === 0) {
      throw new ShellParseError("The TT shell is already open; enter a command such as `runs list`.");
    }
  }
  if (args[0]?.startsWith("/")) {
    throw new ShellParseError("Slash commands are session commands and cannot be routed to a workflow.");
  }

  const override = args[0];
  if (override === "cloud" || override === "local") {
    if (args.length === 1) {
      throw new ShellParseError(`Add a ${override} command, or use \`/mode ${override}\` to switch workflows.`);
    }
    return { target: override, args: args.slice(1) };
  }
  return { target: activeMode, args };
}

/**
 * Decide whether a line is an intentional CLI command. Exact catalog prefixes
 * preserve the existing shell grammar; everything else can be conversational.
 */
export function isCatalogCommand(
  mode: WorkflowMode,
  args: readonly string[],
): boolean {
  if (args.length === 0) return false;
  if (args[0] === "status") {
    return args.length === 1 || args[1]?.startsWith("-") === true;
  }
  return COMMAND_CATALOG.some((command) => {
    if (!command.modes.includes(mode)) return false;
    const path = command.path.split(" ");
    return path.every((token, index) => args[index] === token);
  });
}

function hasCatalogIntent(
  input: string,
  activeMode: WorkflowMode,
): boolean {
  let args = input.trim().split(/\s+/);
  if (args[0] === "tt") args = args.slice(1);
  const override = args[0];
  const mode =
    override === "cloud" || override === "local"
      ? override
      : activeMode;
  if (override === "cloud" || override === "local") args = args.slice(1);
  return isCatalogCommand(mode, args);
}

export type SlashCommandName =
  | "palette"
  | "help"
  | "status"
  | "context"
  | "mode"
  | "cloud"
  | "local"
  | "model"
  | "clear"
  | "cd"
  | "exit";

export interface ParsedSlashCommand {
  name: SlashCommandName;
  args: string[];
}

const SLASH_NAMES = new Set([
  "help",
  "status",
  "context",
  "mode",
  "cloud",
  "local",
  "model",
  "clear",
  "cd",
  "exit",
]);

/** Workflow command roots (runs, models, …) people commonly mistype with a slash. */
const WORKFLOW_ROOT_COMMANDS = new Set(
  COMMAND_CATALOG.map((command) => command.path.split(" ", 1)[0]!),
);

function editDistance(left: string, right: string): number {
  const previous = Array.from({ length: right.length + 1 }, (_, index) => index);
  for (let i = 1; i <= left.length; i += 1) {
    let diagonal = previous[0]!;
    previous[0] = i;
    for (let j = 1; j <= right.length; j += 1) {
      const above = previous[j]!;
      previous[j] = Math.min(
        previous[j]! + 1,
        previous[j - 1]! + 1,
        diagonal + (left[i - 1] === right[j - 1] ? 0 : 1),
      );
      diagonal = above;
    }
  }
  return previous[right.length]!;
}

function suggestSlashCommands(rawName: string): string[] {
  const names = [...SLASH_NAMES].sort();
  const prefixMatches = names.filter((name) => name.startsWith(rawName));
  if (prefixMatches.length > 0) return prefixMatches;
  return names.filter((name) => editDistance(name, rawName) <= 2);
}

function unknownSlashCommandError(rawName: string): ShellParseError {
  if (WORKFLOW_ROOT_COMMANDS.has(rawName)) {
    return new ShellParseError(
      `Unknown session command: /${rawName}. Workflow commands need no slash — try "${rawName} --help".`,
    );
  }
  const suggestions = suggestSlashCommands(rawName);
  const hint = suggestions.length > 0
    ? ` Did you mean ${suggestions.map((name) => `/${name}`).join(" or ")}?`
    : "";
  return new ShellParseError(
    `Unknown session command: /${rawName}.${hint} Use /help to see available commands.`,
  );
}

export function parseSlashCommand(input: string): ParsedSlashCommand | null {
  const trimmed = input.trim();
  if (!trimmed) return null;
  if (trimmed === "?") return { name: "help", args: [] };
  if (trimmed.startsWith("? ")) {
    return {
      name: "help",
      args: tokenizeShellInput(trimmed.slice(1)),
    };
  }
  if (!trimmed.startsWith("/")) return null;
  if (trimmed === "/") return { name: "palette", args: [] };

  const tokens = tokenizeShellInput(trimmed);
  const rawName = tokens[0]!.slice(1).toLowerCase();
  if (!SLASH_NAMES.has(rawName)) {
    throw unknownSlashCommandError(rawName);
  }
  return {
    name: rawName as Exclude<SlashCommandName, "palette">,
    args: tokens.slice(1),
  };
}

export interface ShellCommandRequest {
  target: WorkflowMode;
  args: string[];
  cwd: string;
}

export interface ShellCommandResult {
  exitCode: number | null;
}

export type ShellCommandRunner = (
  request: ShellCommandRequest,
) => Promise<ShellCommandResult | void>;

export interface ShellSessionIO {
  write(text: string): void;
  writeError(text: string): void;
  clear(): void;
}

export type ShellContextProvider = (
  options: { cwd: string; env: Readonly<NodeJS.ProcessEnv> },
) => Promise<ShellContext>;

export interface CreateShellSessionOptions {
  runner: ShellCommandRunner;
  io: ShellSessionIO;
  agent?: ShellAgent;
  cwd?: string;
  env?: Readonly<NodeJS.ProcessEnv>;
  contextProvider?: ShellContextProvider;
  version?: string;
}

export interface ShellSessionSnapshot {
  mode: WorkflowMode;
  modeSource: TargetSource | "session";
  cwd: string;
  context: ShellContext;
  version?: string;
}

export type ShellLineAction = "continue" | "exit";

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/* Shell messages reuse the main app's output style: ✓/✗/! marks, bold
 * headings and labels (see output.ts). Shell chrome stays deliberately quiet
 * so the conversation remains the strongest element on screen. */
const successMark = (): string => chalk.green("✓");
const errorMark = (): string => chalk.red("✗");
const accent = chalk.hex("#8B5CF6");
const ANSI_BOLD = "\u001b[1m";
const ANSI_BOLD_OFF = "\u001b[22m";
const ANSI_CLEAR_LINE = "\u001b[2K";
const ANSI_CLEAR_TO_END = "\u001b[K";
const ANSI_CURSOR_UP = "\u001b[1A";
const ANSI_RESET = "\u001b[0m";

function inputColorCodes(): {
  background: string;
  accent: string;
  text: string;
} | null {
  if (chalk.level === 0) return null;
  if (chalk.level === 1) {
    return {
      background: "\u001b[100m",
      accent: "\u001b[95m",
      text: "\u001b[97m",
    };
  }
  if (chalk.level === 2) {
    return {
      background: "\u001b[48;5;238m",
      accent: "\u001b[38;5;183m",
      text: "\u001b[38;5;255m",
    };
  }
  return {
    background: "\u001b[48;2;50;52;67m",
    accent: "\u001b[38;2;196;181;253m",
    text: "\u001b[38;2;245;243;255m",
  };
}

/** Label width used by formatShellStatus/formatShellContext. */
const DETAIL_LABEL_WIDTH = 15;

function styleDetailLines(lines: string[]): string[] {
  return lines.map((line) => {
    if (line.length <= DETAIL_LABEL_WIDTH) return line;
    const label = line.slice(0, DETAIL_LABEL_WIDTH);
    if (!label.trim()) return line;
    const color = label.startsWith("Warning") ? chalk.yellow : accent.bold;
    return `${color(label)}${line.slice(DETAIL_LABEL_WIDTH)}`;
  });
}

function detailLine(label: string, value: string): string {
  return `${label.padEnd(DETAIL_LABEL_WIDTH)}${value}`;
}

function assertNoArgs(command: string, args: string[]): void {
  if (args.length > 0) {
    throw new ShellParseError(`/${command} does not accept arguments.`);
  }
}

function expandDirectory(
  value: string,
  cwd: string,
  env: Readonly<NodeJS.ProcessEnv>,
): string {
  const home = env.HOME ? resolve(env.HOME) : homedir();
  if (value === "~") return home;
  if (value.startsWith("~/")) return resolve(home, value.slice(2));
  return isAbsolute(value) ? resolve(value) : resolve(cwd, value);
}

function helpText(mode: WorkflowMode, query?: string, palette = false): string {
  const groups = groupedCatalog(mode, query);
  const lines: string[] = [];
  lines.push(accent.bold(palette
    ? `Commands for ${mode} — type a command or use cloud/local as a one-shot prefix`
    : `TT ${mode} commands${query ? ` matching ${JSON.stringify(query)}` : ""}`));

  if (groups.size === 0) {
    lines.push(chalk.dim("  No matching commands."));
  } else {
    for (const [group, commands] of groups) {
      lines.push("");
      lines.push(chalk.bold(group));
      for (const command of commands) {
        lines.push(`  ${accent(command.path.padEnd(22))} ${command.description}`);
      }
    }
  }

  if (!query) {
    lines.push("");
    lines.push(chalk.bold("Conversation"));
    lines.push(`  ${accent("plain text".padEnd(22))} Ask the Tuned Tensor agent anything.`);
    lines.push(`  ${accent("/new".padEnd(22))} Start a new agent conversation.`);
    lines.push(`  ${accent("/threads".padEnd(22))} List recent conversations.`);
    lines.push(`  ${accent("/resume <id>".padEnd(22))} Resume a conversation.`);
    lines.push(`  ${accent("/approve [id]".padEnd(22))} Approve a proposed action.`);
    lines.push(`  ${accent("/reject [id]".padEnd(22))} Reject a proposed action.`);
    lines.push("");
    lines.push(chalk.bold("Session"));
    for (const command of SLASH_COMMANDS) {
      lines.push(`  ${accent(command.path.padEnd(22))} ${command.description}`);
    }
    lines.push(`  ${accent("?".padEnd(22))} Alias for /help.`);
    lines.push(chalk.dim(
      "\nKnown TT commands execute directly; other text goes to the agent. Prefix a command with : to make the intent explicit.",
    ));
  }
  return `${lines.join("\n")}\n`;
}

function activeModelLabel(snapshot: ShellSessionSnapshot): string {
  return snapshot.mode === "local"
    ? snapshot.context.local.activeModelId ?? "base"
    : snapshot.context.spec?.baseModel ?? "—";
}

function agentModelLabel(context: ShellContext): string {
  const agent = context.agent;
  if (!agent?.provider && !agent?.model) return "not configured";
  return [agent.provider, agent.model].filter(Boolean).join("/") || "not configured";
}

export function renderShellBanner(snapshot: ShellSessionSnapshot): string {
  const spec = snapshot.context.spec?.name
    ?? snapshot.context.spec?.path
    ?? "no spec";
  const heading = snapshot.version
    ? `${accent.bold("tt")} ${chalk.dim(`v${snapshot.version}`)}`
    : accent.bold("tt");
  const lines = [
    heading,
    `${accent(snapshot.mode)}${chalk.dim(
      ` · ${snapshot.context.projectName} · ${spec}`,
    )}`,
    chalk.dim(
      `agent ${agentModelLabel(snapshot.context)} · workflow model ${activeModelLabel(snapshot)}`,
    ),
    chalk.dim("ctrl+c stop/clear · ctrl+d exit · /help commands · tab complete"),
    "",
    chalk.dim("Ask TT anything. Known commands run directly."),
  ];
  return `${lines.join("\n")}\n\n`;
}

export function renderShellPrompt(): string {
  // Readline must own a self-contained prompt. Leaving a background style open
  // lets prompt redraws leak into streamed agent output.
  return `${accent.bold("›")} `;
}

export function renderSubmittedShellInput(input: string, columns?: number): string {
  const colors = inputColorCodes();
  if (!colors) return "";
  const safeInput = stripVTControlCharacters(input)
    .replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f]/g, "");
  if (columns && Array.from(`› ${safeInput}`).length >= columns) return "";
  return [
    ANSI_CURSOR_UP,
    "\r",
    ANSI_CLEAR_LINE,
    colors.background,
    colors.accent,
    ANSI_BOLD,
    "›",
    ANSI_BOLD_OFF,
    colors.text,
    " ",
    safeInput,
    ANSI_CLEAR_TO_END,
    ANSI_RESET,
    "\r\n",
  ].join("");
}

export function resetShellPromptStyle(): string {
  return "";
}

export class TunedTensorShellSession {
  private mode: WorkflowMode;
  private modeSource: TargetSource | "session";
  private cwd: string;
  private context: ShellContext;

  private constructor(
    private readonly runner: ShellCommandRunner,
    private readonly io: ShellSessionIO,
    private readonly agent: ShellAgent | undefined,
    private readonly env: Readonly<NodeJS.ProcessEnv>,
    private readonly contextProvider: ShellContextProvider,
    private readonly version: string | undefined,
    initialContext: ShellContext,
  ) {
    this.mode = initialContext.inferredTarget;
    this.modeSource = initialContext.targetSource;
    this.cwd = initialContext.cwd;
    this.context = initialContext;
  }

  static async create(
    options: CreateShellSessionOptions,
  ): Promise<TunedTensorShellSession> {
    const env = options.env ?? process.env;
    const cwd = resolve(options.cwd ?? process.cwd());
    const contextProvider = options.contextProvider
      ?? ((input) => discoverShellContext(input));
    const initialContext = await contextProvider({ cwd, env });
    return new TunedTensorShellSession(
      options.runner,
      options.io,
      options.agent,
      env,
      contextProvider,
      options.version,
      initialContext,
    );
  }

  interruptAgent(): boolean {
    return this.agent?.interrupt() ?? false;
  }

  snapshot(): ShellSessionSnapshot {
    return {
      mode: this.mode,
      modeSource: this.modeSource,
      cwd: this.cwd,
      context: this.context,
      version: this.version,
    };
  }

  prompt(): string {
    return renderShellPrompt();
  }

  banner(): string {
    return renderShellBanner(this.snapshot());
  }

  private async refreshContext(): Promise<void> {
    this.context = await this.contextProvider({ cwd: this.cwd, env: this.env });
    this.cwd = this.context.cwd;
    if (this.modeSource !== "session" && this.modeSource !== "environment") {
      this.mode = this.context.inferredTarget;
      this.modeSource = this.context.targetSource;
    }
  }

  private writeLines(lines: string[]): void {
    this.io.write(`${lines.join("\n")}\n`);
  }

  private modelLines(): string[] {
    if (this.mode === "local") {
      return styleDetailLines([
        detailLine("Active model", this.context.local.activeModelId ?? "base"),
        detailLine("Change", "/model <id> to activate a verified local model"),
      ]);
    }
    return styleDetailLines([
      detailLine("Base model", this.context.spec?.baseModel ?? "—"),
      detailLine("Change", "edit base_model in tunedtensor.json, then run push"),
    ]);
  }

  private async handleSlash(command: ParsedSlashCommand): Promise<ShellLineAction> {
    switch (command.name) {
      case "palette":
        this.io.write(helpText(this.mode, undefined, true));
        return "continue";
      case "help":
        this.io.write(helpText(this.mode, command.args.join(" ") || undefined));
        return "continue";
      case "status":
        assertNoArgs("status", command.args);
        await this.refreshContext();
        this.writeLines(styleDetailLines(formatShellStatus(this.context, this.mode)));
        return "continue";
      case "context":
        assertNoArgs("context", command.args);
        await this.refreshContext();
        this.writeLines(styleDetailLines(
          formatShellContext(this.context, this.mode, this.modeSource),
        ));
        return "continue";
      case "mode": {
        if (command.args.length === 0) {
          this.io.write(`Current workflow: ${this.mode}\n`);
          return "continue";
        }
        if (command.args.length !== 1 || !["cloud", "local"].includes(command.args[0]!)) {
          throw new ShellParseError("Usage: /mode cloud|local");
        }
        this.mode = command.args[0] as WorkflowMode;
        this.modeSource = "session";
        this.io.write(`${successMark()} Workflow switched to ${this.mode}.\n`);
        return "continue";
      }
      case "cloud":
      case "local":
        assertNoArgs(command.name, command.args);
        this.mode = command.name;
        this.modeSource = "session";
        this.io.write(`${successMark()} Workflow switched to ${this.mode}.\n`);
        return "continue";
      case "model": {
        if (command.args.length === 0) {
          await this.refreshContext();
          this.writeLines(this.modelLines());
          return "continue";
        }
        if (command.args.length !== 1) {
          throw new ShellParseError("Usage: /model [model-id]");
        }
        if (this.mode !== "local") {
          throw new ShellParseError(
            "Activating models is a local workflow action. Use /mode local first; cloud specs change base_model in tunedtensor.json.",
          );
        }
        await this.runner({
          target: "local",
          args: ["models", "activate", command.args[0]!],
          cwd: this.cwd,
        });
        await this.refreshContext();
        return "continue";
      }
      case "clear":
        assertNoArgs("clear", command.args);
        this.io.clear();
        return "continue";
      case "cd": {
        if (command.args.length === 0) {
          this.io.write(`${this.cwd}\n`);
          return "continue";
        }
        if (command.args.length !== 1) {
          throw new ShellParseError("Usage: /cd <directory>; quote paths containing spaces.");
        }
        const nextDirectory = expandDirectory(command.args[0]!, this.cwd, this.env);
        const metadata = await stat(nextDirectory).catch(() => null);
        if (!metadata?.isDirectory()) {
          throw new ShellParseError(`Not a directory: ${nextDirectory}`);
        }
        this.cwd = nextDirectory;
        await this.refreshContext();
        this.io.write(`${successMark()} Directory: ${this.cwd}\n`);
        return "continue";
      }
      case "exit":
        assertNoArgs("exit", command.args);
        return "exit";
    }
  }

  async handleLine(input: string): Promise<ShellLineAction> {
    try {
      // Bare exit/quit leaves the shell instead of erroring in a workflow.
      if (/^(exit|quit)$/i.test(input.trim())) return "exit";

      const trimmed = input.trim();
      if (trimmed.startsWith("/")) {
        const rawName = trimmed.slice(1).split(/\s+/, 1)[0]!.toLowerCase();
        if (AGENT_SLASH_COMMANDS.has(rawName)) {
          if (!this.agent) {
            throw new ShellParseError("The conversational agent is unavailable.");
          }
          await this.agent.handleLine(trimmed, {
            mode: this.mode,
            workspaceRoot: this.cwd,
          });
          await this.refreshContext();
          return "continue";
        }
      }

      const slash = parseSlashCommand(input);
      if (slash) return await this.handleSlash(slash);

      const explicitCommand = trimmed.startsWith(":")
        ? trimmed.slice(1).trimStart()
        : null;
      if (explicitCommand !== null && !explicitCommand) {
        throw new ShellParseError("Add a TT command after :.");
      }
      const explicitWorkflowPrefix =
        /^(?:tt\s+)?(?:cloud|local)(?:\s|$)/.test(trimmed)
        || /^tt(?:\s|$)/.test(trimmed);

      if (
        explicitCommand === null
        && !explicitWorkflowPrefix
        && this.agent
        && !hasCatalogIntent(input, this.mode)
      ) {
        await this.agent.handleLine(input, {
          mode: this.mode,
          workspaceRoot: this.cwd,
        });
        await this.refreshContext();
        return "continue";
      }

      const commandInput = explicitCommand ?? input;
      const routed = routeShellCommand(commandInput, this.mode);
      if (!routed) return "continue";
      await this.runner({
        target: routed.target,
        args: [...routed.args],
        cwd: this.cwd,
      });
      await this.refreshContext();
      return "continue";
    } catch (error) {
      this.io.writeError(`${errorMark()} ${errorMessage(error)}\n`);
      return "continue";
    }
  }
}

export async function createShellSession(
  options: CreateShellSessionOptions,
): Promise<TunedTensorShellSession> {
  return TunedTensorShellSession.create(options);
}

export interface InteractiveShellOptions {
  runner: ShellCommandRunner;
  agent?: ShellAgent;
  input?: NodeJS.ReadableStream;
  output?: NodeJS.WritableStream;
  error?: NodeJS.WritableStream;
  cwd?: string;
  env?: Readonly<NodeJS.ProcessEnv>;
  requireTTY?: boolean;
  version?: string;
}

function streamIsTTY(stream: NodeJS.ReadableStream | NodeJS.WritableStream): boolean {
  return Boolean((stream as { isTTY?: boolean }).isTTY);
}

interface RawModeInput extends NodeJS.ReadableStream {
  isRaw?: boolean;
  setRawMode?(mode: boolean): this;
}

export interface SigintSource {
  on(event: "SIGINT", listener: () => void): unknown;
  off(event: "SIGINT", listener: () => void): unknown;
}

export interface ForegroundSignalHandoff {
  input: RawModeInput;
  pauseReadline(): void;
  resumeReadline(): void;
  signals?: SigintSource;
}

/**
 * Give a foreground child normal terminal signal semantics while keeping the
 * parent shell alive long enough to regain the prompt.
 *
 * Readline holds the terminal in raw mode even while paused. Temporarily
 * restoring cooked mode lets the terminal send SIGINT to the whole foreground
 * process group, including the child. The parent's temporary listener prevents
 * Node's default SIGINT exit; it deliberately does not forward or consume a
 * byte, because the child receives the same OS signal directly.
 */
export async function withForegroundSignalHandoff<T>(
  control: ForegroundSignalHandoff,
  run: () => Promise<T>,
): Promise<T> {
  const signals = control.signals ?? process;
  const restoreRawMode = control.input.isRaw === true
    && typeof control.input.setRawMode === "function";
  // Do not make this an AbortController: aborting the parent-side promise
  // could orphan a still-running child. The injected runner owns its process.
  const keepParentAlive = () => {};

  control.pauseReadline();
  if (restoreRawMode) control.input.setRawMode!(false);
  signals.on("SIGINT", keepParentAlive);
  try {
    return await run();
  } finally {
    signals.off("SIGINT", keepParentAlive);
    if (restoreRawMode) control.input.setRawMode!(true);
    control.resumeReadline();
  }
}

function streamIO(
  output: NodeJS.WritableStream,
  error: NodeJS.WritableStream,
): ShellSessionIO {
  return {
    write(text) {
      output.write(text);
    },
    writeError(text) {
      error.write(text);
    },
    clear() {
      output.write("\u001b[2J\u001b[H");
    },
  };
}

/**
 * Start the scrollback-preserving readline UI. A temporary parent-side SIGINT
 * listener exists only while the injected runner owns a foreground child; the
 * terminal still delivers SIGINT directly to that child process.
 */
export async function startInteractiveShell(
  options: InteractiveShellOptions,
): Promise<void> {
  const input = options.input ?? process.stdin;
  const output = options.output ?? process.stdout;
  const error = options.error ?? process.stderr;
  if (
    options.requireTTY !== false
    && (!streamIsTTY(input) || !streamIsTTY(output))
  ) {
    throw new Error("The interactive TT shell requires a TTY.");
  }

  let readline: Interface | undefined;
  const terminalInput = input as RawModeInput;
  const foregroundRunner: ShellCommandRunner = (request) =>
    withForegroundSignalHandoff(
      {
        input: terminalInput,
        pauseReadline: () => readline?.pause(),
        resumeReadline: () => readline?.resume(),
      },
      () => options.runner(request),
    );
  const session = await createShellSession({
    runner: foregroundRunner,
    io: streamIO(output, error),
    agent: options.agent,
    cwd: options.cwd,
    env: options.env,
    version: options.version,
  });
  const completer = createCommandCompleter(() => session.snapshot().mode);
  readline = createInterface({
    input,
    output,
    terminal: true,
    historySize: 100,
    removeHistoryDuplicates: true,
    completer,
  });
  readline.on("SIGINT", () => {
    if (session.interruptAgent()) {
      output.write(`${resetShellPromptStyle()}\n`);
      return;
    }
    // In readline raw mode, Ctrl+C is a line-editing action. Clear the current
    // buffer and redraw instead of pausing or terminating the shell.
    output.write(`${resetShellPromptStyle()}\n`);
    readline?.setPrompt(session.prompt());
    readline?.write(null, { ctrl: true, name: "u" });
  });

  output.write(session.banner());
  readline.setPrompt(session.prompt());
  readline.prompt();

  try {
    for await (const line of readline) {
      // Readline has already advanced to the next row. Repaint the submitted
      // prompt above as a closed, full-row surface; the active prompt itself
      // stays unstyled so it cannot bleed into streamed output.
      output.write(renderSubmittedShellInput(
        line,
        (output as NodeJS.WritableStream & { columns?: number }).columns,
      ));
      const action = await session.handleLine(line);
      if (action === "exit") {
        readline.close();
        break;
      }
      output.write("\n");
      readline.setPrompt(session.prompt());
      readline.prompt();
    }
  } finally {
    output.write(resetShellPromptStyle());
    readline.close();
  }
}

function ciEnabled(value: string | undefined): boolean {
  if (!value) return false;
  return !["0", "false", "no", "off"].includes(value.trim().toLowerCase());
}

export function shouldStartInteractiveShell(options: {
  args: readonly string[];
  stdinIsTTY: boolean | undefined;
  stdoutIsTTY: boolean | undefined;
  env?: Readonly<NodeJS.ProcessEnv>;
}): boolean {
  const env = options.env ?? process.env;
  return options.args.length === 0
    && options.stdinIsTTY === true
    && options.stdoutIsTTY === true
    && !ciEnabled(env.CI)
    && env.TERM !== "dumb";
}
