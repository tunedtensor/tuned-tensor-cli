import { spawn } from "node:child_process";
import { constants as osConstants } from "node:os";
import { Command } from "commander";
import chalk from "chalk";
import {
  type AgentConversationClient,
} from "./agent-client.js";
import { createLocalAgentClient, type LocalPiAgent, type LocalPiAgentOptions } from "./local-agent-client.js";
import { LocalAgentStore } from "./agent-store.js";
import { createPiModelRuntime, type AgentModelRuntime } from "./agent-model.js";
import type { AgentToolApi } from "./agent-tools.js";
import type { AgentMutationApi } from "./agent-approval.js";
import { getApiKey, getAgentSelection, getConfigDir, getConfigRevision } from "./config.js";
import { join } from "node:path";
import { TunedTensorAgentSession } from "./agent.js";
import { executeLocalCommand } from "./local-runner.js";
import {
  shouldStartInteractiveShell,
  startInteractiveShell,
  type ShellCommandRequest,
  type ShellCommandResult,
} from "./shell.js";
import {
  discoverShellContext,
  formatShellContext,
  formatShellStatus,
} from "./shell-context.js";
import { isJsonMode, setJsonMode } from "./output.js";
import { registerLocalCommands } from "./commands/local.js";
import { registerAgentCommands } from "./commands/agent.js";
import { registerPipelineCommands } from "./commands/pipeline.js";
import { checkForCliUpdate, formatCliUpdateNotice } from "./update-check.js";
export { extractPassthroughOptions } from "./passthrough.js";

const LOCAL_ONLY_MESSAGE = "This build of tt is local-only.";

function rejectLocalOnly(): never {
  throw new Error(LOCAL_ONLY_MESSAGE);
}

function createLocalOnlyToolApi(): AgentToolApi {
  return {
    get: async () => rejectLocalOnly(),
    postRead: async () => rejectLocalOnly(),
    propose: async (action) => action,
  };
}

function createLocalOnlyMutationApi(): AgentMutationApi {
  return {
    get: async () => rejectLocalOnly(),
    post: async () => rejectLocalOnly(),
    put: async () => rejectLocalOnly(),
  };
}

export interface SelfCommandOptions {
  cwd?: string;
  env?: NodeJS.ProcessEnv;
  entrypoint?: string;
}

export interface SelfCommandResult extends ShellCommandResult {
  signal: NodeJS.Signals | null;
}

export type SelfCommandRunner = (
  args: string[],
  options?: SelfCommandOptions,
) => Promise<SelfCommandResult>;

export interface CliRuntime {
  argv?: string[];
  cwd?: string;
  env?: NodeJS.ProcessEnv;
  stdin?: NodeJS.ReadableStream;
  stdout?: NodeJS.WritableStream;
  stderr?: NodeJS.WritableStream;
  stdinIsTTY?: boolean;
  stdoutIsTTY?: boolean;
  runSelfCommand?: SelfCommandRunner;
  runLocalCommand?: typeof executeLocalCommand;
  checkForUpdate?: (currentVersion: string) => Promise<{
    currentVersion: string;
    latestVersion: string;
  } | null>;
  startShell?: typeof startInteractiveShell;
  agentClient?: AgentConversationClient;
  modelRuntime?: AgentModelRuntime & { streamSimple?: (...args: any[]) => any };
  createPiAgent?: (options: LocalPiAgentOptions) => LocalPiAgent;
  agentStore?: LocalAgentStore;
  agentToolApi?: AgentToolApi;
  agentMutationApi?: AgentMutationApi;
}

async function createDefaultAgentClient(
  runtime: CliRuntime,
  env: NodeJS.ProcessEnv,
  workspaceRoot: string,
  getModelRuntime: () => Promise<AgentModelRuntime & { streamSimple?: (...args: any[]) => any }>,
): Promise<AgentConversationClient> {
  const selection = getAgentSelection(env);
  if (!selection) {
    throw new Error(
      "The laptop-local agent is not configured. Run `tt agent models --all`, then `tt agent configure --provider <provider> --model <model>`.",
    );
  }
  const modelRuntime = await getModelRuntime();
  const toolApi = runtime.agentToolApi ?? createLocalOnlyToolApi();
  const mutationApi = runtime.agentMutationApi ?? createLocalOnlyMutationApi();
  const secret = getApiKey({ apiKey: env.TUNED_TENSOR_API_KEY });
  const providerSecrets = Object.entries(env)
    .filter(([name, value]) => value && /(?:API_KEY|TOKEN|SECRET)$/i.test(name))
    .map(([, value]) => value!)
    .filter((value) => value.length >= 8 && value !== secret);
  return createLocalAgentClient({
    store: runtime.agentStore ?? new LocalAgentStore(join(getConfigDir(), "agent"), {
      secretValues: [...(secret ? [secret] : []), ...providerSecrets],
    }),
    workspaceRoot,
    selection,
    modelRuntime,
    toolApi,
    mutationApi,
    createAgent: runtime.createPiAgent,
  });
}

function createModelRuntimeGetter(
  runtime: CliRuntime,
): () => Promise<AgentModelRuntime & { streamSimple?: (...args: any[]) => any }> {
  const injected = runtime.modelRuntime;
  let promise: Promise<AgentModelRuntime & { streamSimple?: (...args: any[]) => any }> | undefined;
  return async () => {
    if (injected) return injected;
    promise ??= createPiModelRuntime();
    return await promise;
  };
}

function createLazyDefaultAgentClient(
  runtime: CliRuntime,
  env: NodeJS.ProcessEnv,
  workspaceRoot: string,
  getModelRuntime: () => Promise<AgentModelRuntime & { streamSimple?: (...args: any[]) => any }>,
): AgentConversationClient {
  let pending: Promise<AgentConversationClient> | undefined;
  let builtRevision = -1;
  const client = () => {
    const revision = getConfigRevision();
    if (!pending || builtRevision !== revision) {
      builtRevision = revision;
      // Retry on the next call if creation fails (for example when the user
      // configures the agent later in the same shell session). Changing the
      // model with `/model` writes config and bumps the revision, which also
      // recreates the client here.
      const attempt = createDefaultAgentClient(runtime, env, workspaceRoot, getModelRuntime);
      pending = attempt;
      attempt.catch(() => {
        if (pending === attempt) pending = undefined;
      });
    }
    return pending;
  };
  return {
    createThread: async () => await (await client()).createThread(),
    listThreads: async () => await (await client()).listThreads(),
    getThread: async (id) => await (await client()).getThread(id),
    runTurn: async (id, prompt, onEvent, signal, context) =>
      await (await client()).runTurn(id, prompt, onEvent, signal, context),
    approveAction: async (id, onEvent, signal, context) =>
      await (await client()).approveAction(id, onEvent, signal, context),
    rejectAction: async (id) => await (await client()).rejectAction(id),
  };
}

export async function runSelfCommand(
  args: string[],
  options: SelfCommandOptions = {},
): Promise<SelfCommandResult> {
  const entrypoint = options.entrypoint ?? process.argv[1];
  if (!entrypoint) {
    throw new Error("Cannot locate the tt CLI entrypoint.");
  }

  return await new Promise<SelfCommandResult>((resolve, reject) => {
    const child = spawn(process.execPath, [entrypoint, ...args], {
      cwd: options.cwd,
      env: options.env ?? process.env,
      stdio: "inherit",
      detached: process.platform !== "win32",
    });
    const forwardSignal = (signal: NodeJS.Signals) => {
      // The child command may own another process subtree. Translate a
      // terminal/SSH hangup into SIGTERM so it can run its normal cleanup.
      const childSignal = signal === "SIGHUP" ? "SIGTERM" : signal;
      if (child.pid && process.platform !== "win32") {
        try {
          process.kill(-child.pid, childSignal);
          return;
        } catch {
          // The child may have exited between signal delivery and forwarding.
        }
      }
      child.kill(childSignal);
    };
    const onSigint = () => forwardSignal("SIGINT");
    const onSigterm = () => forwardSignal("SIGTERM");
    const onSighup = () => forwardSignal("SIGHUP");
    const cleanup = () => {
      process.off("SIGINT", onSigint);
      process.off("SIGTERM", onSigterm);
      if (process.platform !== "win32") process.off("SIGHUP", onSighup);
    };
    process.on("SIGINT", onSigint);
    process.on("SIGTERM", onSigterm);
    if (process.platform !== "win32") process.on("SIGHUP", onSighup);
    child.once("error", (error) => {
      cleanup();
      reject(error);
    });
    child.once("close", (exitCode, signal) => {
      cleanup();
      const signalNumber = signal ? osConstants.signals[signal] : undefined;
      resolve({
        exitCode: exitCode ?? (
          typeof signalNumber === "number" ? 128 + signalNumber : 1
        ),
        signal,
      });
    });
  });
}

function childEnvironment(
  root: { color?: boolean },
  base: NodeJS.ProcessEnv,
): NodeJS.ProcessEnv {
  return {
    ...base,
    ...(root.color === false ? { FORCE_COLOR: "0" } : {}),
  };
}

export function createProgram(
  version: string,
  runtime: CliRuntime = {},
): Command {
  const program = new Command();
  const env = runtime.env ?? process.env;
  const invokeSelf = runtime.runSelfCommand ?? runSelfCommand;
  const invokeLocal = runtime.runLocalCommand ?? executeLocalCommand;
  const launchShell = runtime.startShell ?? startInteractiveShell;
  const cwd = runtime.cwd ?? process.cwd();
  const getModelRuntime = createModelRuntimeGetter(runtime);

  const shellRunner = async (
    request: ShellCommandRequest,
  ): Promise<ShellCommandResult> => {
    const root = program.opts<{ color?: boolean }>();
    return await invokeSelf(request.args, {
      cwd: request.cwd,
      env: childEnvironment(root, env),
      entrypoint: runtime.argv?.[1],
    });
  };

  const openShell = async (): Promise<void> => {
    const root = program.opts<{ color?: boolean }>();
    const shellEnvironment = childEnvironment(root, env);
    const output = runtime.stdout ?? process.stdout;
    const error = runtime.stderr ?? process.stderr;
    const agent = createShellAgent({
      client: runtime.agentClient ?? createLazyDefaultAgentClient(
        runtime,
        shellEnvironment,
        cwd,
        getModelRuntime,
      ),
      output,
      error,
    });
    await launchShell({
      runner: shellRunner,
      agent,
      input: runtime.stdin ?? process.stdin,
      output,
      error,
      cwd,
      env: shellEnvironment,
      version,
      agentModelRuntime: getModelRuntime,
    });
  };

  program
    .name("tt")
    .description("Tuned Tensor — converse, train, and inspect from one local terminal")
    .version(version)
    .option("--json", "Output raw JSON")
    .option("--no-color", "Disable colors")
    .showSuggestionAfterError()
    .hook("preAction", () => {
      const root = program.opts<{ json?: boolean; color?: boolean }>();
      if (root.json) setJsonMode(true);
      if (root.color === false) {
        process.env.FORCE_COLOR = "0";
        chalk.level = 0;
      }
    });

  registerLocalCommands(program, {
    invokeLocal,
    cwd,
    env,
    stdin: runtime.stdin ?? process.stdin,
    stdout: runtime.stdout ?? process.stdout,
    stderr: runtime.stderr ?? process.stderr,
  });
  registerPipelineCommands(program);
  registerAgentCommands(program, {
    env,
    output: runtime.stdout ?? process.stdout,
    getRuntime: getModelRuntime,
  });

  program
    .command("status")
    .description("Show local and project context")
    .action(async () => {
      const root = program.opts<{ color?: boolean }>();
      const context = await discoverShellContext({
        cwd,
        env: childEnvironment(root, env),
      });
      const output = runtime.stdout ?? process.stdout;
      if (isJsonMode()) {
        output.write(`${JSON.stringify({ context }, null, 2)}\n`);
        return;
      }
      output.write(`${chalk.bold.hex("#8B5CF6")("Tuned Tensor status")}\n`);
      output.write(`${formatShellStatus(context).join("\n")}\n\n`);
      output.write(`${chalk.bold("Context")}\n`);
      output.write(`${formatShellContext(context).join("\n")}\n`);
    });

  program
    .command("shell")
    .description("Open the conversational terminal")
    .action(openShell);

  program.addHelpText(
    "after",
    `
Examples:
  tt                     Open the conversational terminal (TTY only)
  tt doctor tunedtensor.json
  tt run tunedtensor.json --dry-run
  tt runs list
`,
  );

  return program;
}

function createShellAgent(options: {
  client: AgentConversationClient;
  output: NodeJS.WritableStream;
  error: NodeJS.WritableStream;
}): TunedTensorAgentSession {
  return new TunedTensorAgentSession({
    client: options.client,
    io: {
      write(text) {
        options.output.write(text);
      },
      writeError(text) {
        options.error.write(text);
      },
      clear() {
        options.output.write("\u001b[2J\u001b[H");
      },
    },
  });
}

export async function runCli(
  version: string,
  runtime: CliRuntime = {},
): Promise<void> {
  const argv = runtime.argv ?? process.argv;
  const env = runtime.env ?? process.env;
  const args = argv.slice(2);

  if (
    shouldStartInteractiveShell({
      args,
      stdinIsTTY: runtime.stdinIsTTY ?? process.stdin.isTTY,
      stdoutIsTTY: runtime.stdoutIsTTY ?? process.stdout.isTTY,
      env,
    })
  ) {
    const output = runtime.stdout ?? process.stdout;
    const error = runtime.stderr ?? process.stderr;
    let update: Awaited<ReturnType<NonNullable<CliRuntime["checkForUpdate"]>>> = null;
    try {
      update = await (runtime.checkForUpdate ?? checkForCliUpdate)(version);
    } catch {
      // Version discovery is advisory and must never prevent shell launch.
    }
    if (update) {
      error.write(`${formatCliUpdateNotice(update)}\n\n`);
    }
    const invokeSelf = runtime.runSelfCommand ?? runSelfCommand;
    const getModelRuntime = createModelRuntimeGetter(runtime);
    const agent = createShellAgent({
      client: runtime.agentClient ?? createLazyDefaultAgentClient(
        runtime,
        env,
        runtime.cwd ?? process.cwd(),
        getModelRuntime,
      ),
      output,
      error,
    });
    await (runtime.startShell ?? startInteractiveShell)({
      runner: async (request) => await invokeSelf(
        request.args,
        {
          cwd: request.cwd,
          env,
          entrypoint: argv[1],
        },
      ),
      agent,
      input: runtime.stdin ?? process.stdin,
      output,
      error,
      cwd: runtime.cwd ?? process.cwd(),
      env,
      version,
      agentModelRuntime: getModelRuntime,
    });
    return;
  }

  // Honor --json even when Commander rejects input before a preAction hook.
  if (argv.includes("--json")) setJsonMode(true);
  await createProgram(version, { ...runtime, argv }).parseAsync(argv);
}
