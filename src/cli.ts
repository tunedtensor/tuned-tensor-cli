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
import { get, post, put } from "./client.js";
import { getAgentSelection, getApiKey, getConfigDir } from "./config.js";
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
import { registerAuthCommands } from "./commands/auth.js";
import { registerSpecsCommands } from "./commands/specs.js";
import { registerRunsCommands } from "./commands/runs.js";
import { registerDatasetsCommands } from "./commands/datasets.js";
import { registerLabelCommands } from "./commands/label.js";
import { registerModelsCommands } from "./commands/models.js";
import { registerBalanceCommands } from "./commands/balance.js";
import { registerTopupCommands } from "./commands/topup.js";
import { registerInitCommand } from "./commands/init.js";
import { registerEvalCommand } from "./commands/eval.js";
import { registerPushCommand } from "./commands/push.js";
import { registerAgentCommands } from "./commands/agent.js";
import { registerPipelineCommands } from "./commands/pipeline.js";

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
  cloud: { apiKey?: string; baseUrl?: string },
): Promise<AgentConversationClient> {
  const selection = getAgentSelection(env);
  if (!selection) {
    throw new Error(
      "The laptop-local agent is not configured. Run `tt agent models --all`, then `tt agent configure --provider <provider> --model <model>`.",
    );
  }
  const modelRuntime = runtime.modelRuntime ?? await createPiModelRuntime();
  const clientOpts = { apiKey: cloud.apiKey, baseUrl: cloud.baseUrl };
  const toolApi = runtime.agentToolApi ?? {
    get: async (path: string, query?: Record<string, string | number | undefined>) => await get(path, query, clientOpts),
    postRead: async (path: string, body: unknown) => await post(path, body, clientOpts),
    propose: async (action) => action,
  } satisfies AgentToolApi;
  const mutationApi = runtime.agentMutationApi ?? {
    get: async (path: string) => await get(path, undefined, clientOpts),
    post: async (path: string, body?: unknown, guard?) => await post(path, body, clientOpts, guard ? {
      "X-Tuned-Tensor-Action-Id": guard.actionId,
      "X-Tuned-Tensor-Operation": guard.operation,
    } : undefined),
    put: async (path: string, body?: unknown, guard?) => await put(path, body, clientOpts, guard ? {
      "X-Tuned-Tensor-Action-Id": guard.actionId,
      "X-Tuned-Tensor-Operation": guard.operation,
      ...(guard.expectedUpdatedAt ? { "X-Tuned-Tensor-Expected-Updated-At": guard.expectedUpdatedAt } : {}),
    } : undefined),
  } satisfies AgentMutationApi;
  const secret = getApiKey(clientOpts);
  const providerSecrets = Object.entries(env)
    .filter(([name, value]) => value && /(?:API_KEY|TOKEN|SECRET)$/i.test(name))
    .map(([, value]) => value!)
    .filter((value) => value.length >= 8 && value !== secret);
  return createLocalAgentClient({
    store: runtime.agentStore ?? new LocalAgentStore(join(getConfigDir(), "agent"), {
      secretValues: [...(secret ? [secret] : []), ...providerSecrets],
    }),
    selection,
    modelRuntime,
    toolApi,
    mutationApi,
    createAgent: runtime.createPiAgent,
  });
}

function createLazyDefaultAgentClient(
  runtime: CliRuntime,
  env: NodeJS.ProcessEnv,
  cloud: { apiKey?: string; baseUrl?: string },
): AgentConversationClient {
  let pending: Promise<AgentConversationClient> | undefined;
  const client = () => {
    if (!pending) {
      // Retry on the next call if creation fails (for example when the user
      // configures the agent later in the same shell session).
      const attempt = createDefaultAgentClient(runtime, env, cloud);
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
    runTurn: async (id, prompt, onEvent, signal) =>
      await (await client()).runTurn(id, prompt, onEvent, signal),
    approveAction: async (id, onEvent, signal) =>
      await (await client()).approveAction(id, onEvent, signal),
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

interface PassthroughOptions {
  args: string[];
  json: boolean;
  color: boolean;
}

/**
 * Pull root presentation flags out of a passthrough namespace.
 *
 * Commander cannot know the option grammar of the separately versioned local
 * package, so `tt local --json runs list` arrives as variadic arguments.
 */
export function extractPassthroughOptions(
  args: readonly string[],
  root: { json?: boolean; color?: boolean },
): PassthroughOptions {
  let json = Boolean(root.json);
  let color = root.color !== false;
  const forwarded: string[] = [];

  for (const arg of args) {
    if (arg === "--json") {
      json = true;
    } else if (arg === "--no-color") {
      color = false;
    } else {
      forwarded.push(arg);
    }
  }

  return { args: forwarded, json, color };
}

function childEnvironment(
  root: {
    apiKey?: string;
    baseUrl?: string;
    color?: boolean;
  },
  base: NodeJS.ProcessEnv,
): NodeJS.ProcessEnv {
  return {
    ...base,
    ...(root.apiKey ? { TUNED_TENSOR_API_KEY: root.apiKey } : {}),
    ...(root.baseUrl ? { TUNED_TENSOR_URL: root.baseUrl } : {}),
    ...(root.color === false ? { FORCE_COLOR: "0" } : {}),
  };
}

function applyExitCode(result: ShellCommandResult | void): void {
  if (result?.exitCode !== null && result?.exitCode !== undefined) {
    process.exitCode = result.exitCode;
  }
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
  const injectedModelRuntime = runtime.modelRuntime;
  let modelRuntimePromise: Promise<AgentModelRuntime> | undefined;
  const getModelRuntime = async () => {
    if (injectedModelRuntime) return injectedModelRuntime;
    modelRuntimePromise ??= createPiModelRuntime();
    return await modelRuntimePromise;
  };

  const shellRunner = async (
    request: ShellCommandRequest,
  ): Promise<ShellCommandResult> => {
    const root = program.opts<{
      apiKey?: string;
      baseUrl?: string;
      color?: boolean;
    }>();
    const args = request.target === "local"
      ? ["local", ...request.args]
      : request.args;
    return await invokeSelf(args, {
      cwd: request.cwd,
      env: childEnvironment(root, env),
      entrypoint: runtime.argv?.[1],
    });
  };

  const openShell = async (): Promise<void> => {
    const root = program.opts<{
      apiKey?: string;
      baseUrl?: string;
      color?: boolean;
    }>();
    const shellEnvironment = childEnvironment(root, env);
    const output = runtime.stdout ?? process.stdout;
    const error = runtime.stderr ?? process.stderr;
    const agent = createShellAgent({
      client: runtime.agentClient ?? createLazyDefaultAgentClient(runtime, shellEnvironment, {
        apiKey: root.apiKey ?? shellEnvironment.TUNED_TENSOR_API_KEY,
        baseUrl: root.baseUrl ?? shellEnvironment.TUNED_TENSOR_URL,
      }),
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
    });
  };

  program
    .name("tt")
    .description("Tuned Tensor — converse, train, and inspect from one terminal")
    .version(version)
    .option("-k, --api-key <key>", "API key (overrides stored key)")
    .option(
      "-u, --base-url <url>",
      "API base URL (default: https://tunedtensor.com)",
    )
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

  registerAuthCommands(program);
  registerSpecsCommands(program);
  registerRunsCommands(program);
  registerDatasetsCommands(program);
  registerLabelCommands(program);
  registerModelsCommands(program);
  registerBalanceCommands(program);
  registerTopupCommands(program);
  registerInitCommand(program);
  registerEvalCommand(program);
  registerPushCommand(program);
  registerPipelineCommands(program);
  registerAgentCommands(program, {
    env,
    output: runtime.stdout ?? process.stdout,
    getRuntime: getModelRuntime,
  });

  program
    .command("status")
    .description("Show read-only cloud, local, and project context")
    .option("--target <target>", "Show cloud or local workflow status")
    .action(async (commandOptions: { target?: string }) => {
      const root = program.opts<{
        apiKey?: string;
        baseUrl?: string;
        color?: boolean;
      }>();
      const context = await discoverShellContext({
        cwd,
        env: childEnvironment(root, env),
      });
      const target = commandOptions.target ?? context.inferredTarget;
      if (target !== "cloud" && target !== "local") {
        throw new Error("--target must be cloud or local.");
      }
      const output = runtime.stdout ?? process.stdout;
      if (isJsonMode()) {
        output.write(`${JSON.stringify({
          target,
          context,
        }, null, 2)}\n`);
        return;
      }
      output.write(`${chalk.bold.hex("#8B5CF6")("Tuned Tensor status")}\n`);
      output.write(`${formatShellStatus(context, target).join("\n")}\n\n`);
      output.write(`${chalk.bold("Context")}\n`);
      output.write(
        `${formatShellContext(
          context,
          target,
          commandOptions.target ? "command-option" : context.targetSource,
        ).join("\n")}\n`,
      );
    });

  program
    .command("local")
    .description("Run the local CUDA training and model workflow")
    .helpOption(false)
    .allowUnknownOption()
    .argument("[args...]", "TT Local command and arguments")
    .action(async (args: string[]) => {
      const root = program.opts<{ json?: boolean; color?: boolean }>();
      const passthrough = extractPassthroughOptions(args, root);
      if (!passthrough.color) chalk.level = 0;
      const result = await invokeLocal(
        passthrough.args.length > 0 ? passthrough.args : ["--help"],
        {
          jsonMode: passthrough.json,
          cwd,
          env: childEnvironment(
            { ...root, color: passthrough.color },
            env,
          ),
          stdin: runtime.stdin ?? process.stdin,
          stdout: runtime.stdout ?? process.stdout,
          stderr: runtime.stderr ?? process.stderr,
        },
      );
      applyExitCode(result);
    });

  program
    .command("cloud")
    .description("Run a hosted command explicitly")
    .helpOption(false)
    .allowUnknownOption()
    .argument("[args...]", "Hosted tt command and arguments")
    .action(async (args: string[]) => {
      const root = program.opts<{
        apiKey?: string;
        baseUrl?: string;
        json?: boolean;
        color?: boolean;
      }>();
      const passthrough = extractPassthroughOptions(args, root);
      const forwarded = [
        ...(passthrough.json ? ["--json"] : []),
        ...(!passthrough.color ? ["--no-color"] : []),
        ...(passthrough.args.length > 0 ? passthrough.args : ["--help"]),
      ];
      const result = await invokeSelf(forwarded, {
        cwd,
        env: childEnvironment(
          { ...root, color: passthrough.color },
          env,
        ),
        entrypoint: runtime.argv?.[1],
      });
      applyExitCode(result);
    });

  program
    .command("shell")
    .description("Open the interactive cloud/local terminal")
    .action(openShell);

  program.addHelpText(
    "after",
    `
Workflows:
  tt                     Open the conversational terminal (TTY only)
  tt status              Inspect cloud/local project context
  tt cloud <command>     Run a hosted command explicitly
  tt local <command>     Run a local GPU command

Examples:
  tt runs list
  tt local doctor tunedtensor.json
  tt local run tunedtensor.json --dry-run
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
    const invokeSelf = runtime.runSelfCommand ?? runSelfCommand;
    const output = runtime.stdout ?? process.stdout;
    const error = runtime.stderr ?? process.stderr;
    const agent = createShellAgent({
      client: runtime.agentClient ?? createLazyDefaultAgentClient(runtime, env, {
        apiKey: env.TUNED_TENSOR_API_KEY,
        baseUrl: env.TUNED_TENSOR_URL,
      }),
      output,
      error,
    });
    await (runtime.startShell ?? startInteractiveShell)({
      runner: async (request) => await invokeSelf(
        request.target === "local"
          ? ["local", ...request.args]
          : request.args,
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
    });
    return;
  }

  // Honor --json even when Commander rejects input before a preAction hook.
  if (argv.includes("--json")) setJsonMode(true);
  await createProgram(version, { ...runtime, argv }).parseAsync(argv);
}
