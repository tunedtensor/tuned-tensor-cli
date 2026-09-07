import { spawn } from "node:child_process";
import { constants as osConstants } from "node:os";
import { Command } from "commander";
import chalk from "chalk";
import {
  type AgentConversationClient,
} from "./agent-client.js";
import { createLocalAgentClient, type LocalPiAgent, type LocalPiAgentOptions } from "./local-agent-client.js";
import { LocalAgentStore } from "./agent-store.js";
import { createPiModelRuntime, readStoredProviderSecrets, type AgentModelRuntime } from "./agent-model.js";
import type { AgentToolApi } from "./agent-tools.js";
import type { AgentMutationApi, AgentMutationGuard } from "./agent-approval.js";
import { createCloudActionContext } from "./agent-approval.js";
import * as api from "./client.js";
import { getApiKey, getBaseUrl, getAgentSelection, getAgentConfigDir, getConfigRevision } from "./config.js";
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
import { registerAuthCommands } from "./commands/auth.js";
import { registerBalanceCommands } from "./commands/balance.js";
import { registerTopupCommands } from "./commands/topup.js";
import { registerPublishCommand } from "./commands/publish.js";
import { registerCloudCommands, registerUsageCommand } from "./commands/cloud.js";
import { checkForCliUpdate, formatCliUpdateNotice } from "./update-check.js";
export { extractPassthroughOptions } from "./passthrough.js";

function createAccountToolApi(opts: api.ClientOpts): AgentToolApi {
  return {
    get: async (path, query) => api.get(path, query, opts),
    postRead: async (path, body) => api.post(path, body, opts),
    propose: async (action) => action,
  };
}

function createAccountMutationApi(opts: api.ClientOpts): AgentMutationApi {
  const headers = (guard?: AgentMutationGuard): Record<string, string> | undefined => guard ? {
    "x-tuned-tensor-action-id": guard.actionId,
    ...(guard.expectedUpdatedAt ? { "x-tuned-tensor-expected-updated-at": guard.expectedUpdatedAt } : {}),
  } : undefined;
  return {
    get: async (path) => api.get(path, undefined, opts),
    post: async (path, body, guard) => api.post(path, body, opts, headers(guard)),
    put: async (path, body, guard) => api.put(path, body, opts, headers(guard)),
  };
}

export interface SelfCommandOptions {
  cwd?: string;
  env?: NodeJS.ProcessEnv;
  entrypoint?: string;
  signal?: AbortSignal;
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
      "The laptop-local agent is not configured. Use /login tunedtensor for managed inference, or /login and /model for your own provider. Local workflow commands need no token.",
    );
  }
  const modelRuntime = await getModelRuntime();
  const secret = getApiKey({ apiKey: env.TUNED_TENSOR_API_KEY });
  const accountOpts = { apiKey: secret, baseUrl: getBaseUrl({ baseUrl: env.TUNED_TENSOR_URL }) };
  const toolApi = runtime.agentToolApi ?? createAccountToolApi(accountOpts);
  const mutationApi = runtime.agentMutationApi ?? createAccountMutationApi(accountOpts);
  const providerSecrets = Object.entries(env)
    .filter(([name, value]) => value && /(?:API_KEY|TOKEN|SECRET)$/i.test(name))
    .map(([, value]) => value!)
    .filter((value) => value.length >= 8 && value !== secret);
  return createLocalAgentClient({
    store: runtime.agentStore ?? new LocalAgentStore(getAgentConfigDir(), {
      secretValues: [...(secret ? [secret] : []), ...providerSecrets],
      secretValueProvider: readStoredProviderSecrets,
    }),
    workspaceRoot,
    selection,
    modelRuntime,
    toolApi,
    mutationApi,
    cloudEnabled: Boolean(secret),
    cloudContext: secret ? createCloudActionContext(accountOpts.baseUrl, secret) : undefined,
    runPipelineCommand: async (args, options) => await (
      runtime.runSelfCommand ?? runSelfCommand
    )(args, {
      cwd: options.cwd,
      env,
      entrypoint: runtime.argv?.[1],
      signal: options.signal,
    }),
    createAgent: runtime.createPiAgent,
  });
}

function createModelRuntimeGetter(
  runtime: CliRuntime,
  getEnvironment: () => NodeJS.ProcessEnv = () => runtime.env ?? process.env,
): () => Promise<AgentModelRuntime & { streamSimple?: (...args: any[]) => any }> {
  const injected = runtime.modelRuntime;
  let promise: Promise<AgentModelRuntime & { streamSimple?: (...args: any[]) => any }> | undefined;
  let builtFingerprint: string | undefined;
  return async () => {
    if (injected) return injected;
    const env = getEnvironment();
    const fingerprint = agentConfigFingerprint(env);
    if (!promise || builtFingerprint !== fingerprint) {
      builtFingerprint = fingerprint;
      const attempt = createPiModelRuntime(env);
      promise = attempt;
      attempt.catch(() => { if (promise === attempt) promise = undefined; });
    }
    return await promise;
  };
}

function agentConfigFingerprint(env: NodeJS.ProcessEnv): string {
  return JSON.stringify([
    getConfigRevision(), getAgentSelection(env),
    getApiKey({ apiKey: env.TUNED_TENSOR_API_KEY }),
    getBaseUrl({ baseUrl: env.TUNED_TENSOR_URL }),
  ]);
}

function createLazyDefaultAgentClient(
  runtime: CliRuntime,
  env: NodeJS.ProcessEnv,
  workspaceRoot: string,
  getModelRuntime: () => Promise<AgentModelRuntime & { streamSimple?: (...args: any[]) => any }>,
): AgentConversationClient {
  let pending: Promise<AgentConversationClient> | undefined;
  let builtFingerprint: string | undefined;
  const client = () => {
    const fingerprint = agentConfigFingerprint(env);
    if (!pending || builtFingerprint !== fingerprint) {
      builtFingerprint = fingerprint;
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
    const onAbort = () => forwardSignal("SIGTERM");
    const cleanup = () => {
      process.off("SIGINT", onSigint);
      process.off("SIGTERM", onSigterm);
      if (process.platform !== "win32") process.off("SIGHUP", onSighup);
      options.signal?.removeEventListener("abort", onAbort);
    };
    process.on("SIGINT", onSigint);
    process.on("SIGTERM", onSigterm);
    if (process.platform !== "win32") process.on("SIGHUP", onSighup);
    options.signal?.addEventListener("abort", onAbort, { once: true });
    if (options.signal?.aborted) onAbort();
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
  root: { color?: boolean; apiKey?: string; baseUrl?: string },
  base: NodeJS.ProcessEnv,
): NodeJS.ProcessEnv {
  return {
    ...base,
    ...(root.color === false ? { FORCE_COLOR: "0" } : {}),
    ...(root.apiKey ? { TUNED_TENSOR_API_KEY: root.apiKey } : {}),
    ...(root.baseUrl ? { TUNED_TENSOR_URL: root.baseUrl } : {}),
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
  const getModelRuntime = createModelRuntimeGetter(runtime, () => childEnvironment(program.opts(), env));

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
    .option("--api-key <key>", "TT access token for managed inference and cloud operations")
    .option("--base-url <url>", "Tuned Tensor API base URL")
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
  registerAuthCommands(program);
  registerBalanceCommands(program);
  registerTopupCommands(program);
  registerUsageCommand(program);
  registerPublishCommand(program);
  registerCloudCommands(program);
  registerAgentCommands(program, {
    get env() { return childEnvironment(program.opts(), env); },
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
  tt hardware
  tt doctor tunedtensor.json
  tt pipeline run --spec tunedtensor.json --dry-run
  tt serve active
  tt runs list
  tt auth login          Enable managed agent inference and cloud access
  tt cloud runs list
  tt usage
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
