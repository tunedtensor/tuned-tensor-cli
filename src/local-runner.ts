import { constants as osConstants } from "node:os";
import { randomUUID } from "node:crypto";
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import {
  readFile,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import {
  basename,
  dirname,
  isAbsolute,
  join,
  resolve,
} from "node:path";
import { projectLocalSpec } from "./project-spec.js";
import {
  adaptLocalCliText,
  renderLocalOutput,
  type LocalOutputPayload,
} from "./local-output.js";

const DEFAULT_SPEC_NAME = "tunedtensor.json";

export interface LocalCommandOptions {
  jsonMode?: boolean;
  cwd?: string;
  env?: NodeJS.ProcessEnv;
  entrypoint?: string;
  nodeExecutable?: string;
  stdin?: NodeJS.ReadableStream;
  stdout?: NodeJS.WritableStream;
  stderr?: NodeJS.WritableStream;
}

export interface LocalCommandResult extends LocalOutputPayload {
  entrypoint: string | null;
  projectedArgs: string[];
}

interface ProjectedArguments {
  args: string[];
  droppedSpecKeys: string[];
  pathReplacements: PathReplacement[];
  cleanup(): Promise<void>;
}

interface PathReplacement {
  sourcePath: string;
  projectedPath: string;
}

interface SpecArgument {
  path: string;
  replace(projectedPath: string): void;
}

const VALUE_OPTIONS = new Set([
  "--config",
  "--name",
  "--model",
  "--output",
  "--profile",
  "--host",
  "--port",
  "--device",
  "--max-tokens",
  "--temperature",
  "--top-p",
  "--max-concurrent-requests",
  "--spec",
  "--api-key-env",
]);

function isOptionWithInlineValue(value: string): boolean {
  const equals = value.indexOf("=");
  return equals !== -1 && VALUE_OPTIONS.has(value.slice(0, equals));
}

function positionalIndices(args: string[], start: number): number[] {
  const indices: number[] = [];
  let optionsEnded = false;
  for (let index = start; index < args.length; index += 1) {
    const value = args[index]!;
    if (!optionsEnded && value === "--") {
      optionsEnded = true;
      continue;
    }
    if (!optionsEnded && value.startsWith("-")) {
      if (!isOptionWithInlineValue(value) && VALUE_OPTIONS.has(value)) index += 1;
      continue;
    }
    indices.push(index);
  }
  return indices;
}

function optionValueArgument(args: string[], name: string): SpecArgument | null {
  for (let index = 0; index < args.length; index += 1) {
    const value = args[index]!;
    if (value === name) {
      const path = args[index + 1];
      if (!path) return null;
      return {
        path,
        replace(projectedPath) {
          args[index + 1] = projectedPath;
        },
      };
    }
    if (value.startsWith(`${name}=`)) {
      const path = value.slice(name.length + 1);
      if (!path) return null;
      return {
        path,
        replace(projectedPath) {
          args[index] = `${name}=${projectedPath}`;
        },
      };
    }
  }
  return null;
}

function positionalSpecArgument(
  args: string[],
  start: number,
  defaultPath: string,
): SpecArgument {
  const positional = positionalIndices(args, start)[0];
  if (positional !== undefined) {
    return {
      path: args[positional]!,
      replace(projectedPath) {
        args[positional] = projectedPath;
      },
    };
  }
  return {
    path: defaultPath,
    replace(projectedPath) {
      args.splice(start, 0, projectedPath);
    },
  };
}

function selectedSpecArgument(args: string[], cwd: string): SpecArgument | null {
  const command = args[0];
  if (["doctor", "validate", "run"].includes(command ?? "")) {
    return positionalSpecArgument(args, 1, join(cwd, DEFAULT_SPEC_NAME));
  }
  if (
    command === "models"
    && ["prefetch", "verify-base"].includes(args[1] ?? "")
  ) {
    return positionalSpecArgument(args, 2, join(cwd, DEFAULT_SPEC_NAME));
  }

  const serves = command === "serve"
    || (command === "models" && args[1] === "serve");
  if (!serves) return null;
  const explicit = optionValueArgument(args, "--spec");
  if (explicit) return explicit;
  return null;
}

async function pathIsFile(path: string): Promise<boolean> {
  return (await stat(path).catch(() => null))?.isFile() ?? false;
}

function resolveFromCwd(path: string, cwd: string): string {
  if (isAbsolute(path)) return path;
  return resolve(cwd, path);
}

async function projectSpecArguments(
  originalArgs: string[],
  cwd: string,
): Promise<ProjectedArguments> {
  const args = [...originalArgs];
  if (args.includes("--help") || args.includes("-h")) {
    return {
      args,
      droppedSpecKeys: [],
      pathReplacements: [],
      cleanup: async () => {},
    };
  }
  const selected = selectedSpecArgument(args, cwd);
  if (!selected) {
    return {
      args,
      droppedSpecKeys: [],
      pathReplacements: [],
      cleanup: async () => {},
    };
  }

  const sourcePath = resolveFromCwd(selected.path, cwd);
  if (!await pathIsFile(sourcePath)) {
    return {
      args,
      droppedSpecKeys: [],
      pathReplacements: [],
      cleanup: async () => {},
    };
  }

  const raw = JSON.parse(await readFile(sourcePath, "utf8")) as unknown;
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    throw new Error(`${sourcePath} must contain a JSON object.`);
  }
  const body = raw as Record<string, unknown>;
  const runRequestKeys = [
    "run_id",
    "behavior_spec_id",
    "run_number",
    "spec_snapshot",
  ].filter((key) => key in body);
  if ("spec_snapshot" in body || runRequestKeys.length >= 2) {
    // The public TT Local CLI deliberately rejects full run-request payloads.
    // Preserve that diagnostic instead of projecting one into a partial spec.
    return {
      args,
      droppedSpecKeys: [],
      pathReplacements: [],
      cleanup: async () => {},
    };
  }
  const projected = projectLocalSpec(body);
  if (projected.droppedKeys.length === 0) {
    return {
      args,
      droppedSpecKeys: [],
      pathReplacements: [],
      cleanup: async () => {},
    };
  }
  const temporaryPath = join(
    dirname(sourcePath),
    `.${basename(sourcePath)}.tt-local-${process.pid}-${randomUUID()}.json`,
  );
  await writeFile(
    temporaryPath,
    `${JSON.stringify(projected.body, null, 2)}\n`,
    { encoding: "utf8", flag: "wx", mode: 0o600 },
  );
  selected.replace(temporaryPath);
  return {
    args,
    droppedSpecKeys: projected.droppedKeys,
    pathReplacements: [{
      sourcePath,
      projectedPath: temporaryPath,
    }],
    cleanup: () => rm(temporaryPath, { force: true }),
  };
}

export function resolveLocalCliEntrypoint(): string {
  const entrypoint = fileURLToPath(new URL("./local-runtime.js", import.meta.url));
  if (!existsSync(entrypoint)) {
    throw new Error(
      "The bundled local runtime is missing. Rebuild or reinstall @tuned-tensor/cli.",
    );
  }
  return entrypoint;
}

export function localCommandStreamsStdout(args: readonly string[]): boolean {
  const serves = args[0] === "serve"
    || (args[0] === "models" && args[1] === "serve");
  return serves
    && !args.includes("--print-command")
    && !args.includes("--help")
    && !args.includes("-h");
}

function parseJsonOutput(stdout: string): {
  hasJson: boolean;
  json: unknown;
} {
  if (!stdout.trim()) return { hasJson: false, json: undefined };
  try {
    return { hasJson: true, json: JSON.parse(stdout) as unknown };
  } catch {
    return { hasJson: false, json: undefined };
  }
}

function signalExitCode(signal: NodeJS.Signals | null): number {
  if (!signal) return 1;
  const signalNumber = osConstants.signals[signal];
  return typeof signalNumber === "number" ? 128 + signalNumber : 1;
}

function writeStream(stream: NodeJS.WritableStream, chunk: string): void {
  stream.write(chunk);
}

function normalizeProjectedPaths(
  value: string,
  replacements: readonly PathReplacement[],
): string {
  let normalized = value;
  for (const replacement of replacements) {
    const encodedProjectedPath = JSON.stringify(
      replacement.projectedPath,
    ).slice(1, -1);
    const encodedSourcePath = JSON.stringify(
      replacement.sourcePath,
    ).slice(1, -1);
    normalized = normalized
      .replaceAll(encodedProjectedPath, encodedSourcePath)
      .replaceAll(replacement.projectedPath, replacement.sourcePath);
  }
  return normalized;
}

async function spawnLocalCli(args: {
  entrypoint: string;
  commandArgs: string[];
  cwd: string;
  env: NodeJS.ProcessEnv;
  nodeExecutable: string;
  streamingStdout: boolean;
  jsonMode: boolean;
  stdin?: NodeJS.ReadableStream;
  stdout?: NodeJS.WritableStream;
  stderr?: NodeJS.WritableStream;
  pathReplacements: readonly PathReplacement[];
}): Promise<{
  exitCode: number;
  signal: NodeJS.Signals | null;
  stdout: string;
  stderr: string;
  errorMessage?: string;
}> {
  const output = args.stdout ?? process.stdout;
  const errors = args.stderr ?? process.stderr;
  const pipeInput = Boolean(args.stdin && args.stdin !== process.stdin);
  const pipeStreamingOutput = args.streamingStdout
    && Boolean(args.stdout && args.stdout !== process.stdout);
  const child = spawn(
    args.nodeExecutable,
    [args.entrypoint, ...args.commandArgs],
    {
      cwd: args.cwd,
      env: args.env,
      stdio: [
        pipeInput ? "pipe" : "inherit",
        args.streamingStdout && !pipeStreamingOutput ? "inherit" : "pipe",
        "pipe",
      ],
      detached: process.platform !== "win32",
    },
  );

  if (pipeInput && args.stdin && child.stdin) args.stdin.pipe(child.stdin);
  let stdout = "";
  let stderr = "";
  child.stdout?.setEncoding("utf8");
  child.stderr?.setEncoding("utf8");
  child.stdout?.on("data", (chunk: string) => {
    const normalized = normalizeProjectedPaths(
      chunk,
      args.pathReplacements,
    );
    if (args.streamingStdout) writeStream(output, normalized);
    else stdout += normalized;
  });
  child.stderr?.on("data", (chunk: string) => {
    const normalized = normalizeProjectedPaths(
      chunk,
      args.pathReplacements,
    );
    stderr += normalized;
    writeStream(
      errors,
      args.jsonMode ? normalized : adaptLocalCliText(normalized),
    );
  });

  return new Promise((resolveChild) => {
    let spawnError: Error | undefined;
    const forwardSignal = (signal: NodeJS.Signals) => {
      // SIGHUP has no cleanup handler in the standalone local CLI. Translate
      // it to SIGTERM so its Python/server subtree is stopped deliberately.
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
      spawnError = error;
    });
    child.once("close", (code, signal) => {
      cleanup();
      resolveChild({
        exitCode: spawnError ? 1 : code ?? signalExitCode(signal),
        signal,
        stdout,
        stderr,
        ...(spawnError ? { errorMessage: spawnError.message } : {}),
      });
    });
  });
}

function setupFailure(
  originalArgs: string[],
  error: unknown,
): LocalCommandResult {
  const message = error instanceof Error ? error.message : String(error);
  return {
    args: [...originalArgs],
    projectedArgs: [...originalArgs],
    entrypoint: null,
    exitCode: 1,
    signal: null,
    stdout: "",
    stderr: "",
    hasJson: false,
    json: undefined,
    streamingStdout: false,
    droppedSpecKeys: [],
    errorMessage: message,
  };
}

export async function runLocalCommand(
  args: string[],
  options: LocalCommandOptions = {},
): Promise<LocalCommandResult> {
  const cwd = resolve(options.cwd ?? process.cwd());
  let projected: ProjectedArguments | undefined;
  try {
    const entrypoint = options.entrypoint ?? resolveLocalCliEntrypoint();
    projected = await projectSpecArguments(args, cwd);
    const streamingStdout = localCommandStreamsStdout(projected.args);
    const result = await spawnLocalCli({
      entrypoint,
      commandArgs: projected.args,
      cwd,
      env: options.env ?? process.env,
      nodeExecutable: options.nodeExecutable ?? process.execPath,
      streamingStdout,
      jsonMode: options.jsonMode ?? false,
      stdin: options.stdin,
      stdout: options.stdout,
      stderr: options.stderr,
      pathReplacements: projected.pathReplacements,
    });
    const parsed = parseJsonOutput(result.stdout);
    return {
      args: [...args],
      projectedArgs: projected.args,
      entrypoint,
      ...result,
      ...parsed,
      streamingStdout,
      droppedSpecKeys: projected.droppedSpecKeys,
    };
  } catch (error) {
    return setupFailure(args, error);
  } finally {
    await projected?.cleanup().catch(() => undefined);
  }
}

export async function executeLocalCommand(
  args: string[],
  options: LocalCommandOptions = {},
): Promise<LocalCommandResult> {
  const result = await runLocalCommand(args, options);
  renderLocalOutput(result, {
    jsonMode: options.jsonMode,
    stdout: options.stdout,
  });
  if (result.exitCode !== 0) process.exitCode = result.exitCode;
  return result;
}
