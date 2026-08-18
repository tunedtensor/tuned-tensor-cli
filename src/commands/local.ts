import { Command } from "commander";
import chalk from "chalk";
import { extractPassthroughOptions } from "../passthrough.js";
import type { executeLocalCommand } from "../local-runner.js";
import type { ShellCommandResult } from "../shell.js";

export const LOCAL_COMMANDS = [
  "info",
  "init",
  "doctor",
  "validate",
  "run",
  "serve",
  "runs",
  "models",
] as const;

export type LocalCommandName = (typeof LOCAL_COMMANDS)[number];

export const LOCAL_COMMAND_DESCRIPTIONS: Record<LocalCommandName, string> = {
  info: "Show the installed local runtime version and runner status",
  init: "Create a local tunedtensor.json behavior spec",
  doctor: "Check the host and optional run input before starting work",
  validate: "Validate a local behavior spec without executing it",
  run: "Run the local fine-tuning and evaluation workflow",
  serve: "Serve a verified adapter, the active model, or the protected base",
  runs: "Inspect locally stored runs",
  models: "Inspect, verify, prefetch, or serve local models",
};

export interface LocalCommandRuntime {
  invokeLocal: typeof executeLocalCommand;
  cwd: string;
  env: NodeJS.ProcessEnv;
  stdin: NodeJS.ReadableStream;
  stdout: NodeJS.WritableStream;
  stderr: NodeJS.WritableStream;
}

function applyExitCode(result: ShellCommandResult | void): void {
  if (result?.exitCode !== null && result?.exitCode !== undefined) {
    process.exitCode = result.exitCode;
  }
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

/**
 * Register the local CUDA workflow at the root of `tt`.
 *
 * Hosted command modules stay on disk but are not registered. `tt local …`
 * remains as a hidden alias so older scripts keep working.
 */
export function registerLocalCommands(
  program: Command,
  runtime: LocalCommandRuntime,
): void {
  const runPassthrough = async (forwarded: string[]): Promise<void> => {
    const root = program.opts<{ json?: boolean; color?: boolean }>();
    const passthrough = extractPassthroughOptions(forwarded, root);
    if (!passthrough.color) chalk.level = 0;
    const result = await runtime.invokeLocal(
      passthrough.args.length > 0 ? passthrough.args : ["--help"],
      {
        jsonMode: passthrough.json,
        cwd: runtime.cwd,
        env: childEnvironment({ color: passthrough.color }, runtime.env),
        stdin: runtime.stdin,
        stdout: runtime.stdout,
        stderr: runtime.stderr,
      },
    );
    applyExitCode(result);
  };

  for (const name of LOCAL_COMMANDS) {
    program
      .command(name)
      .description(LOCAL_COMMAND_DESCRIPTIONS[name])
      .helpOption(false)
      .allowUnknownOption()
      .argument("[args...]", "Local command arguments")
      .action(async (args: string[]) => {
        await runPassthrough([name, ...args]);
      });
  }

  program
    .command("local", { hidden: true })
    .description("Hidden alias for the local CUDA workflow")
    .helpOption(false)
    .allowUnknownOption()
    .argument("[args...]", "TT Local command and arguments")
    .action(async (args: string[]) => {
      await runPassthrough(args);
    });
}
