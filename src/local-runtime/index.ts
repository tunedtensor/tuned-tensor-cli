import { readFileSync, realpathSync } from "node:fs";
import { createHash } from "node:crypto";
import { readFile, stat } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { cwd } from "node:process";
import { fileURLToPath } from "node:url";
import { compareRuns } from "./compare.js";
import { assertArtifactManifest } from "./artifacts.js";
import { fineTuneRunRequestSchema, isFoundationSpecFile, localBehaviorSpecFileSchema, localRunnerConfigSchema, type FineTuneRunRequest, type LocalRunnerConfig, type SpecSnapshot } from "./contracts.js";
import { buildSystemMessage } from "./dataset.js";
import {
  loadLocalRunnerConfig,
  fingerprintLocalBaseModel,
  runLocalFineTune,
  validateLocalFineTuneInput,
} from "./orchestrator.js";
import { runDoctor } from "./doctor.js";
import { assessHardware } from "./hardware.js";
import { assertUsableModelArtifact, defaultBaseModelRevision } from "./model-registry.js";
import {
  buildLocalBaseModelServerLaunch,
  buildLocalModelServerLaunch,
  serveLocalModel,
  type LocalModelServerLaunch,
} from "./model-server.js";
import { prefetchBaseModel } from "./prefetch.js";
import { createLocalStore, type LocalModelRecord, type LocalStore } from "./store.js";
import {
  DEFAULT_LOCAL_SPEC_PATH,
  assertFoundationSpecReady,
  assertLocalRunInputReady,
  initLocalRunnerConfigFile,
  initLocalSpecFile,
  loadLocalRunInput,
} from "./local-project.js";
import { sanitizeLogLine, type LocalRunProgressEvent, type LocalRunReporter } from "./run-reporter.js";
import { activateModel, getActiveModel, rollbackActiveModel } from "./active-model.js";
import { localRuntimePackageRoot } from "./package-root.js";

export * from "./compare.js";
export * from "./contracts.js";
export * from "./dataset.js";
export * from "./model-server.js";
export * from "./orchestrator.js";
export * from "./local-project.js";
export * from "./prefetch.js";
export * from "./run-reporter.js";
export * from "./store.js";
export * from "./general-regression.js";
export * from "./active-model.js";

export interface LocalRunnerInfo {
  name: "tuned-tensor-local";
  status: "local";
  description: string;
  version: string;
}

function packageVersion(): string {
  try {
    const packageJson = JSON.parse(
      readFileSync(resolve(localRuntimePackageRoot(import.meta.url), "package.json"), "utf8"),
    ) as { version?: unknown };
    return typeof packageJson.version === "string" ? packageJson.version : "unknown";
  } catch {
    return "unknown";
  }
}

export const TT_LOCAL_VERSION = packageVersion();

export function getLocalRunnerInfo(): LocalRunnerInfo {
  return {
    name: "tuned-tensor-local",
    status: "local",
    description: "Local CUDA LoRA fine-tuning with held-out base-versus-tuned evaluation.",
    version: TT_LOCAL_VERSION,
  };
}

function readOption(argv: string[], name: string): string | undefined {
  const index = argv.indexOf(name);
  if (index !== -1) return argv[index + 1];
  const inline = argv.find((value) => value.startsWith(`${name}=`));
  return inline?.slice(name.length + 1);
}

function hasFlag(argv: string[], name: string): boolean {
  return argv.includes(name);
}

function printHelp(): void {
  console.log(`Usage: tt <command> [options]

Commands:
  info                              Show package and runner information
  init [--name "My Local Model"] [--model Qwen/Qwen3.5-2B] [--output tunedtensor.json] [--profile spark] [--force]
  doctor [tunedtensor.json] [--config local-runner.json]
  hardware [--config local-runner.json] [--quick]
  validate [tunedtensor.json] [--config local-runner.json]
  run [tunedtensor.json] [--config local-runner.json] [--dry-run] [--verbose] [--quiet]
  serve <model-id|active|base> [--config local-runner.json] [--host 127.0.0.1] [--port 8000]
  runs list|get|events|report|compare [args] [--config local-runner.json]
  models list|get|verify|prefetch|verify-base|active|activate|rollback|serve [args] [--config local-runner.json]

Global options:
  -h, --help                       Show help
  -V, --version                    Show the installed version

The run command writes local artifacts under config.artifactRoot, defaulting to
.tuned-tensor/artifacts. The file-backed local store defaults to
~/.tuned-tensor/store unless config.storeRoot, TT_LOCAL_HOME, or
TUNED_TENSOR_HOME is set.`);
}

interface CliOptionDefinition {
  name: string;
  value?: string;
  description: string;
}

interface CliCommandDefinition {
  usage: string;
  description: string;
  options: readonly CliOptionDefinition[];
  minPositionals?: number;
  maxPositionals?: number;
  missingPositionalsMessage?: string;
}

interface CliCommandGroup {
  description: string;
  defaultSubcommand?: string;
  subcommands: Record<string, CliCommandDefinition>;
}

interface ParsedCli {
  command: string;
  subcommand?: string;
  positionals: string[];
  help: "top" | "command" | "group";
  definition?: CliCommandDefinition;
}

const CONFIG_OPTION = { name: "--config", value: "path", description: "Local runner config JSON path" } as const;
const VERBOSE_OPTION = { name: "--verbose", description: "Stream subprocess output" } as const;
const QUIET_OPTION = { name: "--quiet", description: "Suppress progress output on stderr" } as const;
const MODEL_SERVE_OPTIONS = [
  CONFIG_OPTION,
  { name: "--host", value: "host", description: "Bind host (localhost by default)" },
  { name: "--port", value: "port", description: "Bind port" },
  { name: "--device", value: "device", description: "cuda (pinned vLLM runtime)" },
  { name: "--max-tokens", value: "count", description: "Default response token limit" },
  { name: "--temperature", value: "number", description: "Default sampling temperature" },
  { name: "--top-p", value: "number", description: "Default nucleus sampling threshold" },
  { name: "--max-concurrent-requests", value: "count", description: "Maximum sequences in an upstream generation batch" },
  { name: "--context-length", value: "count", description: "Total context token budget (default 16384)" },
  { name: "--gpu-memory-utilization", value: "fraction", description: "vLLM GPU memory budget fraction (default 0.8)" },
  { name: "--spec", value: "path", description: "Behavior spec whose instructions are enforced" },
  { name: "--no-spec-prompt", description: "Do not enforce the stored behavior-spec prompt" },
  { name: "--allow-remote", description: "Allow a non-loopback bind" },
  { name: "--api-key-env", value: "name", description: "Environment variable containing a bearer token" },
  { name: "--print-command", description: "Validate and print the launch plan without starting" },
  { name: "--print-client-config", value: "client", description: "Print Pi models.json; does not start serving" },
] as const satisfies readonly CliOptionDefinition[];

const COMMAND_DEFINITIONS: Record<string, CliCommandDefinition> = {
  info: {
    usage: "tt info",
    description: "Show the installed TT Local version and runner status.",
    options: [],
    maxPositionals: 0,
  },
  init: {
    usage: "tt init [options]",
    description: "Create a local tunedtensor.json behavior spec.",
    options: [
      { name: "--name", value: "name", description: "Behavior spec name" },
      { name: "--engine", value: "engine", description: "adapter (default) or foundation" },
      { name: "--model", value: "model", description: "Base model ID" },
      { name: "--output", value: "path", description: "Output spec path" },
      { name: "--profile", value: "profile", description: "Write a durable runner config (spark)" },
      { name: "--config", value: "path", description: "Runner config path (written with --profile)" },
      { name: "--force", description: "Overwrite an existing output file" },
    ],
    maxPositionals: 0,
  },
  doctor: {
    usage: "tt doctor [tunedtensor.json] [--config path]",
    description: "Check the host and optional run input before starting work.",
    options: [CONFIG_OPTION],
    maxPositionals: 1,
  },
  hardware: {
    usage: "tt hardware [--config path] [--quick]",
    description: "Inventory this host and report what TT can train, fine-tune, or infer.",
    options: [
      CONFIG_OPTION,
      { name: "--quick", description: "Skip the bundled torch/uv runtime probe" },
    ],
    maxPositionals: 0,
  },
  validate: {
    usage: "tt validate [tunedtensor.json] [options]",
    description: "Validate a local behavior spec without executing it.",
    options: [CONFIG_OPTION],
    maxPositionals: 1,
  },
  run: {
    usage: "tt run [tunedtensor.json] [options]",
    description: "Run the baseline, fine-tuning, tuned evaluation, and report workflow.",
    options: [
      CONFIG_OPTION,
      { name: "--dry-run", description: "Write representative artifacts without GPU work" },
      VERBOSE_OPTION,
      QUIET_OPTION,
    ],
    maxPositionals: 1,
  },
  serve: {
    usage: "tt serve <model-id|active|base> [options]",
    description: "Serve a verified adapter, the active model, or the protected base.",
    options: MODEL_SERVE_OPTIONS,
    minPositionals: 1,
    maxPositionals: 1,
    missingPositionalsMessage: "serve requires <model-id>",
  },
};

const COMMAND_GROUPS: Record<string, CliCommandGroup> = {
  runs: {
    description: "Inspect locally stored runs.",
    defaultSubcommand: "list",
    subcommands: {
      list: { usage: "tt runs list [--config path]", description: "List local runs.", options: [CONFIG_OPTION], maxPositionals: 0 },
      get: { usage: "tt runs get <run-id> [--config path]", description: "Get a local run.", options: [CONFIG_OPTION], minPositionals: 1, maxPositionals: 1, missingPositionalsMessage: "runs get requires <run-id>" },
      events: { usage: "tt runs events <run-id> [--config path]", description: "List run events.", options: [CONFIG_OPTION], minPositionals: 1, maxPositionals: 1, missingPositionalsMessage: "runs events requires <run-id>" },
      report: { usage: "tt runs report <run-id> [--config path]", description: "Show the baseline-vs-tuned report, including deltas and regressions.", options: [CONFIG_OPTION], minPositionals: 1, maxPositionals: 1, missingPositionalsMessage: "runs report requires <run-id>" },
      compare: { usage: "tt runs compare <run-id-a> <run-id-b> [--config path]", description: "Compare two run reports.", options: [CONFIG_OPTION], minPositionals: 2, maxPositionals: 2, missingPositionalsMessage: "runs compare requires <run-id-a> <run-id-b>" },
    },
  },
  models: {
    description: "Inspect, verify, prefetch, or serve local models.",
    defaultSubcommand: "list",
    subcommands: {
      list: { usage: "tt models list [--config path]", description: "List local models.", options: [CONFIG_OPTION], maxPositionals: 0 },
      get: { usage: "tt models get <model-id> [--config path]", description: "Get a local model.", options: [CONFIG_OPTION], minPositionals: 1, maxPositionals: 1, missingPositionalsMessage: "models get requires <model-id>" },
      verify: {
        usage: "tt models verify <model-id-or-artifact-path> [--config path]",
        description: "Verify a stored model or manifested artifact path.",
        options: [CONFIG_OPTION],
        minPositionals: 1,
        maxPositionals: 1,
        missingPositionalsMessage: "models verify requires <model-id-or-artifact-path>",
      },
      prefetch: {
        usage: "tt models prefetch [tunedtensor.json] [options]",
        description: "Download the configured base model before a run.",
        options: [CONFIG_OPTION, VERBOSE_OPTION, QUIET_OPTION],
        maxPositionals: 1,
      },
      "verify-base": {
        usage: "tt models verify-base [tunedtensor.json] [options]",
        description: "Verify that the configured base-model snapshot is complete and locally available.",
        options: [CONFIG_OPTION, VERBOSE_OPTION, QUIET_OPTION],
        maxPositionals: 1,
      },
      active: {
        usage: "tt models active [--config path]",
        description: "Show the active adapter, or the protected base when none is active.",
        options: [CONFIG_OPTION],
        maxPositionals: 0,
      },
      activate: {
        usage: "tt models activate <model-id> [--config path]",
        description: "Activate a verified model whose general regression gate passed.",
        options: [CONFIG_OPTION],
        minPositionals: 1,
        maxPositionals: 1,
        missingPositionalsMessage: "models activate requires <model-id>",
      },
      rollback: {
        usage: "tt models rollback [--config path]",
        description: "Restore the previously active adapter or protected base.",
        options: [CONFIG_OPTION],
        maxPositionals: 0,
      },
      serve: {
        usage: "tt models serve <model-id|active|base> [options]",
        description: "Alias for `tt serve`.",
        options: MODEL_SERVE_OPTIONS,
        minPositionals: 1,
        maxPositionals: 1,
        missingPositionalsMessage: "models serve requires <model-id>",
      },
    },
  },
};

function hasHelpFlag(argv: string[]): boolean {
  return argv.includes("--help") || argv.includes("-h");
}

function printCommandHelp(definition: CliCommandDefinition): void {
  console.log(`Usage: ${definition.usage}\n\n${definition.description}`);
  if (definition.options.length > 0) {
    console.log("\nOptions:");
    for (const option of definition.options) {
      const label = option.value ? `${option.name} <${option.value}>` : option.name;
      console.log(`  ${label.padEnd(34)} ${option.description}`);
    }
  }
  console.log("  -h, --help                        Show help");
}

function printGroupHelp(command: string, group: CliCommandGroup): void {
  console.log(`Usage: tt ${command} <command> [options]\n\n${group.description}\n\nCommands:`);
  for (const [name, definition] of Object.entries(group.subcommands)) {
    console.log(`  ${name.padEnd(16)} ${definition.description}`);
  }
  console.log("\nRun `tt " + command + " <command> --help` for command-specific help.");
}

function parseCommandArguments(tokens: string[], definition: CliCommandDefinition): string[] {
  const options = new Map(definition.options.map((option) => [option.name, option]));
  const seen = new Set<string>();
  const positionals: string[] = [];
  let optionsEnded = false;

  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index]!;
    if (!optionsEnded && token === "--") {
      optionsEnded = true;
      continue;
    }
    if (!optionsEnded && token.startsWith("-")) {
      const equalsIndex = token.indexOf("=");
      const name = equalsIndex === -1 ? token : token.slice(0, equalsIndex);
      const inlineValue = equalsIndex === -1 ? undefined : token.slice(equalsIndex + 1);
      const option = options.get(name);
      if (!option) throw new Error(`Unknown option: ${name}`);
      if (seen.has(name)) throw new Error(`Option ${name} may only be specified once.`);
      seen.add(name);
      if (option.value) {
        if (inlineValue !== undefined) {
          if (!inlineValue) throw new Error(`Option ${name} requires a value.`);
          continue;
        }
        const value = tokens[index + 1];
        if (value === undefined || value.startsWith("-")) {
          throw new Error(`Option ${name} requires a value.`);
        }
        index += 1;
      } else if (inlineValue !== undefined) {
        throw new Error(`Option ${name} does not accept a value.`);
      }
      continue;
    }
    positionals.push(token);
  }

  if (positionals.length < (definition.minPositionals ?? 0)) {
    throw new Error(definition.missingPositionalsMessage ?? `Missing required argument. Usage: ${definition.usage}`);
  }
  if (definition.maxPositionals !== undefined && positionals.length > definition.maxPositionals) {
    throw new Error(`Too many arguments. Usage: ${definition.usage}`);
  }
  return positionals;
}

function parseCli(argv: string[]): ParsedCli {
  const command = argv[2] ?? "info";
  if (command === "--help" || command === "-h") {
    return { command: "info", positionals: [], help: "top" };
  }
  if (command === "--version" || command === "-V") {
    if (argv.length > 3) throw new Error(`${command} does not accept arguments.`);
    return { command, positionals: [], help: "command" };
  }
  if (command.startsWith("-")) throw new Error(`Unknown option: ${command}`);

  const definition = COMMAND_DEFINITIONS[command];
  if (definition) {
    if (hasHelpFlag(argv.slice(3))) {
      return { command, positionals: [], help: "command", definition };
    }
    return {
      command,
      positionals: parseCommandArguments(argv.slice(3), definition),
      help: "top",
      definition,
    };
  }

  const group = COMMAND_GROUPS[command];
  if (!group) throw new Error(`Unknown command: ${command}`);
  if (argv[3] === "--help" || argv[3] === "-h") {
    return { command, positionals: [], help: "group" };
  }

  const candidate = argv[3];
  let subcommand: string;
  let tokenStart: number;
  if (candidate && !candidate.startsWith("-")) {
    subcommand = candidate;
    tokenStart = 4;
  } else if (group.defaultSubcommand) {
    subcommand = group.defaultSubcommand;
    tokenStart = 3;
  } else {
    throw new Error(`${command} requires a subcommand. Run 'tt ${command} --help'.`);
  }
  const subcommandDefinition = group.subcommands[subcommand];
  if (!subcommandDefinition) throw new Error(`Unknown ${command} command: ${subcommand}`);
  if (hasHelpFlag(argv.slice(tokenStart))) {
    return { command, subcommand, positionals: [], help: "command", definition: subcommandDefinition };
  }
  return {
    command,
    subcommand,
    positionals: parseCommandArguments(argv.slice(tokenStart), subcommandDefinition),
    help: "top",
    definition: subcommandDefinition,
  };
}

function readNumberOption(argv: string[], name: string): number | undefined {
  const value = readOption(argv, name);
  return value ? Number(value) : undefined;
}

interface LocalConfigSelection {
  config: LocalRunnerConfig;
  path?: string;
}

async function selectedConfigPath(argv: string[], adjacentTo?: string): Promise<string | undefined> {
  const explicitPath = readOption(argv, "--config");
  if (explicitPath) return resolve(explicitPath);
  const candidate = join(adjacentTo ? dirname(resolve(adjacentTo)) : cwd(), "local-runner.json");
  const metadata = await stat(candidate).catch(() => null);
  return metadata?.isFile() ? candidate : undefined;
}

async function configSelectionFromArgv(argv: string[], adjacentTo?: string): Promise<LocalConfigSelection> {
  const path = await selectedConfigPath(argv, adjacentTo);
  return {
    config: await loadLocalRunnerConfig(path),
    ...(path ? { path } : {}),
  };
}

async function configFromArgv(argv: string[], adjacentTo?: string): Promise<LocalRunnerConfig> {
  return (await configSelectionFromArgv(argv, adjacentTo)).config;
}

async function loadCliBehaviorSpec(inputPath: string, runId?: string) {
  const input = await loadLocalRunInput(inputPath, {
    ...(runId ? { runId } : {}),
  });
  if (input.kind === "foundation-spec") {
    throw new Error(
      `This spec uses engine "foundation". Use \`tt pipeline run --spec ${input.path}\`; the legacy adapter runner cannot execute foundation specs.`,
    );
  }
  if (input.kind !== "spec") {
    throw new Error(`TT Local CLI expects a tunedtensor.json behavior spec, not a full run request: ${input.path}`);
  }
  return input;
}

function printJson(value: unknown): void {
  console.log(JSON.stringify(value, null, 2));
}

function shortValue(value: unknown): string | null {
  if (value === undefined || value === null) return null;
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (Array.isArray(value)) return `[${value.length} items]`;
  return null;
}

function formatEvent(event: LocalRunProgressEvent): string {
  const detailText = Object.entries(event.details ?? {})
    .filter(([key]) => key !== "metrics")
    .map(([key, value]) => {
      const formatted = key === "command" && Array.isArray(value)
        ? value.join(" ")
        : shortValue(value);
      return formatted ? `${key}=${formatted}` : null;
    })
    .filter((value): value is string => Boolean(value))
    .slice(0, 5)
    .join(" ");
  return sanitizeLogLine(`[tt-local] ${event.stage}: ${event.message}${detailText ? ` (${detailText})` : ""}`);
}

function createConsoleReporter(options: { verbose: boolean; quiet: boolean }): LocalRunReporter | undefined {
  if (options.quiet) return undefined;
  let lastLogLine = "";
  return {
    verbose: options.verbose,
    onEvent(event) {
      process.stderr.write(`${formatEvent(event)}\n`);
    },
    onLog(log) {
      const line = sanitizeLogLine(`[tt-local] ${log.stage}${log.stream ? ` ${log.stream}` : ""}: ${log.message}`);
      // tqdm redraws the same progress line several times per step; collapse
      // consecutive duplicates so --verbose output stays readable.
      if (line === lastLogLine) return;
      lastLogLine = line;
      process.stderr.write(`${line}\n`);
    },
  };
}

async function verifyStoredModel(model: LocalModelRecord, config: LocalRunnerConfig): Promise<{
  manifest_path: string;
  integrity: Awaited<ReturnType<typeof assertArtifactManifest>>;
  artifact: Awaited<ReturnType<typeof assertUsableModelArtifact>>;
  contract: unknown;
}> {
  const manifestPath = join(model.artifact_dir, "artifact-manifest.json");
  const integrity = await assertArtifactManifest(manifestPath, {
    requiredPaths: ["stage-metadata.json", "training-report.json"],
    scopeToRequired: true,
    verifyModel: true,
  });
  const manifest = JSON.parse(await readFile(manifestPath, "utf8")) as {
    model?: {
      artifact_root?: unknown;
      base_model?: unknown;
      format?: unknown;
      framework?: unknown;
      servable?: unknown;
      base_model_artifact_uri?: unknown;
      base_model_fingerprint?: unknown;
    };
  };
  if (!manifest.model) {
    throw new Error(`Artifact manifest does not contain a model contract: ${manifestPath}`);
  }
  const artifact = await assertUsableModelArtifact(model.artifact_uri);
  if (
    typeof manifest.model.base_model_artifact_uri === "string"
    && typeof manifest.model.base_model_fingerprint === "string"
  ) {
    const actualBaseFingerprint = await fingerprintLocalBaseModel(
      manifest.model.base_model_artifact_uri,
      typeof manifest.model.base_model === "string"
        ? manifest.model.base_model
        : undefined,
      config.paths.modelCache,
    );
    if (actualBaseFingerprint !== manifest.model.base_model_fingerprint) {
      throw new Error("Recorded local base-model content no longer matches the model artifact contract.");
    }
  }
  if (
    typeof manifest.model.artifact_root !== "string"
    || resolve(manifest.model.artifact_root) !== resolve(artifact.path)
  ) {
    throw new Error("Stored model record does not match the artifact covered by its manifest.");
  }
  if (manifest.model.base_model !== model.base_model) {
    throw new Error("Stored model base model does not match its artifact manifest.");
  }
  return {
    manifest_path: manifestPath,
    integrity,
    artifact,
    contract: manifest.model,
  };
}

async function verifyActivationEvidence(model: LocalModelRecord): Promise<void> {
  const generalBaseline = join(model.artifact_dir, "general-baseline-eval.json");
  const generalCandidate = join(model.artifact_dir, "general-candidate-eval.json");
  const [hasBaseline, hasCandidate] = await Promise.all([
    stat(generalBaseline).then((metadata) => metadata.isFile(), () => false),
    stat(generalCandidate).then((metadata) => metadata.isFile(), () => false),
  ]);
  if (!hasBaseline || !hasCandidate) {
    throw new Error(
      "This model cannot be activated because the run did not include a general-regression suite. "
      + "Add evaluation.generalRegression to local-runner.json with a held-out dataset, re-run training, then activate. "
      + "Until then, serve the adapter with `tt serve local-<run-id>`.",
    );
  }
  await assertArtifactManifest(join(model.artifact_dir, "artifact-manifest.json"), {
    requiredPaths: [
      "run-report.json",
      "general-baseline-eval.json",
      "general-candidate-eval.json",
    ],
    scopeToRequired: true,
    verifyModel: true,
  });
}

async function verifyModelArtifactPath(input: string, config: LocalRunnerConfig): Promise<{
  manifest_path: string;
  integrity: Awaited<ReturnType<typeof assertArtifactManifest>>;
  artifact: Awaited<ReturnType<typeof assertUsableModelArtifact>>;
  contract: unknown;
}> {
  const inputPath = resolve(input);
  let manifestPath: string | undefined;
  let artifactUri = inputPath;
  if (basename(inputPath) === "artifact-manifest.json") {
    manifestPath = inputPath;
    const raw = JSON.parse(await readFile(inputPath, "utf8")) as { model?: { artifact_root?: unknown } };
    if (typeof raw.model?.artifact_root !== "string") {
      throw new Error(`Artifact manifest does not contain a model contract: ${inputPath}`);
    }
    artifactUri = raw.model.artifact_root;
  } else {
    const metadata = await stat(inputPath).catch(() => null);
    if (!metadata) throw new Error(`Model not found and artifact path does not exist: ${input}`);
    let current = metadata.isDirectory() ? inputPath : dirname(inputPath);
    for (let depth = 0; depth < 8; depth += 1) {
      const candidate = join(current, "artifact-manifest.json");
      const raw = await readFile(candidate, "utf8").catch(() => null);
      if (raw) {
        const parsed = JSON.parse(raw) as { model?: { artifact_root?: unknown } };
        if (
          typeof parsed.model?.artifact_root === "string"
          && resolve(parsed.model.artifact_root) === resolve(inputPath)
        ) {
          manifestPath = candidate;
          break;
        }
      }
      const parent = dirname(current);
      if (parent === current) break;
      current = parent;
    }
  }
  if (!manifestPath) {
    throw new Error(`No artifact manifest covering model path ${inputPath} was found in its parent run directory.`);
  }
  const integrity = await assertArtifactManifest(manifestPath, {
    requiredPaths: ["stage-metadata.json", "training-report.json"],
    scopeToRequired: true,
    verifyModel: true,
  });
  const manifest = JSON.parse(await readFile(manifestPath, "utf8")) as { model?: Record<string, unknown> };
  if (!manifest.model) throw new Error(`Artifact manifest does not contain a model contract: ${manifestPath}`);
  const artifact = await assertUsableModelArtifact(artifactUri);
  if (
    typeof manifest.model.base_model_artifact_uri === "string"
    && typeof manifest.model.base_model_fingerprint === "string"
  ) {
    const actualBaseFingerprint = await fingerprintLocalBaseModel(
      manifest.model.base_model_artifact_uri,
      typeof manifest.model.base_model === "string"
        ? manifest.model.base_model
        : undefined,
      config.paths.modelCache,
    );
    if (actualBaseFingerprint !== manifest.model.base_model_fingerprint) {
      throw new Error("Recorded local base-model content no longer matches the model artifact contract.");
    }
  }
  if (
    typeof manifest.model.artifact_root !== "string"
    || resolve(manifest.model.artifact_root) !== resolve(artifact.path)
  ) {
    throw new Error("Model path does not match the artifact covered by its manifest.");
  }
  return { manifest_path: manifestPath, integrity, artifact, contract: manifest.model };
}

async function modelSystemPrompt(args: {
  argv: string[];
  model: LocalModelRecord;
  store: LocalStore;
}): Promise<string | undefined> {
  const specPath = readOption(args.argv, "--spec");
  if (specPath && hasFlag(args.argv, "--no-spec-prompt")) {
    throw new Error("Use only one of --spec or --no-spec-prompt.");
  }
  if (hasFlag(args.argv, "--no-spec-prompt")) return undefined;

  let spec: SpecSnapshot;
  if (specPath) {
    const input = JSON.parse(await readFile(resolve(specPath), "utf8")) as unknown;
    const local = localBehaviorSpecFileSchema.safeParse(input);
    if (!local.success) {
      throw new Error(`--spec must contain a tunedtensor.json behavior spec: ${resolve(specPath)}`);
    }
    if (isFoundationSpecFile(local.data)) {
      throw new Error("Foundation specs have no Hugging Face base model. `tt serve` cannot host them yet.");
    }
    spec = local.data;
  } else {
    const runRequestPath = join(args.store.paths.runsDir, args.model.run_id, "request.json");
    let persistedRequest: FineTuneRunRequest | null = null;
    try {
      persistedRequest = fineTuneRunRequestSchema.parse(JSON.parse(await readFile(runRequestPath, "utf8")));
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
    spec = persistedRequest?.spec_snapshot
      ?? (await args.store.getSpec(args.model.behavior_spec_id)).spec;
  }
  if (spec.base_model !== args.model.base_model) {
    throw new Error(
      `Behavior spec base model ${spec.base_model} does not match stored model base ${args.model.base_model}.`,
    );
  }
  const prompt = buildSystemMessage(spec);
  const metadata = JSON.parse(
    await readFile(join(args.model.artifact_dir, "stage-metadata.json"), "utf8"),
  ) as { system_prompt_sha256?: unknown };
  const promptHash = createHash("sha256").update(prompt).digest("hex");
  if (metadata.system_prompt_sha256 !== promptHash) {
    throw new Error(
      "Behavior spec instructions do not match the prompt fingerprint used for this trained model. "
      + "Pass the original --spec, or use --no-spec-prompt only when this change is intentional.",
    );
  }
  return prompt;
}

function modelServeDevice(argv: string[]): "cpu" | "cuda" | undefined {
  const value = readOption(argv, "--device");
  if (value === undefined || value === "cpu" || value === "cuda") {
    return value;
  }
  throw new Error(`--device must be cuda or cpu; got: ${value}`);
}

function printServeClientConfig(argv: string[], launch: LocalModelServerLaunch): boolean {
  const client = readOption(argv, "--print-client-config");
  if (client === undefined) return false;
  if (client !== "pi") throw new Error("--print-client-config must be pi.");
  if (hasFlag(argv, "--print-command")) throw new Error("Use only one of --print-command or --print-client-config.");
  const apiKeyEnv = readOption(argv, "--api-key-env");
  if (apiKeyEnv && !/^[A-Za-z_][A-Za-z0-9_]*$/.test(apiKeyEnv)) {
    throw new Error("--api-key-env must be a plain environment variable name for Pi config.");
  }
  const url = new URL(launch.url);
  if (url.hostname === "0.0.0.0") url.hostname = "127.0.0.1";
  if (url.hostname === "[::]") url.hostname = "[::1]";
  url.pathname = "/v1";
  printJson({
    providers: {
      "tt-local": {
        baseUrl: url.toString(),
        api: "openai-completions",
        apiKey: apiKeyEnv ? "${" + apiKeyEnv + "}" : "tt-local",
        compat: {
          supportsStore: false,
          supportsDeveloperRole: false,
          supportsReasoningEffort: false,
          supportsUsageInStreaming: true,
          maxTokensField: "max_tokens",
        },
        models: [...new Set([launch.modelName, `base:${launch.env.TT_BASE_MODEL}`])].map((id) => ({
          id,
          name: id,
          // This endpoint exports ordinary text/tool calls, not reasoning-effort controls.
          reasoning: false,
          input: ["text"],
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
          contextWindow: Number(launch.env.TT_CONTEXT_LENGTH),
          maxTokens: Number(launch.env.TT_MAX_TOKENS),
        })),
      },
    },
  });
  return true;
}

async function serveStoredModelFromCli(args: {
  argv: string[];
  modelId: string;
  config: LocalRunnerConfig;
}): Promise<void> {
  const store = createLocalStore(args.config.storeRoot);
  const model = await store.getModel(args.modelId);
  const verified = await verifyStoredModel(model, args.config);
  const launch = buildLocalModelServerLaunch({
    model,
    config: args.config,
    options: {
      host: readOption(args.argv, "--host"),
      port: readNumberOption(args.argv, "--port"),
      device: modelServeDevice(args.argv),
      maxTokens: readNumberOption(args.argv, "--max-tokens"),
      temperature: readNumberOption(args.argv, "--temperature"),
      topP: readNumberOption(args.argv, "--top-p"),
      maxConcurrentRequests: readNumberOption(args.argv, "--max-concurrent-requests"),
      contextLength: readNumberOption(args.argv, "--context-length"),
      gpuMemoryUtilization: readNumberOption(args.argv, "--gpu-memory-utilization"),
      systemPrompt: await modelSystemPrompt({ argv: args.argv, model, store }),
      allowRemote: hasFlag(args.argv, "--allow-remote"),
      apiKeyEnv: readOption(args.argv, "--api-key-env"),
      baseModelRevision: (() => {
        const contract = verified.contract as { base_model_revision?: unknown };
        return typeof contract.base_model_revision === "string" ? contract.base_model_revision : undefined;
      })(),
      baseModelArtifactUri: (() => {
        const contract = verified.contract as { base_model_artifact_uri?: unknown };
        return typeof contract.base_model_artifact_uri === "string" ? contract.base_model_artifact_uri : undefined;
      })(),
    },
  });
  if (printServeClientConfig(args.argv, launch)) return;
  if (hasFlag(args.argv, "--print-command")) {
    printJson({
      ok: true,
      model_id: model.id,
      url: launch.url,
      command: launch.displayCommand,
      artifact_path: launch.artifactPath,
      manifest_path: verified.manifest_path,
      integrity: verified.integrity,
    });
    return;
  }
  process.stderr.write(`[tt-local] verified ${verified.integrity.checked} artifact file(s)\n`);
  await serveLocalModel(launch);
}

function serveOptionsFromArgv(argv: string[]) {
  return {
    host: readOption(argv, "--host"),
    port: readNumberOption(argv, "--port"),
    device: modelServeDevice(argv),
    maxTokens: readNumberOption(argv, "--max-tokens"),
    temperature: readNumberOption(argv, "--temperature"),
    topP: readNumberOption(argv, "--top-p"),
    maxConcurrentRequests: readNumberOption(argv, "--max-concurrent-requests"),
    contextLength: readNumberOption(argv, "--context-length"),
    gpuMemoryUtilization: readNumberOption(argv, "--gpu-memory-utilization"),
    allowRemote: hasFlag(argv, "--allow-remote"),
    apiKeyEnv: readOption(argv, "--api-key-env"),
  };
}

async function serveBaseModelFromCli(args: {
  argv: string[];
  config: LocalRunnerConfig;
}): Promise<void> {
  const explicitSpecPath = readOption(args.argv, "--spec");
  if (explicitSpecPath && hasFlag(args.argv, "--no-spec-prompt")) {
    throw new Error("Use only one of --spec or --no-spec-prompt.");
  }
  const defaultSpecPath = resolve(DEFAULT_LOCAL_SPEC_PATH);
  const specPath = explicitSpecPath
    ? resolve(explicitSpecPath)
    : (await stat(defaultSpecPath).catch(() => null))?.isFile()
      ? defaultSpecPath
      : undefined;
  const spec = specPath
    ? localBehaviorSpecFileSchema.parse(
      JSON.parse(await readFile(specPath, "utf8")) as unknown,
    )
    : undefined;
  if (spec && isFoundationSpecFile(spec)) {
    throw new Error("Foundation specs have no Hugging Face base model. `tt serve` cannot host them yet.");
  }
  let baseModel = spec?.base_model;
  if (!baseModel) {
    const store = createLocalStore(args.config.storeRoot);
    const active = await getActiveModel(store);
    const referenceModel = active.model
      ?? (active.pointer?.previous_model_id
        ? await store.getModel(active.pointer.previous_model_id)
        : null);
    baseModel = referenceModel?.base_model;
  }
  if (!baseModel) {
    throw new Error(
      "Cannot identify the protected base model. Pass --spec <tunedtensor.json> "
      + "or activate a model first.",
    );
  }
  const launch = buildLocalBaseModelServerLaunch({
    baseModel,
    config: args.config,
    options: {
      ...serveOptionsFromArgv(args.argv),
      systemPrompt: explicitSpecPath && !hasFlag(args.argv, "--no-spec-prompt")
        ? buildSystemMessage(spec!)
        : undefined,
      baseModelRevision: spec?.hyperparameters?.base_model_revision ?? defaultBaseModelRevision(baseModel),
    },
  });
  if (printServeClientConfig(args.argv, launch)) return;
  if (hasFlag(args.argv, "--print-command")) {
    printJson({
      ok: true,
      model_id: null,
      base_model: baseModel,
      url: launch.url,
      command: launch.displayCommand,
    });
    return;
  }
  await serveLocalModel(launch);
}

async function serveModelTargetFromCli(args: {
  argv: string[];
  target: string;
  config: LocalRunnerConfig;
}): Promise<void> {
  if (args.target === "base") {
    return serveBaseModelFromCli(args);
  }
  if (args.target === "active") {
    const store = createLocalStore(args.config.storeRoot);
    const active = await getActiveModel(store);
    if (!active.model) {
      throw new Error(
        "No adapter is active. `tt serve active` would serve the protected base model. "
        + "Activate a verified adapter first with `tt models activate <model-id>`, or pass `tt serve base` explicitly.",
      );
    }
    return serveStoredModelFromCli({
      argv: args.argv,
      modelId: active.model.id,
      config: args.config,
    });
  }
  return serveStoredModelFromCli({
    argv: args.argv,
    modelId: args.target,
    config: args.config,
  });
}

async function main(argv: string[]): Promise<void> {
  const cli = parseCli(argv);
  const command = cli.command;

  if (command === "--version" || command === "-V") {
    console.log(TT_LOCAL_VERSION);
    return;
  }
  if (cli.help === "top" && (argv[2] === "--help" || argv[2] === "-h")) {
    printHelp();
    return;
  }
  if (cli.help === "group") {
    printGroupHelp(command, COMMAND_GROUPS[command]!);
    return;
  }
  if (cli.help === "command" && cli.definition) {
    printCommandHelp(cli.definition);
    return;
  }

  if (command === "info") {
    const info = getLocalRunnerInfo();
    console.log(`${info.name}: ${info.description}`);
    console.log(`Version: ${info.version}`);
    console.log(`Status: ${info.status}`);
    return;
  }

  if (command === "doctor") {
    const inputPath = resolve(cli.positionals[0] ?? DEFAULT_LOCAL_SPEC_PATH);
    const configSelection = await configSelectionFromArgv(argv, inputPath);
    const hasDefaultSpec = cli.positionals[0]
      ? true
      : Boolean((await stat(inputPath).catch(() => null))?.isFile());
    let request: FineTuneRunRequest | undefined;
    let foundationSpec: import("./contracts.js").LocalFoundationSpecFile | undefined;
    if (hasDefaultSpec) {
      const input = await loadLocalRunInput(inputPath);
      if (input.kind === "foundation-spec") {
        foundationSpec = input.spec;
      } else if (input.kind === "spec") {
        request = input.request;
      } else {
        throw new Error(`TT Local CLI expects a tunedtensor.json behavior spec, not a full run request: ${input.path}`);
      }
    }
    const checks = await runDoctor(configSelection.config, request, foundationSpec);
    const ok = checks.every((check) => check.ok);
    printJson({
      ok,
      config_path: configSelection.path ?? null,
      checks,
    });
    try {
      await assessHardware({
        config: configSelection.config,
        quick: true,
      });
    } catch {
      // Status/context still work without a snapshot; doctor remains the gate.
      // A quick write fills an empty cache and will not replace a fresh full probe.
    }
    if (!ok) process.exitCode = 1;
    return;
  }

  if (command === "hardware") {
    const configSelection = await configSelectionFromArgv(argv);
    const report = await assessHardware({
      config: configSelection.config,
      quick: hasFlag(argv, "--quick"),
    });
    printJson({
      ...report,
      config_path: configSelection.path ?? null,
    });
    return;
  }

  if (command === "init") {
    const outputPath = resolve(readOption(argv, "--output") ?? DEFAULT_LOCAL_SPEC_PATH);
    const profile = readOption(argv, "--profile");
    if (profile !== undefined && profile !== "spark") {
      throw new Error(`--profile must be spark, got: ${profile}`);
    }
    const engine = readOption(argv, "--engine") ?? "adapter";
    if (engine !== "adapter" && engine !== "foundation") {
      throw new Error(`--engine must be adapter or foundation, got: ${engine}`);
    }
    if (engine === "foundation" && readOption(argv, "--model")) {
      throw new Error("Foundation specs do not take --model; they train a tokenizer and GPT from scratch.");
    }
    const spec = await initLocalSpecFile({
      outputPath,
      name: readOption(argv, "--name") ?? (engine === "foundation" ? "Foundation chat model" : "Local Tuned Tensor Spec"),
      engine,
      baseModel: engine === "foundation" ? undefined : (readOption(argv, "--model") ?? "Qwen/Qwen3.5-2B"),
      force: hasFlag(argv, "--force"),
    });
    const configPath = profile
      ? resolve(readOption(argv, "--config") ?? resolve(dirname(outputPath), "local-runner.json"))
      : await selectedConfigPath(argv, outputPath);
    if (profile) {
      await initLocalRunnerConfigFile({
        outputPath: configPath!,
        profile,
        force: hasFlag(argv, "--force"),
      });
    } else if (configPath) {
      await loadLocalRunnerConfig(configPath);
    }
    printJson({
      ok: true,
      path: outputPath,
      id: spec.id,
      name: spec.name,
      engine: spec.engine ?? "adapter",
      base_model: isFoundationSpecFile(spec) ? null : spec.base_model,
      config_path: configPath ?? null,
    });
    return;
  }

  if (command === "validate") {
    const inputPath = resolve(cli.positionals[0] ?? DEFAULT_LOCAL_SPEC_PATH);
    const loaded = await loadLocalRunInput(inputPath);
    if (loaded.kind === "foundation-spec") {
      assertFoundationSpecReady(loaded.spec);
      printJson({
        ok: true,
        input_path: loaded.path,
        engine: "foundation",
        pipeline: "tt pipeline run --spec",
      });
      return;
    }
    if (loaded.kind !== "spec") {
      throw new Error(`TT Local CLI expects a tunedtensor.json behavior spec, not a full run request: ${loaded.path}`);
    }
    const input = loaded;
    assertLocalRunInputReady(input.request);
    const configSelection = await configSelectionFromArgv(argv, inputPath);
    const validated = await validateLocalFineTuneInput({
      request: input.request,
      config: configSelection.config,
    });
    const request = validated.request;
    const config = validated.config;
    printJson({
      ok: true,
      input_path: input.path,
      config_path: configSelection.path ?? null,
      behavior_spec_id: request.behavior_spec_id,
      base_model: request.spec_snapshot.base_model,
      dataset_format: request.dataset_prebuilt?.format ?? null,
      artifact_root: config.artifactRoot,
      store_root: config.storeRoot,
      dry_run: config.dryRun,
    });
    return;
  }

  if (command === "run") {
    const inputPath = resolve(cli.positionals[0] ?? DEFAULT_LOCAL_SPEC_PATH);
    const configSelection = await configSelectionFromArgv(argv, inputPath);
    const configInput = configSelection.config;
    const config = localRunnerConfigSchema.parse({
      ...configInput,
      dryRun: hasFlag(argv, "--dry-run") ? true : configInput.dryRun,
    });
    const input = await loadCliBehaviorSpec(inputPath);
    assertLocalRunInputReady(input.request);
    const validated = await validateLocalFineTuneInput({
      request: input.request,
      config,
    });
    let request = validated.request;
    const reporter = createConsoleReporter({
      verbose: hasFlag(argv, "--verbose"),
      quiet: hasFlag(argv, "--quiet"),
    });
    if (
      !config.dryRun
      && !config.paths.baseModel
      && !request.hyperparameters.base_model_revision
    ) {
      const prefetch = await prefetchBaseModel({
        request,
        config,
        reporter,
      });
      if (!prefetch.snapshot_revision) {
        throw new Error("Base-model prefetch did not return an immutable snapshot revision.");
      }
      request = fineTuneRunRequestSchema.parse({
        ...request,
        hyperparameters: {
          ...request.hyperparameters,
          base_model_revision: prefetch.snapshot_revision,
        },
      });
    }
    const result = await runLocalFineTune({
      request,
      config,
      reporter,
    });
    printJson({
      status: result.report.status,
      run_id: result.report.run_id,
      behavior_spec_id: result.report.behavior_spec_id,
      report_path: result.reportPath,
      artifact_dir: result.artifactDir,
      ...(!config.dryRun ? {
        model_id: `local-${result.report.run_id}`,
        fine_tuned_model_id: result.report.fine_tuned_model_id,
      } : {}),
      training_log: result.report.training.log_uri,
      baseline_eval: result.report.artifact_uris.baseline_eval,
      candidate_eval: result.report.artifact_uris.candidate_eval,
      comparison: result.report.comparison,
      general_regression: result.report.general_regression,
    });
    return;
  }

  if (command === "serve") {
    const config = await configFromArgv(argv);
    await serveModelTargetFromCli({
      argv,
      target: cli.positionals[0]!,
      config,
    });
    return;
  }

  if (command === "runs") {
    const subcommand = cli.subcommand!;
    const config = await configFromArgv(argv);
    const store = createLocalStore(config.storeRoot);
    if (subcommand === "list") return printJson(await store.listRuns());
    if (subcommand === "get") {
      const id = cli.positionals[0];
      if (!id) throw new Error("runs get requires <run-id>");
      return printJson(await store.getRun(id));
    }
    if (subcommand === "events") {
      const id = cli.positionals[0];
      if (!id) throw new Error("runs events requires <run-id>");
      return printJson(await store.getRunEvents(id));
    }
    if (subcommand === "report") {
      const id = cli.positionals[0];
      if (!id) throw new Error("runs report requires <run-id>");
      return printJson(await store.getRunReport(id));
    }
    if (subcommand === "compare") {
      const idA = cli.positionals[0];
      const idB = cli.positionals[1];
      if (!idA || !idB) throw new Error("runs compare requires <run-id-a> <run-id-b>");
      const [reportA, reportB] = await Promise.all([
        store.getRunReport(idA),
        store.getRunReport(idB),
      ]);
      return printJson(compareRuns(reportA, reportB));
    }
    throw new Error(`Unknown runs command: ${subcommand}`);
  }

  if (command === "models") {
    const subcommand = cli.subcommand!;
    const modelInputPath = subcommand === "prefetch" || subcommand === "verify-base"
      ? resolve(cli.positionals[0] ?? DEFAULT_LOCAL_SPEC_PATH)
      : undefined;
    const config = await configFromArgv(argv, modelInputPath);
    const store = createLocalStore(config.storeRoot);
    if (subcommand === "list") return printJson(await store.listModels());
    if (subcommand === "get") {
      const id = cli.positionals[0];
      if (!id) throw new Error("models get requires <model-id>");
      const model = await store.getModel(id);
      const verified = await verifyStoredModel(model, config);
      return printJson({
        ok: true,
        model,
        ...verified,
      });
    }
    if (subcommand === "verify") {
      const id = cli.positionals[0];
      if (!id) throw new Error("models verify requires <model-id-or-artifact-path>");
      if (await stat(resolve(id)).then(() => true, () => false)) {
        return printJson({ ok: true, model: null, ...await verifyModelArtifactPath(id, config) });
      }
      const model = await store.getModel(id);
      const verified = await verifyStoredModel(model, config);
      return printJson({ ok: true, model, ...verified });
    }
    if (subcommand === "active") {
      const active = await getActiveModel(store);
      return printJson({
        active: active.model?.id ?? "base",
        pointer: active.pointer,
        model: active.model,
      });
    }
    if (subcommand === "activate") {
      const id = cli.positionals[0];
      if (!id) throw new Error("models activate requires <model-id>");
      const model = await store.getModel(id);
      const verified = await verifyStoredModel(model, config);
      await verifyActivationEvidence(model);
      const pointer = await activateModel(store, model.id);
      return printJson({
        active: pointer.model_id,
        pointer,
        manifest_path: verified.manifest_path,
        integrity: verified.integrity,
      });
    }
    if (subcommand === "rollback") {
      const current = await getActiveModel(store);
      if (current.pointer?.previous_model_id) {
        const previous = await store.getModel(current.pointer.previous_model_id);
        await verifyStoredModel(previous, config);
        await verifyActivationEvidence(previous);
      }
      const pointer = await rollbackActiveModel(store);
      return printJson({ active: pointer.model_id ?? "base", pointer });
    }
    if (subcommand === "prefetch" || subcommand === "verify-base") {
      const input = await loadCliBehaviorSpec(modelInputPath!);
      const report = await prefetchBaseModel({
        request: input.request,
        config,
        localOnly: subcommand === "verify-base",
        reporter: createConsoleReporter({
          verbose: hasFlag(argv, "--verbose"),
          quiet: hasFlag(argv, "--quiet"),
        }),
      });
      return printJson({
        ...report,
        input_path: input.path,
      });
    }
    if (subcommand === "serve") {
      const id = cli.positionals[0];
      if (!id) throw new Error("models serve requires <model-id>");
      await serveModelTargetFromCli({ argv, target: id, config });
      return;
    }
    throw new Error(`Unknown models command: ${subcommand}`);
  }

  console.error(`Unknown command: ${command}`);
  process.exitCode = 1;
}

function isCliEntrypoint(): boolean {
  if (!process.argv[1]) return false;
  return realpathSync(fileURLToPath(import.meta.url)) === realpathSync(process.argv[1]);
}

if (isCliEntrypoint()) {
  main(process.argv).catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
