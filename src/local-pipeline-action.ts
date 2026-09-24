import { createHash, randomUUID } from "node:crypto";
import { lstat, mkdtemp, readFile, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import {
  canonicalJson,
  createExecutionPlan,
  pipelineForRunInput,
  type ExecutionPlan,
  type Pipeline,
} from "./pipeline.js";
import {
  parseLocalRunInput,
  type LocalRunInput,
} from "./local-runtime/local-project.js";
import { parseLocalRunnerConfig } from "./local-runtime/orchestrator.js";
import { configForSpec } from "./local-runtime/project-workflow.js";
import type { LocalRunnerConfig } from "./local-runtime/contracts.js";
import { canonicalWorkspace, fingerprintWorkspace } from "./local-spec-workspace.js";

const DEFAULT_SPEC_PATH = "tunedtensor.json";
const MAX_SPEC_BYTES = 2_000_000;
const MAX_CONFIG_BYTES = 1_000_000;

export interface PreparedLocalPipelineAction {
  pipeline: Pipeline;
  plan: ExecutionPlan;
  specPath: string;
  specSha256: string;
  dryRun: boolean;
  engine: "adapter" | "foundation";
  resolvedSpec: Exclude<LocalRunInput, { kind: "request" }>[
    "spec"
  ];
  configPath?: string;
  configSha256?: string;
  resolvedConfig?: LocalRunnerConfig;
  workspaceRoot: string;
  workspaceFingerprint: string;
}

export interface LocalPipelineCommandResult {
  exitCode: number | null;
  signal?: NodeJS.Signals | null;
}

export type LocalPipelineCommandRunner = (
  args: string[],
  options: { cwd: string; signal?: AbortSignal },
) => Promise<LocalPipelineCommandResult>;

export interface ValidatePreparedLocalPipelineActionArgs {
  workspaceRoot: string;
  pipeline: unknown;
  specPath: string;
  expectedSpecSha256: string;
  expectedWorkspaceFingerprint: string;
  expectedConfigPath?: string;
  expectedConfigSha256?: string;
  dryRun: boolean;
}

export class LocalPipelineActionError extends Error {
  constructor(message: string, cause?: unknown) {
    super(message, { cause });
    this.name = "LocalPipelineActionError";
  }
}

export class LocalPipelineWorkspaceMismatchError extends LocalPipelineActionError {
  constructor(message: string) {
    super(message);
    this.name = "LocalPipelineWorkspaceMismatchError";
  }
}

export function isKnownLocalPipelineFailure(
  error: unknown,
): error is LocalPipelineActionError {
  return error instanceof LocalPipelineActionError;
}

function isWithin(root: string, candidate: string): boolean {
  const relation = relative(root, candidate);
  return relation === ""
    || (!isAbsolute(relation) && relation !== ".." && !relation.startsWith(`..${sep}`));
}

async function resolveWorkspaceSpec(
  workspaceRoot: string,
  requestedPath: string,
): Promise<{
  workspaceRoot: string;
  workspaceFingerprint: string;
  path: string;
  displayPath: string;
  source: Buffer;
}> {
  if (!requestedPath || isAbsolute(requestedPath)) {
    throw new Error("Pipeline spec_path must be a relative file inside the current workspace.");
  }
  const canonicalRoot = await canonicalWorkspace(workspaceRoot);
  const lexicalPath = resolve(canonicalRoot, requestedPath);
  if (!isWithin(canonicalRoot, lexicalPath)) {
    throw new Error("Pipeline spec_path must stay inside the current workspace.");
  }
  const fileInfo = await lstat(lexicalPath);
  if (fileInfo.isSymbolicLink() || !fileInfo.isFile()) {
    throw new Error("Pipeline spec_path must name a regular file, not a directory or symlink.");
  }
  if (fileInfo.size > MAX_SPEC_BYTES) {
    throw new Error(`Pipeline spec_path exceeds ${MAX_SPEC_BYTES} bytes.`);
  }
  const physicalPath = await realpath(lexicalPath);
  if (!isWithin(canonicalRoot, physicalPath)) {
    throw new Error("Pipeline spec_path resolves outside the current workspace.");
  }
  const source = await readFile(physicalPath);
  if (source.byteLength > MAX_SPEC_BYTES) {
    throw new Error(`Pipeline spec_path exceeds ${MAX_SPEC_BYTES} bytes.`);
  }
  const workspaceRelative = relative(canonicalRoot, physicalPath).split(sep).join("/");
  return {
    workspaceRoot: canonicalRoot,
    workspaceFingerprint: await fingerprintWorkspace(canonicalRoot),
    path: physicalPath,
    displayPath: `./${workspaceRelative}`,
    source,
  };
}

function errorCode(error: unknown): string | undefined {
  return typeof error === "object" && error !== null && "code" in error
    ? String((error as { code?: unknown }).code)
    : undefined;
}

async function resolveAdjacentConfig(args: {
  workspaceRoot: string;
  specPath: string;
}): Promise<{ path: string; displayPath: string; source: Buffer } | undefined> {
  const candidate = join(dirname(args.specPath), "local-runner.json");
  let fileInfo: Awaited<ReturnType<typeof lstat>>;
  try {
    fileInfo = await lstat(candidate);
  } catch (error) {
    if (errorCode(error) === "ENOENT") return undefined;
    throw error;
  }
  if (fileInfo.isSymbolicLink() || !fileInfo.isFile()) {
    throw new Error("Adjacent local-runner.json must be a regular file, not a directory or symlink.");
  }
  if (fileInfo.size > MAX_CONFIG_BYTES) {
    throw new Error(`Adjacent local-runner.json exceeds ${MAX_CONFIG_BYTES} bytes.`);
  }
  const physicalPath = await realpath(candidate);
  if (!isWithin(args.workspaceRoot, physicalPath)) {
    throw new Error("Adjacent local-runner.json resolves outside the current workspace.");
  }
  const source = await readFile(physicalPath);
  if (source.byteLength > MAX_CONFIG_BYTES) {
    throw new Error(`Adjacent local-runner.json exceeds ${MAX_CONFIG_BYTES} bytes.`);
  }
  return {
    path: physicalPath,
    displayPath: `./${relative(args.workspaceRoot, physicalPath).split(sep).join("/")}`,
    source,
  };
}

export async function prepareLocalPipelineAction(args: {
  workspaceRoot: string;
  pipeline?: unknown;
  specPath?: string;
  dryRun?: boolean;
}): Promise<PreparedLocalPipelineAction> {
  const spec = await resolveWorkspaceSpec(
    args.workspaceRoot,
    args.specPath ?? DEFAULT_SPEC_PATH,
  );
  const input = parseLocalRunInput(
    JSON.parse(spec.source.toString("utf8")) as unknown,
    spec.path,
  );
  if (input.kind === "request") {
    throw new Error("Agent pipeline execution requires a tunedtensor.json spec, not a full run request.");
  }
  const config = await resolveAdjacentConfig({
    workspaceRoot: spec.workspaceRoot,
    specPath: spec.path,
  });
  if (config && (input.spec.runtime || input.spec.evaluation)) throw new Error("Conflicting configuration sources: migrate local-runner.json into tunedtensor.json.");
  const resolvedConfig = config
    ? parseLocalRunnerConfig(
        JSON.parse(config.source.toString("utf8")) as unknown,
        config.path,
      )
    : configForSpec(input.spec, spec.path);

  const pipeline = pipelineForRunInput(input, args.pipeline);
  const plan = createExecutionPlan(pipeline);
  const remote = plan.steps.find((step) => step.target !== "local");
  if (remote) {
    throw new Error(
      `Step "${remote.id}" targets cloud execution. The laptop-local agent can only approve local pipelines.`,
    );
  }
  if (canonicalJson(pipeline) !== canonicalJson(pipelineForRunInput(input))) {
    throw new Error("The saved spec drives execution. Update tunedtensor.json.pipeline and review the edit before previewing a different recipe.");
  }
  const dryRun = args.dryRun ?? true;
  if (!dryRun) {
    throw new Error(
      "Real pipeline execution requires the explicit direct tt pipeline run command; model-mediated approvals are dry-run only.",
    );
  }

  return {
    pipeline,
    plan,
    specPath: spec.displayPath,
    specSha256: createHash("sha256").update(spec.source).digest("hex"),
    dryRun,
    engine: input.kind === "foundation-spec" ? "foundation" : "adapter",
    resolvedSpec: input.spec,
    resolvedConfig,
    ...(config ? {
      configPath: config.displayPath,
      configSha256: createHash("sha256").update(config.source).digest("hex"),
    } : {}),
    workspaceRoot: spec.workspaceRoot,
    workspaceFingerprint: spec.workspaceFingerprint,
  };
}

export async function validatePreparedLocalPipelineAction(
  args: ValidatePreparedLocalPipelineActionArgs,
): Promise<PreparedLocalPipelineAction> {
  let prepared: PreparedLocalPipelineAction;
  try {
    prepared = await prepareLocalPipelineAction({
      workspaceRoot: args.workspaceRoot,
      pipeline: args.pipeline,
      specPath: args.specPath,
      dryRun: args.dryRun,
    });
  } catch (error) {
    throw new LocalPipelineActionError(
      error instanceof Error ? error.message : String(error),
      error,
    );
  }
  if (prepared.workspaceFingerprint !== args.expectedWorkspaceFingerprint) {
    throw new LocalPipelineWorkspaceMismatchError(
      "The local workspace changed after this pipeline action was prepared; return to the original workspace and prepare it again.",
    );
  }
  if (prepared.specSha256 !== args.expectedSpecSha256) {
    throw new LocalPipelineActionError(
      "The Tuned Tensor spec changed after this pipeline action was prepared; review and prepare it again.",
    );
  }
  if (
    prepared.configPath !== args.expectedConfigPath
    || prepared.configSha256 !== args.expectedConfigSha256
  ) {
    throw new LocalPipelineActionError(
      "The adjacent local-runner.json changed after this pipeline action was prepared; review and prepare it again.",
    );
  }
  return prepared;
}

export async function executeLocalPipelineAction(args: ValidatePreparedLocalPipelineActionArgs & {
  runCommand: LocalPipelineCommandRunner;
  signal?: AbortSignal;
}): Promise<{
  completed: true;
  command: string[];
  engine: "adapter" | "foundation";
  spec_path: string;
  config_path: string | null;
  dry_run: boolean;
}> {
  const prepared = await validatePreparedLocalPipelineAction(args);

  const temporaryDirectory = await mkdtemp(join(tmpdir(), "tt-agent-pipeline-"));
  const temporaryId = randomUUID();
  const specPath = join(temporaryDirectory, `${temporaryId}.spec.json`);
  const config = prepared.resolvedConfig!;
  const { evaluation, dryRun: _dryRun, ...runtime } = config;
  const sealedSpec = {
    ...prepared.resolvedSpec,
    pipeline: prepared.pipeline,
    ...(prepared.engine === "adapter" ? { runtime, evaluation } : {
      runtime: { ...prepared.resolvedSpec.runtime, ...(config.gpu ? { gpu: config.gpu } : {}) },
    }),
  };
  const command = ["pipeline", "run", "--spec", specPath, "--dry-run"];
  try {
    await writeFile(specPath, `${JSON.stringify(sealedSpec, null, 2)}\n`, { encoding: "utf8", mode: 0o600, flag: "wx" });
    const result = await args.runCommand(command, {
      cwd: prepared.workspaceRoot,
      signal: args.signal,
    });
    if (result.exitCode !== 0) {
      const suffix = result.signal ? ` after ${result.signal}` : "";
      throw new LocalPipelineActionError(
        `Approved pipeline dry-run exited with code ${result.exitCode ?? 1}${suffix}.`,
      );
    }
    return {
      completed: true,
      command: ["tt", "pipeline", "run", "--dry-run", "--spec", prepared.specPath],
      engine: prepared.engine,
      spec_path: prepared.specPath,
      config_path: prepared.configPath ?? null,
      dry_run: prepared.dryRun,
    };
  } catch (error) {
    if (isKnownLocalPipelineFailure(error)) throw error;
    throw new LocalPipelineActionError(
      error instanceof Error ? error.message : String(error),
      error,
    );
  } finally {
    await rm(temporaryDirectory, { recursive: true, force: true });
  }
}
