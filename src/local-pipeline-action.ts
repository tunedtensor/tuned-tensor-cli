import { createHash, randomUUID } from "node:crypto";
import { lstat, readFile, realpath, rm, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import {
  canonicalPipeline,
  createExecutionPlan,
  isFoundationPipeline,
  parsePipeline,
  pipelineFromFoundationHyperparameters,
  type ExecutionPlan,
  type Pipeline,
} from "./pipeline.js";
import {
  assertFoundationSpecReady,
  assertLocalRunInputReady,
  parseLocalRunInput,
  type LocalRunInput,
} from "./local-runtime/local-project.js";

const DEFAULT_SPEC_PATH = "tunedtensor.json";
const MAX_SPEC_BYTES = 2_000_000;

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
  specDirectory: string;
  workspaceRoot: string;
}

export interface LocalPipelineCommandResult {
  exitCode: number | null;
  signal?: NodeJS.Signals | null;
}

export type LocalPipelineCommandRunner = (
  args: string[],
  options: { cwd: string; signal?: AbortSignal },
) => Promise<LocalPipelineCommandResult>;

export class LocalPipelineActionError extends Error {
  constructor(message: string, cause?: unknown) {
    super(message, { cause });
    this.name = "LocalPipelineActionError";
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
): Promise<{ workspaceRoot: string; path: string; displayPath: string; source: Buffer }> {
  if (!requestedPath || isAbsolute(requestedPath)) {
    throw new Error("Pipeline spec_path must be a relative file inside the current workspace.");
  }
  const rootInfo = await lstat(workspaceRoot);
  if (rootInfo.isSymbolicLink() || !rootInfo.isDirectory()) {
    throw new Error("The local agent workspace must be a real directory, not a symlink.");
  }
  const canonicalRoot = await realpath(workspaceRoot);
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
    path: physicalPath,
    displayPath: `./${workspaceRelative}`,
    source,
  };
}

function pipelineForInput(
  input: Exclude<LocalRunInput, { kind: "request" }>,
  requested?: unknown,
): Pipeline {
  if (requested !== undefined) return requested as Pipeline;
  return input.kind === "foundation-spec"
    ? pipelineFromFoundationHyperparameters(input.spec.name, input.spec.foundation)
    : canonicalPipeline("local");
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
  if (input.kind === "foundation-spec") assertFoundationSpecReady(input.spec);
  else assertLocalRunInputReady(input.request);

  const pipeline = pipelineForInput(input, args.pipeline);
  const normalized = parsePipeline(pipeline);
  const foundationPipeline = isFoundationPipeline(normalized);
  if (foundationPipeline !== (input.kind === "foundation-spec")) {
    throw new Error(
      foundationPipeline
        ? "A foundation pipeline requires a foundation tunedtensor.json spec."
        : "A foundation tunedtensor.json spec requires a foundation pipeline.",
    );
  }
  const plan = createExecutionPlan(pipeline);
  const remote = plan.steps.find((step) => step.target !== "local");
  if (remote) {
    throw new Error(
      `Step "${remote.id}" targets cloud execution. The laptop-local agent can only approve local pipelines.`,
    );
  }
  return {
    pipeline,
    plan,
    specPath: spec.displayPath,
    specSha256: createHash("sha256").update(spec.source).digest("hex"),
    dryRun: args.dryRun ?? true,
    engine: input.kind === "foundation-spec" ? "foundation" : "adapter",
    resolvedSpec: input.spec,
    specDirectory: dirname(spec.path),
    workspaceRoot: spec.workspaceRoot,
  };
}

export async function executeLocalPipelineAction(args: {
  workspaceRoot: string;
  pipeline: unknown;
  specPath: string;
  expectedSpecSha256: string;
  dryRun: boolean;
  runCommand: LocalPipelineCommandRunner;
  signal?: AbortSignal;
}): Promise<{
  completed: true;
  command: string[];
  engine: "adapter" | "foundation";
  spec_path: string;
  dry_run: boolean;
}> {
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
  if (prepared.specSha256 !== args.expectedSpecSha256) {
    throw new LocalPipelineActionError(
      "The Tuned Tensor spec changed after this pipeline action was prepared; review and prepare it again.",
    );
  }

  const temporaryId = randomUUID();
  const pipelinePath = join(
    prepared.specDirectory,
    `.tt-agent-${temporaryId}.pipeline.json`,
  );
  const specPath = join(
    prepared.specDirectory,
    `.tt-agent-${temporaryId}.spec.json`,
  );
  const command = [
    "pipeline",
    "run",
    "--file",
    pipelinePath,
    "--spec",
    specPath,
    ...(prepared.dryRun ? ["--dry-run"] : []),
  ];
  try {
    await Promise.all([
      writeFile(pipelinePath, `${JSON.stringify(prepared.pipeline, null, 2)}\n`, {
        encoding: "utf8",
        mode: 0o600,
        flag: "wx",
      }),
      writeFile(specPath, `${JSON.stringify(prepared.resolvedSpec, null, 2)}\n`, {
        encoding: "utf8",
        mode: 0o600,
        flag: "wx",
      }),
    ]);
    const result = await args.runCommand(command, {
      cwd: prepared.workspaceRoot,
      signal: args.signal,
    });
    if (result.exitCode !== 0) {
      const suffix = result.signal ? ` after ${result.signal}` : "";
      throw new LocalPipelineActionError(
        `Approved pipeline exited with code ${result.exitCode ?? 1}${suffix}. Inspect the local run events before preparing another action.`,
      );
    }
    return {
      completed: true,
      command: ["tt", "pipeline", "run", "--spec", prepared.specPath],
      engine: prepared.engine,
      spec_path: prepared.specPath,
      dry_run: prepared.dryRun,
    };
  } catch (error) {
    if (isKnownLocalPipelineFailure(error)) throw error;
    throw new LocalPipelineActionError(
      error instanceof Error ? error.message : String(error),
      error,
    );
  } finally {
    await Promise.all([
      rm(pipelinePath, { force: true }),
      rm(specPath, { force: true }),
    ]);
  }
}
