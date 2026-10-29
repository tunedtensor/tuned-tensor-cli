import { createHash, randomUUID } from "node:crypto";
import { constants } from "node:fs";
import { lstat, open, realpath, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { Type, type Static } from "typebox";
import { Value } from "typebox/value";
import { TRAINING_MODELS } from "./local-runtime/model-registry.js";
import { DECISION_MODELS } from "./local-runtime/decision-models.js";

import { parseLocalRunInput } from "./local-runtime/local-project.js";
import { pipelineForRunInput } from "./pipeline.js";
import { WindowsWorkspace } from "./windows-workspace.js";

const LocalBaseModel = Type.String({
  minLength: 3,
  maxLength: 193,
  pattern: "^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$",
  description: `Hugging Face repo id. Certified adapter models: ${TRAINING_MODELS.map((model) => model.id).join(", ")}. `
    + `Decision models: ${DECISION_MODELS.map((model) => model.id).join(", ")}. `
    + "Other adapter repos require hyperparameters.base_model_revision (40-character commit SHA).",
});
const NonEmptyText = Type.String({ minLength: 1, maxLength: 1_000 });

export const LocalProjectSpecSchema = Type.Object({
  name: Type.String({ minLength: 1, maxLength: 255 }),
  description: Type.Optional(Type.String({ maxLength: 5_000 })),
  base_model: Type.Optional(LocalBaseModel),
  engine: Type.Optional(Type.Union([Type.Literal("adapter"), Type.Literal("foundation"), Type.Literal("decision")])),
  decision: Type.Optional(Type.Record(Type.String(), Type.Unknown())),
  hyperparameters: Type.Optional(Type.Record(Type.String(), Type.Unknown())),
  foundation: Type.Optional(Type.Record(Type.String(), Type.Unknown())),
  dataset_prebuilt: Type.Optional(Type.Record(Type.String(), Type.Unknown())),
  evaluation: Type.Optional(Type.Record(Type.String(), Type.Unknown())),
  runtime: Type.Optional(Type.Record(Type.String(), Type.Unknown())),
  pipeline: Type.Optional(Type.Record(Type.String(), Type.Unknown())),
  system_prompt: Type.String({ minLength: 1, maxLength: 10_000 }),
  guidelines: Type.Array(NonEmptyText, { minItems: 1, maxItems: 50 }),
  constraints: Type.Optional(Type.Array(NonEmptyText, { maxItems: 50 })),
  examples: Type.Array(Type.Object({
    input: Type.String({ minLength: 1, maxLength: 100_000 }),
    output: Type.String({ minLength: 1, maxLength: 100_000 }),
  }, { additionalProperties: false }), { minItems: 2, maxItems: 500 }),
}, { additionalProperties: false });

export type LocalProjectSpec = Static<typeof LocalProjectSpecSchema>;

// New specs live where tt reads them by default: the workspace root.
const SPEC_FILE = "tunedtensor.json";
const SPEC_DISPLAY_PATH = `./${SPEC_FILE}`;

export interface PreparedLocalSpecProject {
  specPath: string;
  workspaceFingerprint: string;
}

export interface CreatedLocalSpecProject {
  created: true;
  path: string;
}

export interface LocalSpecFileOperations {
  writeFile(
    path: string,
    data: string,
    options: { encoding: "utf8"; flag: "wx"; mode: number },
  ): Promise<unknown>;
}

export class LocalSpecMutationError extends Error {
  constructor(
    message: string,
    readonly outcome: "not_applied" | "unknown",
    cause?: unknown,
  ) {
    super(message, { cause });
    this.name = "LocalSpecMutationError";
  }
}

export function isKnownLocalSpecFailure(error: unknown): boolean {
  return error instanceof LocalSpecMutationError && error.outcome === "not_applied";
}

const NODE_FILE_OPERATIONS: LocalSpecFileOperations = { writeFile };

function errorCode(error: unknown): string | undefined {
  return typeof error === "object" && error !== null && "code" in error
    ? String((error as { code?: unknown }).code)
    : undefined;
}

function validateInputs(spec: unknown): asserts spec is LocalProjectSpec {
  if (!Value.Check(LocalProjectSpecSchema, spec)) {
    throw new Error("Local spec content does not match the canonical tunedtensor.json schema.");
  }
  // Same behavior-spec rules as review, edits and execution.
  pipelineForRunInput(parseLocalRunInput(spec, SPEC_FILE));
}

export async function canonicalWorkspace(workspaceRoot: string): Promise<string> {
  const rootInfo = await lstat(workspaceRoot);
  if (rootInfo.isSymbolicLink() || !rootInfo.isDirectory()) {
    throw new Error("The local agent workspace must be a real directory, not a symlink.");
  }
  return await realpath(workspaceRoot);
}

export async function fingerprintWorkspace(canonicalRoot: string): Promise<string> {
  return fingerprintWorkspaceInfo(await lstat(canonicalRoot));
}

function fingerprintWorkspaceInfo(
  info: { dev: number | bigint; ino: number | bigint; birthtimeMs: number },
): string {
  const identity = `${info.dev}:${info.ino}:${info.birthtimeMs}`;
  return createHash("sha256").update(identity).digest("hex");
}

export async function prepareLocalSpecProject(
  workspaceRoot: string,
  spec: unknown,
  expectedWorkspaceFingerprint?: string,
): Promise<PreparedLocalSpecProject> {
  const canonicalRoot = await canonicalWorkspace(workspaceRoot);
  const workspaceFingerprint = await fingerprintWorkspace(canonicalRoot);
  if (
    expectedWorkspaceFingerprint !== undefined
    && workspaceFingerprint !== expectedWorkspaceFingerprint
  ) {
    throw new Error(
      "The local workspace changed after this action was prepared; return to the original workspace and prepare it again.",
    );
  }
  validateInputs(spec);
  try {
    await lstat(join(canonicalRoot, SPEC_FILE));
  } catch (error) {
    if (errorCode(error) === "ENOENT") {
      return { specPath: SPEC_DISPLAY_PATH, workspaceFingerprint };
    }
    throw error;
  }
  throw new Error(
    `Refusing to overwrite existing ${SPEC_DISPLAY_PATH}; edit it with prepare_update_local_spec instead.`,
  );
}

export async function createLocalSpecProject(
  workspaceRoot: string,
  spec: unknown,
  expectedWorkspaceFingerprint: string,
  operationOverrides: Partial<LocalSpecFileOperations> = {},
): Promise<CreatedLocalSpecProject> {
  let prepared: PreparedLocalSpecProject;
  let canonicalRoot: string;
  try {
    prepared = await prepareLocalSpecProject(workspaceRoot, spec, expectedWorkspaceFingerprint);
    canonicalRoot = await canonicalWorkspace(workspaceRoot);
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new LocalSpecMutationError(detail, "not_applied", error);
  }
  if (process.platform !== "linux" && process.platform !== "win32") {
    throw new LocalSpecMutationError(
      "Secure local spec creation currently requires Linux filesystem handle support.",
      "not_applied",
    );
  }

  let workspaceHandle: Awaited<ReturnType<typeof open>> | undefined;
  let windows: WindowsWorkspace | undefined;
  try {
    if (process.platform === "win32") {
      windows = await WindowsWorkspace.open(canonicalRoot);
    } else workspaceHandle = await open(
      canonicalRoot,
      constants.O_RDONLY | constants.O_DIRECTORY | constants.O_NOFOLLOW,
    );
    if ((windows ? await fingerprintWorkspace(canonicalRoot) : fingerprintWorkspaceInfo(await workspaceHandle!.stat())) !== expectedWorkspaceFingerprint) {
      throw new LocalSpecMutationError(
        "The local workspace changed after this action was prepared; return to the original workspace and prepare it again.",
        "not_applied",
      );
    }
  } catch (error) {
    if (workspaceHandle) {
      try { await workspaceHandle.close(); } catch { /* No mutation has started. */ }
    }
    await windows?.close();
    if (error instanceof LocalSpecMutationError) throw error;
    const detail = error instanceof Error ? error.message : String(error);
    throw new LocalSpecMutationError(detail, "not_applied", error);
  }

  const operations = { ...NODE_FILE_OPERATIONS, ...operationOverrides };
  // Write through the opened workspace handle so a swapped root is never followed.
  const stableSpecPath = windows ? join(canonicalRoot, SPEC_FILE) : join(`/proc/self/fd/${workspaceHandle!.fd}`, SPEC_FILE);
  try {
    // A stable ID keeps run identity and the eval split fixed across later edits.
    const source = `${JSON.stringify({ id: randomUUID(), ...spec as LocalProjectSpec }, null, 2)}\n`;
    if (windows && !operationOverrides.writeFile) await windows.create(canonicalRoot, SPEC_FILE, source);
    else await operations.writeFile(stableSpecPath, source, {
      encoding: "utf8",
      flag: "wx",
      mode: 0o600,
    });
    const finalWorkspaceInfo = await lstat(canonicalRoot);
    if (
      finalWorkspaceInfo.isSymbolicLink()
      || !finalWorkspaceInfo.isDirectory()
      || fingerprintWorkspaceInfo(finalWorkspaceInfo) !== expectedWorkspaceFingerprint
    ) {
      throw new Error("The local workspace changed while the spec was being written.");
    }
  } catch (error) {
    try { await workspaceHandle?.close(); } catch { /* The outcome is already decided below. */ }
    await windows?.close();
    const detail = error instanceof Error ? error.message : String(error);
    if (errorCode(error) === "EEXIST") {
      throw new LocalSpecMutationError(`Refusing to overwrite existing ${SPEC_DISPLAY_PATH}.`, "not_applied", error);
    }
    // Never unlink the spec after an ambiguous write: a racing writer may own it.
    throw new LocalSpecMutationError(
      `Local spec writing did not complete conclusively for ${SPEC_DISPLAY_PATH}; inspect the workspace before retrying. ${detail}`,
      "unknown",
      error,
    );
  }
  try { await workspaceHandle?.close(); } catch { /* The spec is written. */ }
  await windows?.close();

  return { created: true, path: prepared.specPath };
}
