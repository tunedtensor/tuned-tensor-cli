import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { lstat, mkdir, open, realpath, rmdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { Type, type Static } from "typebox";
import { Value } from "typebox/value";
import { TRAINING_MODELS } from "./local-runtime/model-registry.js";
import { validateSpec } from "./eval/rules.js";
import type { LocalSpec } from "./eval/types.js";

const FolderName = Type.String({
  minLength: 1,
  maxLength: 120,
  pattern: "^[A-Za-z0-9][A-Za-z0-9._-]*$",
  description: "One new folder name directly beneath the current workspace",
});
const LocalBaseModel = Type.String({
  type: "string",
  enum: TRAINING_MODELS.map((model) => model.id),
});
const NonEmptyText = Type.String({ minLength: 1, maxLength: 1_000 });

export const LocalProjectSpecSchema = Type.Object({
  name: Type.String({ minLength: 1, maxLength: 255 }),
  description: Type.Optional(Type.String({ maxLength: 5_000 })),
  base_model: LocalBaseModel,
  system_prompt: Type.String({ minLength: 1, maxLength: 10_000 }),
  guidelines: Type.Array(NonEmptyText, { minItems: 1, maxItems: 50 }),
  constraints: Type.Optional(Type.Array(NonEmptyText, { maxItems: 50 })),
  examples: Type.Array(Type.Object({
    input: Type.String({ minLength: 1, maxLength: 100_000 }),
    output: Type.String({ minLength: 1, maxLength: 100_000 }),
  }, { additionalProperties: false }), { minItems: 2, maxItems: 500 }),
}, { additionalProperties: false });

export type LocalProjectSpec = Static<typeof LocalProjectSpecSchema>;
export { FolderName as LocalProjectFolderSchema };

export interface PreparedLocalSpecProject {
  directory: string;
  specPath: string;
  workspaceFingerprint: string;
}

export interface CreatedLocalSpecProject {
  created: true;
  directory: string;
  path: string;
}

export interface LocalSpecFileOperations {
  mkdir(path: string, options: { mode: number }): Promise<unknown>;
  writeFile(
    path: string,
    data: string,
    options: { encoding: "utf8"; flag: "wx"; mode: number },
  ): Promise<unknown>;
  rmdir(path: string): Promise<unknown>;
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

const NODE_FILE_OPERATIONS: LocalSpecFileOperations = {
  mkdir,
  writeFile,
  rmdir,
};

function errorCode(error: unknown): string | undefined {
  return typeof error === "object" && error !== null && "code" in error
    ? String((error as { code?: unknown }).code)
    : undefined;
}

function validateInputs(directory: string, spec: unknown): asserts spec is LocalProjectSpec {
  if (!Value.Check(FolderName, directory)) {
    throw new Error(
      "Local spec directory must be one portable folder name using only letters, numbers, dots, underscores, or hyphens.",
    );
  }
  if (!Value.Check(LocalProjectSpecSchema, spec)) {
    throw new Error("Local spec content does not match the canonical tunedtensor.json schema.");
  }
  const validation = validateSpec({
    ...spec,
    description: spec.description ?? "",
    constraints: spec.constraints ?? [],
  } as LocalSpec);
  if (!validation.valid) {
    const messages = validation.checks
      .filter((check) => !check.passed)
      .map((check) => check.message ?? check.name);
    throw new Error(`Local spec validation failed:\n- ${messages.join("\n- ")}`);
  }
}

async function canonicalWorkspace(workspaceRoot: string): Promise<string> {
  const rootInfo = await lstat(workspaceRoot);
  if (rootInfo.isSymbolicLink() || !rootInfo.isDirectory()) {
    throw new Error("The local agent workspace must be a real directory, not a symlink.");
  }
  return await realpath(workspaceRoot);
}

async function fingerprintWorkspace(canonicalRoot: string): Promise<string> {
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
  directory: string,
  spec: unknown,
): Promise<PreparedLocalSpecProject> {
  validateInputs(directory, spec);
  const canonicalRoot = await canonicalWorkspace(workspaceRoot);
  const targetDirectory = join(canonicalRoot, directory);
  try {
    await lstat(targetDirectory);
  } catch (error) {
    if (errorCode(error) === "ENOENT") {
      return {
        directory: `./${directory}`,
        specPath: `./${directory}/tunedtensor.json`,
        workspaceFingerprint: await fingerprintWorkspace(canonicalRoot),
      };
    }
    throw error;
  }
  throw new Error(`Refusing to overwrite existing path ./${directory}.`);
}

export async function createLocalSpecProject(
  workspaceRoot: string,
  directory: string,
  spec: unknown,
  expectedWorkspaceFingerprint: string,
  operationOverrides: Partial<LocalSpecFileOperations> = {},
): Promise<CreatedLocalSpecProject> {
  let prepared: PreparedLocalSpecProject;
  try {
    prepared = await prepareLocalSpecProject(workspaceRoot, directory, spec);
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new LocalSpecMutationError(detail, "not_applied", error);
  }
  if (prepared.workspaceFingerprint !== expectedWorkspaceFingerprint) {
    throw new LocalSpecMutationError(
      "The local workspace changed after this action was prepared; return to the original workspace and prepare it again.",
      "not_applied",
    );
  }

  let canonicalRoot: string;
  try {
    canonicalRoot = await canonicalWorkspace(workspaceRoot);
    if (await fingerprintWorkspace(canonicalRoot) !== expectedWorkspaceFingerprint) {
      throw new LocalSpecMutationError(
        "The local workspace changed after this action was prepared; return to the original workspace and prepare it again.",
        "not_applied",
      );
    }
  } catch (error) {
    if (error instanceof LocalSpecMutationError) throw error;
    const detail = error instanceof Error ? error.message : String(error);
    throw new LocalSpecMutationError(detail, "not_applied", error);
  }
  if (process.platform !== "linux") {
    throw new LocalSpecMutationError(
      "Secure local spec creation currently requires Linux filesystem handle support.",
      "not_applied",
    );
  }

  let workspaceHandle: Awaited<ReturnType<typeof open>> | undefined;
  try {
    workspaceHandle = await open(
      canonicalRoot,
      constants.O_RDONLY | constants.O_DIRECTORY | constants.O_NOFOLLOW,
    );
    const openedWorkspaceInfo = await workspaceHandle.stat();
    if (fingerprintWorkspaceInfo(openedWorkspaceInfo) !== expectedWorkspaceFingerprint) {
      throw new LocalSpecMutationError(
        "The local workspace changed after this action was prepared; return to the original workspace and prepare it again.",
        "not_applied",
      );
    }
  } catch (error) {
    if (workspaceHandle) {
      try { await workspaceHandle.close(); } catch { /* No mutation has started. */ }
    }
    if (error instanceof LocalSpecMutationError) throw error;
    const detail = error instanceof Error ? error.message : String(error);
    throw new LocalSpecMutationError(detail, "not_applied", error);
  }
  if (!workspaceHandle) {
    throw new LocalSpecMutationError("The local workspace could not be opened safely.", "not_applied");
  }

  const stableWorkspaceRoot = `/proc/self/fd/${workspaceHandle.fd}`;
  const targetDirectory = join(stableWorkspaceRoot, directory);
  const operations = { ...NODE_FILE_OPERATIONS, ...operationOverrides };
  let createdDirectory = false;
  let writeStarted = false;
  let directoryHandle: Awaited<ReturnType<typeof open>> | undefined;
  try {
    await operations.mkdir(targetDirectory, { mode: 0o700 });
    createdDirectory = true;
    const targetInfo = await lstat(targetDirectory);
    if (targetInfo.isSymbolicLink() || !targetInfo.isDirectory()) {
      throw new Error(`Local spec destination ./${directory} changed before the spec could be written.`);
    }
    directoryHandle = await open(
      targetDirectory,
      constants.O_RDONLY | constants.O_DIRECTORY | constants.O_NOFOLLOW,
    );
    const openedInfo = await directoryHandle.stat();
    if (openedInfo.dev !== targetInfo.dev || openedInfo.ino !== targetInfo.ino) {
      throw new Error(`Local spec destination ./${directory} changed before the spec could be written.`);
    }
    const stableSpecPath = join(`/proc/self/fd/${directoryHandle.fd}`, "tunedtensor.json");
    writeStarted = true;
    await operations.writeFile(stableSpecPath, `${JSON.stringify(spec, null, 2)}\n`, {
      encoding: "utf8",
      flag: "wx",
      mode: 0o600,
    });
    const finalInfo = await lstat(targetDirectory);
    if (
      finalInfo.isSymbolicLink()
      || !finalInfo.isDirectory()
      || finalInfo.dev !== openedInfo.dev
      || finalInfo.ino !== openedInfo.ino
    ) {
      throw new Error(`Local spec destination ./${directory} changed while the spec was being written.`);
    }
    const finalWorkspaceInfo = await lstat(canonicalRoot);
    if (
      finalWorkspaceInfo.isSymbolicLink()
      || !finalWorkspaceInfo.isDirectory()
      || fingerprintWorkspaceInfo(finalWorkspaceInfo) !== expectedWorkspaceFingerprint
    ) {
      throw new Error("The local workspace changed while the spec was being written.");
    }
    await directoryHandle.close();
    directoryHandle = undefined;
    await workspaceHandle.close();
  } catch (error) {
    const rollbackErrors: unknown[] = [];
    if (directoryHandle) {
      try {
        await directoryHandle.close();
      } catch (rollbackError) {
        rollbackErrors.push(rollbackError);
      }
    }
    if (createdDirectory) {
      // Never unlink the spec path after an ambiguous write: a racing writer
      // may own it. Only remove the directory when it is still empty.
      try {
        await operations.rmdir(targetDirectory);
      } catch (rollbackError) {
        if (errorCode(rollbackError) !== "ENOENT") rollbackErrors.push(rollbackError);
      }
    }
    try {
      await workspaceHandle.close();
    } catch (rollbackError) {
      rollbackErrors.push(rollbackError);
    }
    if (writeStarted) {
      throw new LocalSpecMutationError(
        `Local spec writing did not complete conclusively for ./${directory}; inspect the workspace before retrying.`,
        "unknown",
        error,
      );
    }
    if (rollbackErrors.length > 0) {
      throw new LocalSpecMutationError(
        `Local spec creation failed and rollback could not be completed for ./${directory}; inspect the workspace before retrying.`,
        "unknown",
        error,
      );
    }
    if (createdDirectory || errorCode(error) === "EEXIST") {
      const detail = error instanceof Error ? error.message : String(error);
      throw new LocalSpecMutationError(detail, "not_applied", error);
    }
    throw new LocalSpecMutationError(
      `Local spec directory creation failed for ./${directory}; inspect the workspace before retrying.`,
      "unknown",
      error,
    );
  }

  return {
    created: true,
    directory: prepared.directory,
    path: prepared.specPath,
  };
}
