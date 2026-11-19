import { z } from "zod";
import type { AgentLocalServer } from "./agent-local-server.js";
import { validateProjectModelAction, executeProjectModelAction, buildProjectModelLaunch } from "./agent-local-models.js";
import { validateDatasetSplit, executeDatasetSplit } from "./local-runtime/dataset-split.js";
import { applySpecUpdate, validateSpecUpdate, SpecWorkspaceMismatchError, type SpecUpdate } from "./spec-workspace.js";
import type { AgentAction } from "./agent-client.js";
import {
  canonicalWorkspace,
  createLocalSpecProject,
  fingerprintWorkspace,
  isKnownLocalSpecFailure,
  prepareLocalSpecProject,
  type LocalSpecFileOperations,
} from "./local-spec-workspace.js";
import {
  executeLocalPipelineAction,
  isKnownLocalPipelineFailure,
  LocalPipelineWorkspaceMismatchError,
  type LocalPipelineCommandRunner,
  validatePreparedLocalPipelineAction,
} from "./local-pipeline-action.js";
import { executeLocalPrefetch, LocalPrefetchError, revalidateLocalPrefetch } from "./local-prefetch-action.js";
import {
  executeBackgroundRunCancel,
  executeBackgroundRunStart,
  isBackgroundRunFailure,
  revalidateBackgroundRunStart,
  type BackgroundRunExecutor,
} from "./local-background-run-action.js";
import type { BackgroundRunRequester } from "./local-runtime/background-runs.js";

export type PersistAction = (action: AgentAction) => Promise<void>;

export interface AgentApprovalOptions {
  localServer?: AgentLocalServer;
  workspaceRoot?: string;
  localFileOperations?: Partial<LocalSpecFileOperations>;
  runPipelineCommand?: LocalPipelineCommandRunner;
  signal?: AbortSignal;
  durableClaimed?: boolean;
  backgroundRuns?: BackgroundRunExecutor;
  /** Recorded as the approver of background runs and cancellations. */
  requestedBy?: BackgroundRunRequester;
}

export class PreparedActionWorkspaceMismatchError extends Error {
  constructor(message: string, cause?: unknown) {
    super(message, { cause });
    this.name = "PreparedActionWorkspaceMismatchError";
  }
}

function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

function args(action: AgentAction): Record<string, unknown> {
  const value = record(action.arguments);
  if (!value) throw new Error("Prepared action arguments are invalid.");
  return value;
}

function requiredString(value: unknown, label: string): string {
  if (typeof value !== "string" || !value) throw new Error(`Prepared action ${label} is invalid.`);
  return value;
}

function requiredBoolean(value: unknown, label: string): boolean {
  if (typeof value !== "boolean") throw new Error(`Prepared action ${label} is invalid.`);
  return value;
}

function optionalString(value: unknown, label: string): string | undefined {
  if (value === null || value === undefined) return undefined;
  return requiredString(value, label);
}

async function fail(
  action: AgentAction,
  persist: PersistAction,
  error: unknown,
): Promise<never> {
  action.status = "failed";
  try {
    await persist(action);
  } catch {
    // Preserve the original safety failure.
  }
  throw error;
}

async function prepareLocalPipelineApproval(
  action: AgentAction,
  options: AgentApprovalOptions,
) {
  const input = args(action);
  const workspaceRoot = options.workspaceRoot;
  if (!workspaceRoot) {
    throw new Error("A local workspace is required to run a local pipeline.");
  }
  if (!options.runPipelineCommand) {
    throw new Error("The local pipeline command runner is unavailable.");
  }
  const pipeline = record(input.pipeline);
  if (!pipeline) throw new Error("Prepared pipeline content is invalid.");
  const specPath = requiredString(input.spec_path, "spec path");
  const specSha256 = requiredString(input.spec_sha256, "spec fingerprint");
  const workspaceFingerprint = requiredString(
    input.workspace_fingerprint,
    "workspace fingerprint",
  );
  const configPath = optionalString(input.config_path, "config path");
  const configSha256 = optionalString(input.config_sha256, "config fingerprint");
  if (Boolean(configPath) !== Boolean(configSha256)) {
    throw new Error("Prepared action config path and fingerprint must be provided together.");
  }
  const dryRun = requiredBoolean(input.dry_run, "dry-run flag");
  if (!dryRun) {
    throw new Error(
      "Real pipeline execution requires the explicit direct tt pipeline run command; model-mediated approvals are dry-run only.",
    );
  }
  const validation = {
    workspaceRoot,
    pipeline,
    specPath,
    expectedSpecSha256: specSha256,
    expectedWorkspaceFingerprint: workspaceFingerprint,
    expectedConfigPath: configPath,
    expectedConfigSha256: configSha256,
    dryRun,
  };
  await validatePreparedLocalPipelineAction(validation);
  return {
    validation,
    runCommand: options.runPipelineCommand,
    signal: options.signal,
  };
}

const serveArguments = z.object({ selection: z.unknown(), port: z.number().int().min(1024).max(65535) }).strict();
const stopArguments = z.object({ server_id: z.string().uuid() }).strict();

export async function preflightPreparedAction(
  action: AgentAction,
  options: AgentApprovalOptions = {},
): Promise<void> {
  if (action.status !== "proposed") {
    throw new Error(`Action ${action.id} is not proposed and cannot be approved.`);
  }
  if (action.operation === "serve_local_model") {
    if (!options.workspaceRoot || !options.localServer) throw new Error("Local serving is unavailable.");
    const input = serveArguments.parse(args(action));
    await buildProjectModelLaunch(options.workspaceRoot, input.selection, input.port);
  } else if (action.operation === "stop_local_service") {
    if (!options.localServer || (await options.localServer.status()).server_id !== stopArguments.parse(args(action)).server_id) throw new Error("Server selection changed.");
  } else if (action.operation === "activate_local_model" || action.operation === "rollback_local_model") {
    if (!options.workspaceRoot) throw new Error("Model selection needs a local workspace.");
    const input = args(action);
    if (input.operation !== (action.operation === "activate_local_model" ? "activate" : "rollback")) throw new Error("Model action operation mismatch.");
    await validateProjectModelAction(options.workspaceRoot, input);
  } else if (action.operation === "update_local_spec") {
    if (!options.workspaceRoot) throw new Error("A local workspace is required to edit a spec.");
    try { await validateSpecUpdate(options.workspaceRoot, args(action) as unknown as SpecUpdate); }
    catch (error) {
      if (error instanceof SpecWorkspaceMismatchError) throw new PreparedActionWorkspaceMismatchError(error.message, error);
      throw error;
    }
  } else if (action.operation === "create_local_spec") {
    const input = args(action);
    if (input.directory !== undefined) {
      throw new Error("This proposal targets a subfolder from an older tt version; prepare the spec again.");
    }
    const workspaceRoot = options.workspaceRoot;
    if (!workspaceRoot) {
      throw new Error("A local workspace is required to create a local spec.");
    }
    const expectedWorkspaceFingerprint = requiredString(
      input.workspace_fingerprint,
      "workspace fingerprint",
    );
    const canonicalRoot = await canonicalWorkspace(workspaceRoot);
    const actualWorkspaceFingerprint = await fingerprintWorkspace(canonicalRoot);
    if (actualWorkspaceFingerprint !== expectedWorkspaceFingerprint) {
      throw new PreparedActionWorkspaceMismatchError(
        "The local workspace changed after this action was prepared; return to the original workspace and prepare it again.",
      );
    }
    await prepareLocalSpecProject(workspaceRoot, input.spec, expectedWorkspaceFingerprint);
  } else if (action.operation === "split_local_dataset") {
    if (!options.workspaceRoot) throw new Error("A local workspace is required to split data.");
    try { await validateDatasetSplit(options.workspaceRoot, args(action), options.signal); }
    catch (error) {
      if (error instanceof SpecWorkspaceMismatchError) throw new PreparedActionWorkspaceMismatchError(error.message, error);
      throw error;
    }
  } else if (action.operation === "prefetch_local") {
    if (!options.workspaceRoot) throw new Error("A local workspace is required to download.");
    try { await revalidateLocalPrefetch(options.workspaceRoot, args(action)); }
    catch (error) {
      if (error instanceof SpecWorkspaceMismatchError) throw new PreparedActionWorkspaceMismatchError(error.message, error);
      throw error;
    }
  } else if (action.operation === "start_local_pipeline") {
    if (!options.workspaceRoot || !options.backgroundRuns) throw new Error("Background runs need a local workspace and the tt command runner.");
    try {
      await revalidateBackgroundRunStart(options.workspaceRoot, args(action));
    } catch (error) {
      if (error instanceof LocalPipelineWorkspaceMismatchError) throw new PreparedActionWorkspaceMismatchError(error.message, error);
      throw error;
    }
  } else if (action.operation === "run_local_pipeline") {
    try {
      await prepareLocalPipelineApproval(action, options);
    } catch (error) {
      if (error instanceof LocalPipelineWorkspaceMismatchError) {
        throw new PreparedActionWorkspaceMismatchError(error.message, error);
      }
      throw error;
    }
  }
}

export async function approvePreparedAction(
  action: AgentAction,
  persist: PersistAction,
  options: AgentApprovalOptions = {},
): Promise<unknown> {
  if (action.status !== "proposed") {
    throw new Error(`Action ${action.id} is not proposed and cannot be approved.`);
  }
  let execute: (() => Promise<unknown>) | undefined;
  try {
    const input = args(action);

    switch (action.operation) {
      case "serve_local_model": {
        const { workspaceRoot, localServer } = options;
        if (!workspaceRoot || !localServer) throw new Error("Local serving is unavailable.");
        const serving = serveArguments.parse(input);
        await buildProjectModelLaunch(workspaceRoot, serving.selection, serving.port);
        execute = async () => {
          const current = await buildProjectModelLaunch(workspaceRoot, serving.selection, serving.port);
          return localServer.start(current.launch, current.logPath);
        };
        break;
      }
      case "stop_local_service": {
        if (!options.localServer) throw new Error("Local serving is unavailable.");
        const serverId = stopArguments.parse(input).server_id;
        execute = () => options.localServer!.stop(serverId);
        break;
      }
      case "activate_local_model":
      case "rollback_local_model": {
        if (!options.workspaceRoot) throw new Error("Model selection needs a local workspace.");
        if (input.operation !== (action.operation === "activate_local_model" ? "activate" : "rollback")) throw new Error("Model action operation mismatch.");
        await validateProjectModelAction(options.workspaceRoot, input);
        execute = () => executeProjectModelAction(options.workspaceRoot!, input);
        break;
      }
      case "update_local_spec": {
        if (!options.workspaceRoot) throw new Error("A local workspace is required to edit a spec.");
        const update = input as unknown as SpecUpdate;
        await validateSpecUpdate(options.workspaceRoot, update);
        execute = () => applySpecUpdate(options.workspaceRoot!, update);
        break;
      }
      case "create_local_spec": {
        const workspaceRoot = options.workspaceRoot;
        if (!workspaceRoot) {
          throw new Error("A local workspace is required to create a local spec.");
        }
        if (input.directory !== undefined) {
          throw new Error("This proposal targets a subfolder from an older tt version; prepare the spec again.");
        }
        const workspaceFingerprint = requiredString(
          input.workspace_fingerprint,
          "workspace fingerprint",
        );
        execute = async () => await createLocalSpecProject(
          workspaceRoot,
          input.spec,
          workspaceFingerprint,
          options.localFileOperations,
        );
        break;
      }
      case "run_local_pipeline": {
        const prepared = await prepareLocalPipelineApproval(action, options);
        execute = async () => await executeLocalPipelineAction({
          ...prepared.validation,
          runCommand: prepared.runCommand,
          signal: prepared.signal,
        });
        break;
      }
      case "start_local_pipeline": {
        const { workspaceRoot, backgroundRuns, requestedBy } = options;
        if (!workspaceRoot || !backgroundRuns || !requestedBy) throw new Error("Background runs need a local workspace and the tt command runner.");
        await revalidateBackgroundRunStart(workspaceRoot, input);
        execute = async () => await executeBackgroundRunStart({ workspaceRoot, input, executor: backgroundRuns, requestedBy });
        break;
      }
      case "cancel_local_pipeline": {
        const { backgroundRuns, requestedBy } = options;
        if (!backgroundRuns || !requestedBy) throw new Error("Cancelling a background run needs the local run store.");
        execute = async () => await executeBackgroundRunCancel({ input, store: backgroundRuns.store, requestedBy });
        break;
      }
      case "split_local_dataset": {
        if (!options.workspaceRoot) throw new Error("A local workspace is required to split data.");
        await validateDatasetSplit(options.workspaceRoot, input, options.signal);
        execute = () => executeDatasetSplit(options.workspaceRoot!, input, options.signal);
        break;
      }
      case "prefetch_local": {
        const runCommand = options.runPipelineCommand;
        if (!options.workspaceRoot || !runCommand) throw new Error("Downloads need a local workspace and the tt command runner.");
        const prepared = await revalidateLocalPrefetch(options.workspaceRoot, input);
        execute = async () => await executeLocalPrefetch(prepared, runCommand, options.signal);
        break;
      }
      default:
        throw new Error(`Unsupported prepared operation: ${String(action.operation)}.`);
    }

  } catch (error) {
    if (
      (error instanceof LocalPipelineWorkspaceMismatchError || error instanceof SpecWorkspaceMismatchError)
      && options.durableClaimed !== true
    ) {
      throw error;
    }
    return await fail(action, persist, error);
  }

  // Persisting "executing" is the durable one-way gate. After dispatch, an
  // error may mean the mutation succeeded but its response or completion
  // record was lost, so it must never be represented as safely failed.
  action.status = "executing";
  try {
    await persist(action);
  } catch (error) {
    return await fail(action, persist, error);
  }

  try {
    const output = await execute();
    action.status = "completed";
    await persist(action);
    return output;
  } catch (error) {
    if (
      (["create_local_spec", "update_local_spec"].includes(action.operation ?? "") && isKnownLocalSpecFailure(error))
      || (action.operation === "run_local_pipeline" && isKnownLocalPipelineFailure(error))
      // Refused before a supervisor started (validation, parallel limit, launch failure).
      || (["start_local_pipeline", "cancel_local_pipeline"].includes(action.operation ?? "")
        && (isBackgroundRunFailure(error) || isKnownLocalPipelineFailure(error)))
      // Downloads resume into a content-addressed cache; retrying is safe.
      || error instanceof LocalPrefetchError
    ) {
      action.status = "failed";
      try {
        await persist(action);
      } catch (persistError) {
        action.status = "outcome_unknown";
        try { await persist(action); } catch { /* Keep the durable executing record fail-closed. */ }
        throw new Error(
          "The local action did not complete, but its final status could not be recorded. Inspect the workspace and local runs before preparing another action.",
          { cause: persistError },
        );
      }
      throw error;
    }
    action.status = "outcome_unknown";
    try {
      await persist(action);
    } catch {
      // The already-persisted executing record remains fail-closed.
    }
    const detail = error instanceof Error ? error.message : String(error);
    const inspectionTarget = ["create_local_spec", "update_local_spec"].includes(action.operation ?? "")
      ? "local workspace"
      : action.operation === "run_local_pipeline"
        ? "local run store"
        : action.operation === "start_local_pipeline" || action.operation === "cancel_local_pipeline"
          ? "background runs (tt pipeline list)"
        : action.operation === "prefetch_local"
          ? "local Hugging Face cache (tt datasets verify / tt models verify-base)"
          : "local workspace";
    throw new Error(
      `The mutation outcome is unknown and this action cannot be retried automatically. Inspect the ${inspectionTarget} before preparing another action. ${detail}`,
      { cause: error },
    );
  }
}

export async function rejectPreparedAction(
  action: AgentAction,
  persist: PersistAction,
): Promise<void> {
  if (action.status !== "proposed") {
    throw new Error(`Action ${action.id} is not proposed and cannot be rejected.`);
  }
  action.status = "rejected";
  await persist(action);
}
