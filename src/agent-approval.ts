import type { AgentAction } from "./agent-client.js";
import {
  createLocalSpecProject,
  isKnownLocalSpecFailure,
  type LocalSpecFileOperations,
} from "./local-spec-workspace.js";
import {
  executeLocalPipelineAction,
  isKnownLocalPipelineFailure,
  type LocalPipelineCommandRunner,
} from "./local-pipeline-action.js";

export interface AgentMutationApi {
  get(path: string): Promise<unknown>;
  post(path: string, body?: unknown, guard?: AgentMutationGuard): Promise<unknown>;
  put(path: string, body?: unknown, guard?: AgentMutationGuard): Promise<unknown>;
}

export interface AgentMutationGuard {
  actionId: string;
  operation: "create_spec" | "update_spec";
  expectedUpdatedAt?: string;
}

export type PersistAction = (action: AgentAction) => Promise<void>;

export interface AgentApprovalOptions {
  workspaceRoot?: string;
  localFileOperations?: Partial<LocalSpecFileOperations>;
  runPipelineCommand?: LocalPipelineCommandRunner;
  signal?: AbortSignal;
}

function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

function data(value: unknown): unknown {
  return record(value)?.data ?? value;
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

async function requireMutationGuardSupport(api: AgentMutationApi): Promise<void> {
  const version = record(data(await api.get("/version")));
  const capabilities = version?.capabilities;
  if (
    !Array.isArray(capabilities) ||
    !capabilities.includes("local_agent_spec_mutation_guards_v1")
  ) {
    throw new Error(
      "The Tuned Tensor API does not advertise local-agent mutation guard support. Refusing to mutate; update the server or use an explicit direct CLI command.",
    );
  }
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

export async function approvePreparedAction(
  action: AgentAction,
  api: AgentMutationApi,
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
      case "create_local_spec": {
        const workspaceRoot = options.workspaceRoot;
        if (!workspaceRoot) {
          throw new Error("A local workspace is required to create a local spec.");
        }
        const directory = requiredString(input.directory, "local directory");
        const workspaceFingerprint = requiredString(
          input.workspace_fingerprint,
          "workspace fingerprint",
        );
        execute = async () => await createLocalSpecProject(
          workspaceRoot,
          directory,
          input.spec,
          workspaceFingerprint,
          options.localFileOperations,
        );
        break;
      }
      case "run_local_pipeline": {
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
        const dryRun = requiredBoolean(input.dry_run, "dry-run flag");
        execute = async () => await executeLocalPipelineAction({
          workspaceRoot,
          pipeline,
          specPath,
          expectedSpecSha256: specSha256,
          dryRun,
          runCommand: options.runPipelineCommand!,
          signal: options.signal,
        });
        break;
      }
      case "create_spec": {
        await requireMutationGuardSupport(api);
        const spec = record(input.spec);
        if (!spec) throw new Error("Prepared create spec body is invalid.");
        execute = async () => data(await api.post("/behavior-specs", spec, {
          actionId: action.id,
          operation: "create_spec",
        }));
        break;
      }
      case "update_spec": {
        await requireMutationGuardSupport(api);
        const specId = requiredString(input.spec_id, "spec ID");
        const expected = requiredString(input.expected_spec_updated_at, "spec version");
        const changes = record(input.changes);
        if (!changes) throw new Error("Prepared spec changes are invalid.");
        const current = record(data(await api.get(`/behavior-specs/${specId}`)));
        if (current?.updated_at !== expected) {
          throw new Error("The behaviour spec changed after this action was prepared; prepare it again.");
        }
        execute = async () => data(await api.put(`/behavior-specs/${specId}`, changes, {
          actionId: action.id,
          operation: "update_spec",
          expectedUpdatedAt: expected,
        }));
        break;
      }

      default:
        throw new Error(`Unsupported prepared operation: ${String(action.operation)}.`);
    }

  } catch (error) {
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
      (action.operation === "create_local_spec" && isKnownLocalSpecFailure(error))
      || (action.operation === "run_local_pipeline" && isKnownLocalPipelineFailure(error))
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
    const inspectionTarget = action.operation === "create_local_spec"
      ? "local workspace"
      : action.operation === "run_local_pipeline"
        ? "local run store"
        : "remote resource";
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
