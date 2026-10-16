import {
  prepareLocalPipelineAction,
  sealPreparedSpec,
  validatePreparedLocalPipelineAction,
  type PreparedLocalPipelineAction,
} from "./local-pipeline-action.js";
import {
  BackgroundRunError,
  backgroundRunsRoot,
  createBackgroundRunStore,
  describeBackgroundRun,
  launchBackgroundSupervisor,
  maxParallelRuns,
  isActiveBackgroundRun,
  normalizeReason,
  summarizeResources,
  type BackgroundRunRecord,
  type BackgroundRunRequester,
  type BackgroundRunStore,
  type ResourceSummary,
} from "./local-runtime/background-runs.js";
import { sampleLiveUsage, type LiveUsage } from "./local-runtime/live-usage.js";

/**
 * Approval-side half of agent background runs. The model can only prepare a
 * proposal; approval revalidates the reviewed spec bytes and hands the sealed
 * spec to a detached supervisor, so the conversation continues immediately.
 */
export interface BackgroundRunExecutor {
  store: BackgroundRunStore;
  /** Start the detached supervisor for a created record. */
  launch(run: BackgroundRunRecord): Promise<BackgroundRunRecord>;
  maxParallel(): number;
  sampleUsage?: () => Promise<LiveUsage>;
}

export interface PreparedBackgroundRunStart {
  arguments: {
    pipeline: unknown;
    spec_path: string;
    spec_sha256: string;
    workspace_fingerprint: string;
    config_path: string | null;
    config_sha256: string | null;
    reason: string;
  };
  preview: Record<string, unknown>;
  engine: PreparedLocalPipelineAction["engine"];
}

function targetFor(prepared: PreparedLocalPipelineAction): string {
  const gpu = prepared.resolvedConfig?.gpu ?? prepared.resolvedSpec.runtime?.gpu;
  return gpu && typeof gpu === "object" && "instanceId" in gpu && typeof gpu.instanceId === "string"
    ? `aws:${gpu.instanceId}`
    : "local";
}

async function sampleResources(executor: BackgroundRunExecutor): Promise<ResourceSummary | undefined> {
  try {
    return executor.sampleUsage ? summarizeResources(await executor.sampleUsage()) : undefined;
  } catch {
    return undefined;
  }
}

function assertTrainable(prepared: PreparedLocalPipelineAction): void {
  if (prepared.resolvedConfig?.dryRun) {
    throw new BackgroundRunError("The spec's runtime sets dryRun, so a real run would only preview. Remove runtime.dryRun from tunedtensor.json before starting a run.");
  }
}

export async function prepareBackgroundRunStart(args: {
  workspaceRoot: string;
  specPath?: string;
  pipeline?: unknown;
  reason: unknown;
  executor: BackgroundRunExecutor;
}): Promise<PreparedBackgroundRunStart> {
  const reason = normalizeReason(args.reason);
  const prepared = await prepareLocalPipelineAction({
    workspaceRoot: args.workspaceRoot,
    pipeline: args.pipeline,
    specPath: args.specPath,
    dryRun: false,
    allowReal: true,
  });
  assertTrainable(prepared);
  const limit = args.executor.maxParallel();
  const active = await args.executor.store.active();
  if (active.length >= limit) {
    throw new BackgroundRunError(
      `${active.length} background run${active.length === 1 ? " is" : "s are"} already active (limit ${limit}). `
      + "Check them (tt pipeline status, or get_background_run in the agent) or cancel one before starting another; TT_MAX_PARALLEL_RUNS raises the limit.",
    );
  }
  const target = targetFor(prepared);
  const resources = await sampleResources(args.executor);
  const warnings: string[] = [];
  const sharing = active.filter((run) => run.choices.target === target);
  if (sharing.length) {
    warnings.push(`Run${sharing.length > 1 ? "s" : ""} ${sharing.map((run) => run.id.slice(0, 8)).join(", ")} already use ${target}; parallel runs share its GPU memory and may run out of memory or slow each other down.`);
  }
  if (target !== "local") {
    warnings.push("Resource figures describe this laptop; GPU work runs on the AWS instance. Keep the laptop awake until outputs return.");
  }
  return {
    engine: prepared.engine,
    arguments: {
      pipeline: prepared.pipeline,
      spec_path: prepared.specPath,
      spec_sha256: prepared.specSha256,
      workspace_fingerprint: prepared.workspaceFingerprint,
      config_path: prepared.configPath ?? null,
      config_sha256: prepared.configSha256 ?? null,
      reason,
    },
    preview: {
      plan: prepared.plan,
      engine: prepared.engine,
      spec_path: prepared.specPath,
      spec_sha256: prepared.specSha256,
      target,
      reason,
      execution: "Starts in the background after approval; the conversation continues and the run survives closing tt.",
      active_runs: active.map((run) => ({ id: run.id, status: run.status, target: run.choices.target })),
      max_parallel_runs: limit,
      ...(resources ? { resources } : {}),
      ...(warnings.length ? { warnings } : {}),
    },
  };
}

export async function revalidateBackgroundRunStart(workspaceRoot: string, input: Record<string, unknown>): Promise<PreparedLocalPipelineAction> {
  const string = (value: unknown, label: string) => {
    if (typeof value !== "string" || !value) throw new BackgroundRunError(`Prepared run ${label} is invalid.`);
    return value;
  };
  normalizeReason(input.reason);
  const configPath = input.config_path ?? undefined;
  const configSha256 = input.config_sha256 ?? undefined;
  if ((configPath === undefined) !== (configSha256 === undefined)) {
    throw new BackgroundRunError("Prepared run config path and fingerprint must be provided together.");
  }
  const prepared = await validatePreparedLocalPipelineAction({
    workspaceRoot,
    pipeline: input.pipeline,
    specPath: string(input.spec_path, "spec path"),
    expectedSpecSha256: string(input.spec_sha256, "spec fingerprint"),
    expectedWorkspaceFingerprint: string(input.workspace_fingerprint, "workspace fingerprint"),
    expectedConfigPath: configPath === undefined ? undefined : string(configPath, "config path"),
    expectedConfigSha256: configSha256 === undefined ? undefined : string(configSha256, "config fingerprint"),
    dryRun: false,
    allowReal: true,
  });
  assertTrainable(prepared);
  return prepared;
}

export async function executeBackgroundRunStart(args: {
  workspaceRoot: string;
  input: Record<string, unknown>;
  executor: BackgroundRunExecutor;
  requestedBy: BackgroundRunRequester;
}): Promise<Record<string, unknown>> {
  const prepared = await revalidateBackgroundRunStart(args.workspaceRoot, args.input);
  const run = await args.executor.store.create({
    reason: normalizeReason(args.input.reason),
    requested_by: args.requestedBy,
    choices: {
      engine: prepared.engine,
      spec_path: prepared.specPath,
      spec_sha256: prepared.specSha256,
      config_path: prepared.configPath ?? null,
      config_sha256: prepared.configSha256 ?? null,
      steps: prepared.plan.steps.map((step) => ({ id: step.id, uses: step.uses, target: step.target })),
      target: targetFor(prepared),
      workspace_root: prepared.workspaceRoot,
      ...("base_model" in prepared.resolvedSpec && typeof prepared.resolvedSpec.base_model === "string"
        ? { base_model: prepared.resolvedSpec.base_model }
        : {}),
    },
    sealed_spec: sealPreparedSpec(prepared),
    resources_at_launch: await sampleResources(args.executor),
  }, { maxParallel: args.executor.maxParallel() });
  const launched = await args.executor.launch(run);
  return { background_run_id: launched.id, ...describeBackgroundRun(launched) };
}

export async function prepareBackgroundRunCancel(args: {
  id: string;
  reason: unknown;
  store: BackgroundRunStore;
}): Promise<{ arguments: Record<string, unknown>; preview: Record<string, unknown>; run: BackgroundRunRecord }> {
  const reason = normalizeReason(args.reason);
  const run = await args.store.get(args.id);
  if (!isActiveBackgroundRun(run)) {
    throw new BackgroundRunError(`Background run ${run.id.slice(0, 8)} is already ${run.status}; nothing to cancel.`);
  }
  return {
    run,
    arguments: { background_run_id: run.id, reason },
    preview: { ...describeBackgroundRun(run), cancel_reason: reason },
  };
}

export async function executeBackgroundRunCancel(args: {
  input: Record<string, unknown>;
  store: BackgroundRunStore;
  requestedBy: BackgroundRunRequester;
}): Promise<Record<string, unknown>> {
  const id = args.input.background_run_id;
  if (typeof id !== "string" || !/^[0-9a-f-]{36}$/.test(id)) throw new BackgroundRunError("Prepared cancellation run ID is invalid.");
  const run = await args.store.requestCancel(id, { reason: normalizeReason(args.input.reason), requested_by: args.requestedBy });
  return { background_run_id: run.id, status: run.status, note: "Stopping; the run records cancelled once its process exits." };
}

export function isBackgroundRunFailure(error: unknown): boolean {
  return error instanceof BackgroundRunError;
}

/** The executor the shell uses: the laptop-wide run store and a detached `tt pipeline supervise`. */
export function createBackgroundRunExecutor(args: {
  env: NodeJS.ProcessEnv;
  entrypoint: string | undefined;
}): BackgroundRunExecutor {
  const store = createBackgroundRunStore(backgroundRunsRoot(args.env));
  return {
    store,
    maxParallel: () => maxParallelRuns(args.env),
    sampleUsage: async () => await sampleLiveUsage({ env: args.env }),
    launch: async (run) => {
      if (!args.entrypoint) throw new BackgroundRunError("Cannot locate the tt CLI entrypoint.");
      return await launchBackgroundSupervisor({ store, run, entrypoint: args.entrypoint, env: args.env });
    },
  };
}
