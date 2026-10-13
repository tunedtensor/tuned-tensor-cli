import { createHash } from "node:crypto";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, relative, sep } from "node:path";
import { resolveAdjacentConfig, type LocalPipelineCommandRunner } from "./local-pipeline-action.js";
import { inspectLocalSpec, SpecWorkspaceMismatchError } from "./spec-workspace.js";
import { parseLocalRunInput } from "./local-runtime/local-project.js";
import { configForSpec } from "./local-runtime/project-workflow.js";
import { parseLocalRunnerConfig } from "./local-runtime/orchestrator.js";

/**
 * Agent downloads. The model prepares a `prefetch_local` proposal; approval
 * runs the same `tt datasets prefetch` or `tt models prefetch` command a user
 * would, against the spec version they reviewed.
 */
export type PrefetchKind = "dataset" | "base_model";

export class LocalPrefetchError extends Error {
  constructor(message: string, cause?: unknown) {
    super(message, { cause });
    this.name = "LocalPrefetchError";
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

async function inspect(workspaceRoot: string, kind: PrefetchKind, specPath = "tunedtensor.json") {
  const current = await inspectLocalSpec(workspaceRoot, specPath);
  if (!current.validation.valid || !isRecord(current.document)) {
    throw new Error(`Fix ${current.displayPath} before downloading: ${current.validation.errors.join("; ")}`);
  }
  const spec = current.document;
  const engine = spec.engine ?? "adapter";
  if (engine !== "adapter") {
    throw new Error(`Downloads apply to adapter specs; this spec uses engine "${String(engine)}", which fetches what it needs during \`tt pipeline run\`.`);
  }
  const dataset = isRecord(spec.dataset_prebuilt) ? spec.dataset_prebuilt : undefined;
  const source = isRecord(dataset?.huggingface) ? dataset.huggingface : undefined;
  if (kind === "dataset" && !source) {
    throw new Error("This spec has no dataset_prebuilt.huggingface source. Propose a spec edit that sets one before downloading.");
  }
  const preview = kind === "dataset"
    ? {
      repo: source!.repo,
      revision: source!.revision ?? "main (unpinned)",
      files: Object.fromEntries((["training", "validation", "test"] as const)
        .filter((split) => typeof dataset![split] === "string")
        .map((split) => [split, dataset![split]])),
    }
    : { base_model: spec.base_model };
  const input = parseLocalRunInput(spec, current.path);
  if (input.kind !== "spec") throw new Error("Downloads require an adapter spec.");
  const legacy = await resolveAdjacentConfig({ workspaceRoot: current.root, specPath: current.path });
  if (legacy && (input.spec.runtime || input.spec.evaluation)) {
    throw new Error("Conflicting configuration sources: migrate local-runner.json into tunedtensor.json.");
  }
  const config = legacy
    ? parseLocalRunnerConfig(JSON.parse(legacy.source.toString("utf8")), legacy.path)
    : configForSpec(input.spec, current.path);
  const { evaluation, dryRun: _dryRun, ...runtime } = config;
  return {
    current, preview,
    sealedSpec: { ...input.spec, runtime, evaluation },
    configSha256: legacy ? createHash("sha256").update(legacy.source).digest("hex") : null,
  };
}

function command(kind: PrefetchKind, specPath: string): string[] {
  return [kind === "dataset" ? "datasets" : "models", "prefetch", specPath];
}

export async function prepareLocalPrefetch(workspaceRoot: string, kind: PrefetchKind, specPath?: string) {
  const { current, preview, configSha256 } = await inspect(workspaceRoot, kind, specPath);
  const what = kind === "dataset" ? `dataset ${String(preview.repo)}` : `base model ${String(preview.base_model)}`;
  return {
    summary: `Download the ${what} for ${current.displayPath} into the local Hugging Face cache.`,
    args: {
      kind, spec_path: current.displayPath, spec_sha256: current.sha256,
      workspace_fingerprint: current.workspaceFingerprint, config_sha256: configSha256,
    },
    preview: {
      ...preview,
      command: ["tt", ...command(kind, current.displayPath)],
      note: "Large downloads can take a while. Rerunning resumes from files already downloaded.",
    },
  };
}

/** Re-read the spec at approval so the download matches what the user reviewed. */
export async function revalidateLocalPrefetch(workspaceRoot: string, input: Record<string, unknown>) {
  const kind = input.kind as PrefetchKind;
  if (kind !== "dataset" && kind !== "base_model") throw new Error("Prepared download kind is invalid.");
  if (typeof input.spec_path !== "string" || typeof input.spec_sha256 !== "string") {
    throw new Error("Prepared download spec is invalid.");
  }
  const { current, sealedSpec, configSha256 } = await inspect(workspaceRoot, kind, input.spec_path);
  if (current.workspaceFingerprint !== input.workspace_fingerprint) {
    throw new SpecWorkspaceMismatchError("The local workspace changed after this download was prepared; return to the original workspace and prepare it again.");
  }
  if (current.sha256 !== input.spec_sha256) {
    throw new Error("The spec changed after this download was prepared; prepare it again.");
  }
  if (configSha256 !== input.config_sha256) {
    throw new Error("The adjacent local-runner.json changed after this download was prepared; prepare it again.");
  }
  return { kind, specPath: relative(current.root, current.path).split(sep).join("/"), cwd: current.root, sealedSpec };
}

export async function executeLocalPrefetch(
  prepared: Awaited<ReturnType<typeof revalidateLocalPrefetch>>,
  runCommand: LocalPipelineCommandRunner,
  signal?: AbortSignal,
) {
  const args = command(prepared.kind, prepared.specPath);
  let directory: string | undefined;
  try {
    // The child must consume the reviewed bytes, not reopen a mutable spec or
    // adjacent config after the durable approval claim. Paths are resolved
    // against the original spec before writing this private snapshot.
    directory = await mkdtemp(join(tmpdir(), "tt-agent-prefetch-"));
    const specPath = join(directory, "tunedtensor.json");
    await writeFile(specPath, JSON.stringify(prepared.sealedSpec), { mode: 0o600, flag: "wx" });
    const result = await runCommand(command(prepared.kind, specPath), { cwd: prepared.cwd, signal });
    if (result.exitCode !== 0) {
      const suffix = result.signal ? ` after ${result.signal}` : "";
      throw new LocalPrefetchError(
        `tt ${args.join(" ")} exited with code ${result.exitCode ?? 1}${suffix}. Approving a new download resumes from the files already fetched.`,
      );
    }
    return { completed: true, kind: prepared.kind, command: ["tt", ...args] };
  } catch (error) {
    if (error instanceof LocalPrefetchError) throw error;
    throw new LocalPrefetchError(error instanceof Error ? error.message : String(error), error);
  } finally {
    if (directory) await rm(directory, { recursive: true, force: true });
  }
}
