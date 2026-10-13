import { relative, sep } from "node:path";
import type { LocalPipelineCommandRunner } from "./local-pipeline-action.js";
import { inspectLocalSpec } from "./spec-workspace.js";

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
  return { current, preview };
}

function command(kind: PrefetchKind, specPath: string): string[] {
  return [kind === "dataset" ? "datasets" : "models", "prefetch", specPath];
}

export async function prepareLocalPrefetch(workspaceRoot: string, kind: PrefetchKind, specPath?: string) {
  const { current, preview } = await inspect(workspaceRoot, kind, specPath);
  const what = kind === "dataset" ? `dataset ${String(preview.repo)}` : `base model ${String(preview.base_model)}`;
  return {
    summary: `Download the ${what} for ${current.displayPath} into the local Hugging Face cache.`,
    args: { kind, spec_path: current.displayPath, spec_sha256: current.sha256 },
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
  const { current } = await inspect(workspaceRoot, kind, input.spec_path);
  if (current.sha256 !== input.spec_sha256) {
    throw new Error("The spec changed after this download was prepared; prepare it again.");
  }
  return { kind, specPath: relative(current.root, current.path).split(sep).join("/"), cwd: current.root };
}

export async function executeLocalPrefetch(
  prepared: Awaited<ReturnType<typeof revalidateLocalPrefetch>>,
  runCommand: LocalPipelineCommandRunner,
  signal?: AbortSignal,
) {
  const args = command(prepared.kind, prepared.specPath);
  const result = await runCommand(args, { cwd: prepared.cwd, signal });
  if (result.exitCode !== 0) {
    const suffix = result.signal ? ` after ${result.signal}` : "";
    throw new LocalPrefetchError(
      `tt ${args.join(" ")} exited with code ${result.exitCode ?? 1}${suffix}. Approving a new download resumes from the files already fetched.`,
    );
  }
  return { completed: true, kind: prepared.kind, command: ["tt", ...args] };
}
