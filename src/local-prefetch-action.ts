import { createHash, randomUUID } from "node:crypto";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join, relative, sep } from "node:path";
import { resolveAdjacentConfig, type LocalPipelineCommandRunner } from "./local-pipeline-action.js";
import { inspectLocalSpec, SpecWorkspaceMismatchError } from "./spec-workspace.js";
import { parseLocalRunInput } from "./local-runtime/local-project.js";
import { configForSpec } from "./local-runtime/project-workflow.js";
import { parseLocalRunnerConfig } from "./local-runtime/orchestrator.js";
import { specLockPath } from "./local-runtime/assets.js";
import { huggingFaceRepoIdSchema, baseModelRevisionSchema, isRepoRelativePath } from "./local-runtime/contracts.js";
import { isCertifiedTrainingModel } from "./local-runtime/model-registry.js";

/**
 * Agent downloads. The model prepares a `prefetch_local` proposal; approval
 * runs the same `tt datasets prefetch` or `tt models prefetch` command a user
 * would.
 *
 * Without a source, the download is the spec's own dataset or base model and
 * also updates tunedtensor.lock.json. With a source, any Hugging Face repo can
 * be downloaded first and adopted into the spec afterwards; the spec only
 * supplies the cache location.
 */
export type PrefetchKind = "dataset" | "base_model";

export interface PrefetchSource {
  repo: string;
  revision?: string;
  training?: string;
  validation?: string;
  test?: string;
}

const SPLITS = ["training", "validation", "test"] as const;

export class LocalPrefetchError extends Error {
  constructor(message: string, cause?: unknown) {
    super(message, { cause });
    this.name = "LocalPrefetchError";
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function parseSource(kind: PrefetchKind, value: unknown): PrefetchSource | undefined {
  if (value === undefined || value === null) return undefined;
  if (!isRecord(value)) throw new Error("Download source must be an object.");
  const repo = huggingFaceRepoIdSchema.parse(value.repo);
  const revision = value.revision === undefined ? undefined : baseModelRevisionSchema.parse(value.revision);
  const files = Object.fromEntries(SPLITS.flatMap((split) => {
    const path = value[split];
    if (path === undefined) return [];
    if (typeof path !== "string" || !isRepoRelativePath(path)) {
      throw new Error(`source.${split} must be a file path inside the repo, such as data/${split}.jsonl.`);
    }
    return [[split, path]];
  }));
  if (kind === "dataset" && !files.training) {
    throw new Error("A dataset download needs source.training (a repo-relative .jsonl or .parquet file); validation and test are optional.");
  }
  if (kind === "base_model" && Object.keys(files).length > 0) {
    throw new Error("A base-model download takes only source.repo and source.revision.");
  }
  return { repo, ...(revision ? { revision } : {}), ...files };
}

async function inspect(workspaceRoot: string, kind: PrefetchKind, specPath = "tunedtensor.json", source?: PrefetchSource) {
  const current = await inspectLocalSpec(workspaceRoot, specPath, { allowMissing: Boolean(source) });
  const spec = current.validation.valid && isRecord(current.document) ? current.document : undefined;
  const engine = spec?.engine ?? "adapter";
  if (!source) {
    if (!spec) throw new Error(`Fix ${current.displayPath} before downloading: ${current.validation.errors.join("; ")}`);
    if (engine !== "adapter") {
      throw new Error(`Downloads of the spec's own sources apply to adapter specs; this spec uses engine "${String(engine)}", which fetches what it needs during \`tt pipeline run\`. Pass a source to download a specific repo.`);
    }
  }
  const dataset = isRecord(spec?.dataset_prebuilt) ? spec.dataset_prebuilt : undefined;
  const specDataset = isRecord(dataset?.huggingface) ? dataset.huggingface : undefined;
  if (!source && kind === "dataset" && !specDataset) {
    throw new Error("This spec has no dataset_prebuilt.huggingface source. Pass a source to download a dataset, then propose a spec edit to train on it.");
  }
  const repo = source?.repo ?? String(kind === "dataset" ? specDataset!.repo : spec!.base_model);
  const preview = kind === "dataset"
    ? {
      repo,
      revision: source ? source.revision ?? "latest (resolved by the download)" : specDataset!.revision ?? "main (unpinned)",
      files: Object.fromEntries(SPLITS
        .map((split) => [split, source ? source[split] : dataset![split]] as const)
        .filter(([, path]) => typeof path === "string")),
    }
    : {
      base_model: repo,
      ...(source ? {
        revision: source.revision ?? (isCertifiedTrainingModel(repo) ? "TT-certified revision" : "latest (resolved by the download)"),
        certified: isCertifiedTrainingModel(repo),
      } : {}),
    };

  // Adapter specs supply the download cache location; other workspaces use the default cache.
  let sealedSpec: Record<string, unknown> | null = null;
  let configSha256: string | null = null;
  if (spec && engine === "adapter") {
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
    sealedSpec = { ...input.spec, runtime, evaluation };
    configSha256 = legacy ? createHash("sha256").update(legacy.source).digest("hex") : null;
  }
  return { current, preview, repo, sealedSpec, configSha256 };
}

function command(kind: PrefetchKind, specPath: string, source?: PrefetchSource): string[] {
  const args = [kind === "dataset" ? "datasets" : "models", "prefetch", specPath];
  if (!source) return args;
  args.push("--repo", source.repo);
  if (source.revision) args.push("--revision", source.revision);
  for (const split of SPLITS) if (source[split]) args.push(`--${split}`, source[split]!);
  return args;
}

export async function prepareLocalPrefetch(workspaceRoot: string, kind: PrefetchKind, specPath?: string, rawSource?: unknown) {
  const source = parseSource(kind, rawSource);
  const { current, preview, repo, configSha256 } = await inspect(workspaceRoot, kind, specPath, source);
  const what = kind === "dataset" ? `dataset ${repo}` : `base model ${repo}`;
  return {
    summary: source
      ? `Download the ${what} into the local Hugging Face cache. The spec is not changed.`
      : `Download the ${what} for ${current.displayPath} into the local Hugging Face cache and pin it in tunedtensor.lock.json.`,
    args: {
      kind, spec_path: current.displayPath, spec_sha256: current.sha256,
      workspace_fingerprint: current.workspaceFingerprint, config_sha256: configSha256,
      ...(source ? { source } : {}),
    },
    preview: {
      ...preview,
      command: ["tt", ...command(kind, current.displayPath, source)],
      note: source
        ? "Downloads only. To train on it, propose a spec edit with the revision the download reports."
        : "Large downloads can take a while. Rerunning resumes from files already downloaded.",
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
  const source = parseSource(kind, input.source);
  const { current, repo, sealedSpec, configSha256 } = await inspect(workspaceRoot, kind, input.spec_path, source);
  if (current.workspaceFingerprint !== input.workspace_fingerprint) {
    throw new SpecWorkspaceMismatchError("The local workspace changed after this download was prepared; return to the original workspace and prepare it again.");
  }
  if (current.sha256 !== input.spec_sha256) {
    throw new Error("The spec changed after this download was prepared; prepare it again.");
  }
  if (configSha256 !== input.config_sha256) {
    throw new Error("The adjacent local-runner.json changed after this download was prepared; prepare it again.");
  }
  return {
    kind, source, repo, sealedSpec,
    specPath: relative(current.root, current.path).split(sep).join("/"),
    specDirectory: dirname(current.path),
    cwd: current.root,
  };
}

export async function executeLocalPrefetch(
  prepared: Awaited<ReturnType<typeof revalidateLocalPrefetch>>,
  runCommand: LocalPipelineCommandRunner,
  signal?: AbortSignal,
) {
  const args = command(prepared.kind, prepared.specPath, prepared.source);
  // The child must consume the reviewed bytes, not reopen a mutable spec or
  // adjacent config after the durable approval claim. The snapshot sits beside
  // the spec so a spec download updates the spec's own tunedtensor.lock.json.
  // Without an adapter spec, a path that does not exist selects the default cache.
  const directory = prepared.sealedSpec ? undefined : await mkdtemp(join(tmpdir(), "tt-agent-prefetch-"));
  const snapshot = join(directory ?? prepared.specDirectory, `.${basename(prepared.specPath)}.tt-agent-${randomUUID()}.json`);
  try {
    if (prepared.sealedSpec) {
      await writeFile(snapshot, JSON.stringify(prepared.sealedSpec), { mode: 0o600, flag: "wx" });
    }
    const result = await runCommand(command(prepared.kind, snapshot, prepared.source), { cwd: prepared.cwd, signal, captureJson: true });
    if (result.exitCode !== 0) {
      const suffix = result.signal ? ` after ${result.signal}` : "";
      throw new LocalPrefetchError(
        `tt ${args.join(" ")} exited with code ${result.exitCode ?? 1}${suffix}. Approving a new download resumes from the files already fetched.`,
      );
    }
    // Bind the adoption result to this command, never the latest global manifest
    // (another download may share the repo name, revision or cache).
    const report: unknown = result.stdout === undefined ? undefined : JSON.parse(result.stdout);
    let download: Record<string, unknown> = {};
    if (report !== undefined) {
      if (!isRecord(report) || (report.repo ?? report.base_model) !== prepared.repo) {
        throw new Error("The download returned a report for a different repository.");
      }
      if (report.status === "skipped" && prepared.kind === "base_model" && !prepared.source) {
        download = { status: "skipped", local_base_model_path: report.local_base_model_path, reason: report.reason };
      } else {
        const revision = baseModelRevisionSchema.parse(report.revision ?? report.snapshot_revision);
        if (prepared.source?.revision && revision !== prepared.source.revision) {
          throw new Error("The download returned a different revision than requested.");
        }
        download = {
          revision, snapshot_path: report.snapshot_path, manifest_path: report.manifest_path,
          ...(prepared.kind === "base_model" ? {
            file_count: report.file_count,
            size_bytes: report.size_bytes,
            checksums_verified: typeof report.file_count === "number" && report.file_count > 0
              && report.verified_blob_count === report.file_count,
          } : {}),
          ...(prepared.kind === "dataset" && isRecord(report.files)
            ? { files: Object.values(report.files).flatMap((file) => isRecord(file) && typeof file.path === "string" ? [file.path] : []) }
            : {}),
        };
      }
    }
    return {
      completed: true,
      kind: prepared.kind,
      command: ["tt", ...args],
      repo: prepared.repo,
      ...download,
      ...(prepared.source || download.status === "skipped" ? {} : { lock_path: specLockPath(join(prepared.cwd, prepared.specPath)) }),
    };
  } catch (error) {
    if (error instanceof LocalPrefetchError) throw error;
    throw new LocalPrefetchError(error instanceof Error ? error.message : String(error), error);
  } finally {
    await rm(snapshot, { force: true });
    if (directory) await rm(directory, { recursive: true, force: true });
  }
}
