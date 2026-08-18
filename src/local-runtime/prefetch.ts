import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { lstat, mkdir, readFile, readdir, realpath, stat } from "node:fs/promises";
import { isAbsolute, join, relative, resolve, sep } from "node:path";
import {
  baseModelRevisionSchema,
  type FineTuneRunRequest,
  type LocalRunnerConfig,
} from "./contracts.js";
import { fileUri, writeJson } from "./artifacts.js";
import {
  assertCertifiedBaseModelConfig,
  defaultBaseModelRevision,
  resolveRequestedBaseModelRevision,
  resolveTrainingModel,
} from "./model-registry.js";
import {
  buildBundledPythonCommand,
  runLoggedProcess,
  withBundledPythonEnvironment,
} from "./process-runner.js";
import type { LocalRunReporter } from "./run-reporter.js";
import {
  minimalMachineLearningEnvironment,
  resolveHuggingFaceCacheLayout,
  withHuggingFaceCacheEnvironment,
} from "./huggingface-cache.js";

export interface ModelPrefetchPayload {
  base_model: string;
  revision?: string;
  model_cache?: string;
  local_files_only?: boolean;
}

export interface ModelPrefetchReport {
  ok: boolean;
  status: "completed" | "skipped";
  base_model: string;
  model_cache?: string;
  hf_home?: string;
  hub_cache?: string;
  snapshot_path?: string;
  snapshot_revision?: string;
  file_count?: number;
  size_bytes?: number;
  verified_blob_count?: number;
  local_base_model_path?: string;
  artifact_dir: string;
  input_path?: string;
  output_path?: string;
  log_uri?: string;
  command?: string[];
  reason?: string;
}

type CertifiedFileDigest = { algorithm: "sha1" | "sha256"; digest: string };

const NEMOTRON_BF16_FILE_DIGESTS: Readonly<Record<string, CertifiedFileDigest>> = {
  ".gitattributes": { algorithm: "sha1", digest: "fe3f694ce14b577c9833fca6e83c5e26a7119127" },
  "LICENSE": { algorithm: "sha1", digest: "4d76cc87c9edb280753e3bee3b499ae461113835" },
  "README.md": { algorithm: "sha1", digest: "8ec43ee9f3ee345202e4a6ef752c51b2619f7425" },
  "accuracy_plot.png": { algorithm: "sha256", digest: "1397995f8a5d34d819a069d86c02914990e9ecfc2fce101e233924da9cf5ddcf" },
  "agentic_coding_benchmarks.png": { algorithm: "sha256", digest: "06cf5486ae94ecd7cc6f91ae3b4bae21c7435253a294cb89f54f7ec42eda8847" },
  "bias.md": { algorithm: "sha1", digest: "cdcc8055f965222fabddabcabf22885d3a8bacc1" },
  "chat_template.jinja": { algorithm: "sha1", digest: "d85b0c772f8fe585063847c5f6bf5ec48eb210be" },
  "config.json": { algorithm: "sha1", digest: "993b612989e6aa67da45f908ea5983cb777e7678" },
  "explainability.md": { algorithm: "sha1", digest: "2435f23c865e7daba6a6eceef2f214b86df75d24" },
  "generation_config.json": { algorithm: "sha1", digest: "a41201df53f9d0769947989353b5d02e70491c56" },
  "model.safetensors.index.json": { algorithm: "sha1", digest: "3cd409443b1896d9adc5c3981af301841f85329a" },
  "special_tokens_map.json": { algorithm: "sha1", digest: "0451f37912edb7f8a4bf04d77eef13f6130f0515" },
  "tokenizer.json": { algorithm: "sha256", digest: "623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7" },
  "tokenizer_config.json": { algorithm: "sha1", digest: "c96e5ad00986bfa0b666d4bc425666f1c51d6d1b" },
  "model-00001-of-00014.safetensors": { algorithm: "sha256", digest: "7a3e74d24969eac6657cb9b6422091fd062b7d30fd4bc90ecee6c3d7f692626a" },
  "model-00002-of-00014.safetensors": { algorithm: "sha256", digest: "a80c3846e959f6dda34aa031cbb668b766643691ac7e5c9f43f62e6fb1221059" },
  "model-00003-of-00014.safetensors": { algorithm: "sha256", digest: "996bf3a46668e14075cb82259390267c2b1f71446caeb3a5f1f9e0bd7a079ae7" },
  "model-00004-of-00014.safetensors": { algorithm: "sha256", digest: "cce1d79e7ff2e2b746b9d97166156aa1dcec796116675cc7fd1aa70a8d73ccb2" },
  "model-00005-of-00014.safetensors": { algorithm: "sha256", digest: "9338f59d3fd5f074288c71a9559452d5b5c581da00e9ccb9fe7aa2ce9b04d2ad" },
  "model-00006-of-00014.safetensors": { algorithm: "sha256", digest: "e947fa4097c68263ba3d097fceb32a300a22cebd9d909fb5198af7d18a4022fe" },
  "model-00007-of-00014.safetensors": { algorithm: "sha256", digest: "c7362630bf4256275ca58354ed8f31506cec67ffab30573e753703fd5ae48795" },
  "model-00008-of-00014.safetensors": { algorithm: "sha256", digest: "1b33a5a6b5505b25b57c90ddda19e9cd1598176468917df0c56116fa6f6072cb" },
  "model-00009-of-00014.safetensors": { algorithm: "sha256", digest: "fcc9b48caff427beaf1f87b78cf4ab42bc2a3c43b3c487bccb013ba22d7468b2" },
  "model-00010-of-00014.safetensors": { algorithm: "sha256", digest: "d1924fd54d1b77a15aff8df591a317e8a82a58e46d6f5742fa21e90e0be3cfa6" },
  "model-00011-of-00014.safetensors": { algorithm: "sha256", digest: "e9d0b83bcf04bd4d5758b94e6495a3f1595d121d82faffaa27ce6a884e2c8757" },
  "model-00012-of-00014.safetensors": { algorithm: "sha256", digest: "fce184f112cb0b97024cadc60f993f6f660e365d04e21c240fb3c75747bd3a01" },
  "model-00013-of-00014.safetensors": { algorithm: "sha256", digest: "f427d037d2a3e4d44323325aa38571e3551e8ad702f469c3dacc4e59ea2e50d1" },
  "model-00014-of-00014.safetensors": { algorithm: "sha256", digest: "64577b275ca4e7e5266eae0903674f7f46ec2a8cbf4f4f1a3207f80d503cd1d0" },
  "privacy.md": { algorithm: "sha1", digest: "e3bf30aa1c42a488298ef37383647dc99e1cd39e" },
  "safety.md": { algorithm: "sha1", digest: "f53bf94a4635862eed5cf3e0bc799b93f81d20ac" },
};

const MUSE_GLIMMER_FILE_DIGESTS: Readonly<Record<string, CertifiedFileDigest>> = {
  ".gitattributes": { algorithm: "sha1", digest: "52373fe24473b1aa44333d318f578ae6bf04b49b" },
  "LICENSE": { algorithm: "sha1", digest: "d645695673349e3947e8e5ae42332d0ac3164cd7" },
  "README.md": { algorithm: "sha1", digest: "a25f87c1c1fbd88f367a552fd14427baa8cb947e" },
  "USAGE_POLICY.md": { algorithm: "sha1", digest: "1a9ed6cffc54585daf9cc904c07cb6e436a9444b" },
  "chat_template.jinja": { algorithm: "sha1", digest: "7507f3c9f38809152732c045df3977848f3916a6" },
  "config.json": { algorithm: "sha1", digest: "190826dc834c13c86b8dc68e775a888a852d7c34" },
  "generation_config.json": { algorithm: "sha1", digest: "b69a50a4239f42707d54a479e0143b8ee56c8bcc" },
  "model.safetensors.index.json": { algorithm: "sha1", digest: "00b257ef0c8bce1b001a6266f6b1f66f4fe16a6b" },
  "processor_config.json": { algorithm: "sha1", digest: "ec9a07be13ea1dadf7395131871753b13f406c26" },
  "tokenizer_config.json": { algorithm: "sha1", digest: "fe7f0bb90fb0d0288e3a505974ae0e4baf1d3e61" },
  "tokenizer.json": { algorithm: "sha256", digest: "c9dbee66967b58f31a7c27f723c3760da3526ccd0427578e8905b0abb0031c4d" },
  "model-00001-of-00002.safetensors": { algorithm: "sha256", digest: "8eef61530e1283642c77ce2e6721feb5c6f348fa055c00e90f2844a136372694" },
  "model-00002-of-00002.safetensors": { algorithm: "sha256", digest: "b58cc2144ba1ba1af4420f67f4ca3ced7f09298510b80464cc75018a0be14381" },
};

function certifiedFileDigests(
  modelId: string,
): Readonly<Record<string, CertifiedFileDigest>> | undefined {
  switch (modelId) {
    case "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16":
      return NEMOTRON_BF16_FILE_DIGESTS;
    case "meta-models/Muse-Glimmer-30B":
      return MUSE_GLIMMER_FILE_DIGESTS;
    default:
      return undefined;
  }
}


async function certifiedFileDigest(path: string, expected: CertifiedFileDigest): Promise<string> {
  const metadata = await stat(path);
  const digest = createHash(expected.algorithm);
  if (expected.algorithm === "sha1") digest.update(`blob ${metadata.size}\0`);
  for await (const chunk of createReadStream(path)) digest.update(chunk);
  return digest.digest("hex");
}

function isStrictPhysicalDescendant(parent: string, child: string): boolean {
  const candidate = relative(parent, child);
  return candidate !== ""
    && candidate !== ".."
    && !candidate.startsWith(`..${sep}`)
    && !isAbsolute(candidate);
}

async function verifySnapshotCacheLocation(
  path: string,
  modelId: string,
  revision: string,
  modelCache: string,
): Promise<{ repositoryRoot: string; physicalBlobRoot: string }> {
  const repository = `models--${modelId.replaceAll("/", "--")}`;
  const layout = resolveHuggingFaceCacheLayout(modelCache);
  const repositoryRoot = join(layout.hubCache, repository);
  const expectedPath = join(repositoryRoot, "snapshots", revision);
  const [
    physicalCacheRoot,
    physicalHubRoot,
    physicalRepositoryRoot,
    physicalBlobRoot,
    physicalPath,
    physicalExpectedPath,
  ] = await Promise.all([
    realpath(layout.hfHome).catch(() => null),
    realpath(layout.hubCache).catch(() => null),
    realpath(repositoryRoot).catch(() => null),
    realpath(join(repositoryRoot, "blobs")).catch(() => null),
    realpath(path).catch(() => null),
    realpath(expectedPath).catch(() => null),
  ]);
  if (
    !physicalCacheRoot
    || !physicalHubRoot
    || !physicalRepositoryRoot
    || !physicalBlobRoot
    || !physicalPath
    || !physicalExpectedPath
    || physicalPath !== physicalExpectedPath
  ) {
    throw new Error(
      `${modelId} must use the certified Hugging Face cache snapshot ${expectedPath}; got ${path}.`,
    );
  }
  if (!isStrictPhysicalDescendant(physicalCacheRoot, physicalHubRoot)) {
    throw new Error(`Hugging Face hub cache escapes its configured cache root: ${layout.hubCache}`);
  }
  if (!isStrictPhysicalDescendant(physicalHubRoot, physicalRepositoryRoot)) {
    throw new Error(`Hugging Face repository escapes its configured hub cache: ${repositoryRoot}`);
  }
  if (!isStrictPhysicalDescendant(physicalRepositoryRoot, physicalBlobRoot)) {
    throw new Error(`Hugging Face blob store escapes its repository: ${join(repositoryRoot, "blobs")}`);
  }
  if (!isStrictPhysicalDescendant(physicalRepositoryRoot, physicalExpectedPath)) {
    throw new Error(`Hugging Face snapshot escapes its configured repository: ${expectedPath}`);
  }
  return { repositoryRoot, physicalBlobRoot };
}

async function requireReportedCachePath(
  label: string,
  reported: string | undefined,
  expected: string,
): Promise<void> {
  const [physicalReported, physicalExpected] = await Promise.all([
    reported ? realpath(reported).catch(() => null) : null,
    realpath(expected).catch(() => null),
  ]);
  if (!physicalReported || !physicalExpected || physicalReported !== physicalExpected) {
    throw new Error(`Model prefetch returned an unexpected ${label}: ${String(reported)}.`);
  }
}

export async function verifyModelPrefetchCacheReport(args: {
  modelId: string;
  revision: string;
  modelCache: string;
  output: {
    model_cache?: string;
    hf_home?: string;
    hub_cache?: string;
    snapshot_path?: string;
  };
}): Promise<void> {
  if (!args.output.snapshot_path) {
    throw new Error("Model prefetch did not return a snapshot path.");
  }
  const effectiveHubCache = resolveHuggingFaceCacheLayout(args.modelCache).hubCache;
  await Promise.all([
    requireReportedCachePath("model cache", args.output.model_cache, args.modelCache),
    requireReportedCachePath("HF home", args.output.hf_home, args.modelCache),
    requireReportedCachePath("hub cache", args.output.hub_cache, effectiveHubCache),
  ]);
  await verifySnapshotCacheLocation(
    args.output.snapshot_path,
    args.modelId,
    args.revision,
    args.modelCache,
  );
}

async function verifyCertifiedLocalSnapshot(
  path: string,
  modelId: string,
  snapshotFiles: ReadonlyArray<string>,
  modelCache?: string,
): Promise<void> {
  const revision = defaultBaseModelRevision(modelId);
  if (!revision) return;
  const digests = certifiedFileDigests(modelId);
  // Revision pinning and per-file digest manifests are independent. Qwen is
  // revision-pinned; Nemotron and Muse also ship a certified file list.
  if (!digests) return;
  if (!modelCache) {
    throw new Error(`A certified local snapshot for ${modelId} requires paths.modelCache.`);
  }
  const { physicalBlobRoot } = await verifySnapshotCacheLocation(
    path,
    modelId,
    revision,
    modelCache,
  );

  const physicalFiles = new Map<string, string>();
  for (const file of snapshotFiles) {
    const metadata = await lstat(file);
    const target = await realpath(file);
    if (metadata.isSymbolicLink()) {
      if (!isStrictPhysicalDescendant(physicalBlobRoot, target)) {
        throw new Error(`Certified snapshot symbolic link escapes its Hugging Face blob store: ${file}`);
      }
    }
    physicalFiles.set(file, target);
  }
  const actualFiles = new Set(snapshotFiles.map((file) => relative(path, file).split("\\").join("/")));
  const expectedFiles = new Set(Object.keys(digests));
  const unexpected = [...actualFiles].filter((name) => !expectedFiles.has(name)).sort();
  if (unexpected.length > 0) {
    throw new Error(`Certified snapshot contains unexpected file: ${join(path, unexpected[0]!)}`);
  }
  for (const [name, expected] of Object.entries(digests)) {
    if (!actualFiles.has(name)) {
      throw new Error(`Certified snapshot file is missing: ${join(path, name)}`);
    }
    const file = join(path, name);
    const physicalFile = physicalFiles.get(file)!;
    if (await certifiedFileDigest(physicalFile, expected) !== expected.digest) {
      throw new Error(`Certified snapshot file checksum mismatch: ${file}`);
    }
    if (await realpath(file) !== physicalFile) {
      throw new Error(`Certified snapshot file changed during verification: ${file}`);
    }
  }
}

export function buildModelPrefetchPayload(
  request: FineTuneRunRequest,
  config: LocalRunnerConfig,
): ModelPrefetchPayload {
  const model = resolveTrainingModel(request.spec_snapshot.base_model);
  const revision = resolveRequestedBaseModelRevision(
    model.id,
    request.hyperparameters.base_model_revision,
  );
  return {
    base_model: model.id,
    ...(revision ? { revision } : {}),
    ...(config.paths.modelCache ? { model_cache: resolve(config.paths.modelCache) } : {}),
  };
}

function safeName(value: string): string {
  return value.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "") || "model";
}

function transformersWeightName(name: string): boolean {
  const lower = name.split(/[\\/]/).at(-1)?.toLowerCase() ?? "";
  return (lower.endsWith(".safetensors") && !lower.startsWith("adapter_"))
    || /^pytorch_model.*\.bin$/.test(lower);
}

export async function verifyLocalBaseModel(
  path: string,
  expectedModelId?: string,
  modelCache?: string,
): Promise<{ fileCount: number; sizeBytes: number }> {
  const root = await lstat(path).catch(() => null);
  if (!root) throw new Error(`paths.baseModel is set to ${path}, but that path does not exist.`);
  if (root.isSymbolicLink()) {
    throw new Error(`paths.baseModel must not itself be a symbolic link: ${path}`);
  }
  if (!root.isDirectory()) {
    throw new Error(
      `paths.baseModel must be a Hugging Face snapshot directory, not a standalone file or archive: ${path}`,
    );
  }
  const files: Array<{ path: string; size: number }> = [];
  const visit = async (directory: string): Promise<void> => {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const child = join(directory, entry.name);
      if (entry.isSymbolicLink()) {
        const metadata = await stat(child);
        if (!metadata.isFile()) {
          throw new Error(`Local base model contains a non-file symbolic link: ${child}`);
        }
        files.push({ path: child, size: metadata.size });
        continue;
      }
      if (entry.isDirectory()) {
        await visit(child);
        continue;
      }
      const metadata = await stat(child);
      if (metadata.isFile()) files.push({ path: child, size: metadata.size });
    }
  };
  await visit(path);
  const nonEmpty = files.filter((file) => file.size > 0);
  const weights = nonEmpty.filter((file) => transformersWeightName(file.path));
  if (weights.length === 0) {
    throw new Error(`Local base-model directory contains no non-empty Transformers model weights: ${path}`);
  }
  const required = async (name: string) => {
    const metadata = await stat(join(path, name)).catch(() => null);
    return Boolean(metadata?.isFile() && metadata.size > 0);
  };
  if (!await required("config.json")) throw new Error(`Local base-model directory is missing config.json: ${path}`);
  try {
    const config = JSON.parse(await readFile(join(path, "config.json"), "utf8")) as unknown;
    assertCertifiedBaseModelConfig(
      config,
      `Local base-model config ${join(path, "config.json")}`,
      expectedModelId,
    );
  } catch (error) {
    const detail = error instanceof Error ? `: ${error.message}` : "";
    throw new Error(
      `Local base-model directory has an invalid or unsupported config.json: ${path}${detail}`,
      { cause: error },
    );
  }
  if (expectedModelId) {
    await verifyCertifiedLocalSnapshot(
      path,
      expectedModelId,
      files.map((file) => file.path),
      modelCache,
    );
  }
  const vocabNames = [
    "tokenizer.json", "tokenizer.model", "sentencepiece.bpe.model", "spiece.model", "vocab.json", "tokenizer.tiktoken",
  ];
  if (!await required("tokenizer_config.json") || !(await Promise.all(vocabNames.map(required))).some(Boolean)) {
    throw new Error(`Local base-model directory is missing tokenizer metadata or vocabulary: ${path}`);
  }
  for (const file of files.filter((candidate) => candidate.path.endsWith(".index.json"))) {
    const parsed = JSON.parse(await readFile(file.path, "utf8")) as { weight_map?: unknown };
    if (!parsed.weight_map || typeof parsed.weight_map !== "object" || Array.isArray(parsed.weight_map)) {
      throw new Error(`Local base-model weight index is invalid: ${file.path}`);
    }
    for (const shard of new Set(Object.values(parsed.weight_map as Record<string, unknown>))) {
      if (typeof shard !== "string" || !await required(shard)) {
        throw new Error(`Local base-model directory is missing indexed weight shard: ${String(shard)}`);
      }
    }
  }
  return {
    fileCount: files.length,
    sizeBytes: files.reduce((total, file) => total + file.size, 0),
  };
}

export async function prefetchBaseModel(args: {
  request: FineTuneRunRequest;
  config: LocalRunnerConfig;
  reporter?: LocalRunReporter;
  localOnly?: boolean;
}): Promise<ModelPrefetchReport> {
  const payload: ModelPrefetchPayload = {
    ...buildModelPrefetchPayload(args.request, args.config),
    ...(args.localOnly ? { local_files_only: true } : {}),
  };
  const artifactDir = resolve(
    args.config.artifactRoot,
    args.localOnly ? "verify-base" : "prefetch",
    `${safeName(args.request.spec_snapshot.base_model)}-${new Date().toISOString().replace(/[:.]/g, "-")}`,
  );
  await mkdir(artifactDir, { recursive: true });

  if (args.config.paths.baseModel) {
    const localPath = resolve(args.config.paths.baseModel);
    const verified = await verifyLocalBaseModel(
      localPath,
      args.request.spec_snapshot.base_model,
      args.config.paths.modelCache,
    );
    return {
      ok: true,
      status: args.localOnly ? "completed" : "skipped",
      base_model: args.request.spec_snapshot.base_model,
      local_base_model_path: localPath,
      file_count: verified.fileCount,
      size_bytes: verified.sizeBytes,
      artifact_dir: artifactDir,
      reason: args.localOnly
        ? "Verified the configured local base-model artifact; no network access was used."
        : "paths.baseModel points at a verified local base-model artifact; no Hugging Face download is needed.",
    };
  }

  const inputPath = join(artifactDir, "prefetch-input.json");
  const outputPath = join(artifactDir, "prefetch-output.json");
  const logPath = join(artifactDir, "prefetch.log");
  await writeJson(inputPath, payload);

  const entrypoint = buildBundledPythonCommand(
    "prefetch.py",
    ["--input", inputPath, "--output", outputPath],
  );

  await args.reporter?.onEvent?.({
    stage: "model_prefetch",
    status: "running",
    message: args.localOnly
      ? "Verifying the Hugging Face base model is available in the local cache."
      : "Prefetching Hugging Face base model.",
    details: {
      base_model: payload.base_model,
      model_cache: payload.model_cache ?? null,
      command: entrypoint.displayCommand,
      log_path: logPath,
    },
  });

  const cacheEnvironment = withHuggingFaceCacheEnvironment(
    minimalMachineLearningEnvironment(process.env),
    payload.model_cache,
  );
  const effectiveModelCache = cacheEnvironment.HF_HOME!;
  const effectiveHubCache = resolveHuggingFaceCacheLayout(effectiveModelCache).hubCache;
  const env = withBundledPythonEnvironment(cacheEnvironment);

  const { exitCode } = await runLoggedProcess({
    command: entrypoint.command,
    commandArgs: entrypoint.commandArgs,
    env,
    logPath,
    // A base model can be several gigabytes. Surface the downloader's progress
    // by default; callers can still opt out by omitting the reporter (`--quiet`).
    reporter: args.reporter ? { ...args.reporter, verbose: true } : undefined,
    stage: "model_prefetch",
  });

  if (exitCode !== 0) {
    throw new Error(`Model prefetch exited with code ${exitCode}. See ${logPath}.`);
  }

  const output = JSON.parse(await readFile(outputPath, "utf8")) as {
    ok?: boolean;
    base_model?: string;
    model_cache?: string;
    hf_home?: string;
    hub_cache?: string;
    snapshot_path?: string;
    snapshot_revision?: string;
    file_count?: number;
    size_bytes?: number;
    verified_blob_count?: number;
  };
  const snapshotRevision = baseModelRevisionSchema.safeParse(output.snapshot_revision);
  if (!snapshotRevision.success) {
    throw new Error(
      "Model prefetch did not resolve the base model to a 40-character immutable commit SHA.",
    );
  }
  if (output.base_model !== payload.base_model) {
    throw new Error(`Model prefetch returned ${String(output.base_model)} for requested model ${payload.base_model}.`);
  }
  if (payload.revision && snapshotRevision.data !== payload.revision.toLowerCase()) {
    throw new Error(
      `Model prefetch resolved a different revision (${snapshotRevision.data}) than requested (${payload.revision}).`,
    );
  }
  await verifyModelPrefetchCacheReport({
    modelId: payload.base_model,
    revision: snapshotRevision.data,
    modelCache: effectiveModelCache,
    output,
  });
  if (defaultBaseModelRevision(payload.base_model)) {
    await verifyLocalBaseModel(
      output.snapshot_path!,
      payload.base_model,
      effectiveModelCache,
    );
  }

  await args.reporter?.onEvent?.({
    stage: "model_prefetch",
    status: "completed",
    message: "Base model is available in the local Hugging Face cache.",
    details: {
      base_model: output.base_model ?? payload.base_model,
      hf_home: effectiveModelCache,
      hub_cache: effectiveHubCache,
      snapshot_path: output.snapshot_path ?? null,
      snapshot_revision: snapshotRevision.data,
      file_count: output.file_count ?? null,
      size_bytes: output.size_bytes ?? null,
      verified_blob_count: output.verified_blob_count ?? null,
    },
  });

  return {
    ok: true,
    status: "completed",
    base_model: output.base_model ?? payload.base_model,
    model_cache: effectiveModelCache,
    hf_home: effectiveModelCache,
    hub_cache: effectiveHubCache,
    snapshot_path: output.snapshot_path,
    snapshot_revision: snapshotRevision.data,
    file_count: output.file_count,
    size_bytes: output.size_bytes,
    verified_blob_count: output.verified_blob_count,
    artifact_dir: artifactDir,
    input_path: inputPath,
    output_path: outputPath,
    log_uri: fileUri(logPath),
    command: entrypoint.displayCommand,
  };
}
