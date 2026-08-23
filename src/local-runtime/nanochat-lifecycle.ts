import { createHash, randomUUID } from "node:crypto";
import { execFile } from "node:child_process";
import { constants, createReadStream } from "node:fs";
import { promisify } from "node:util";
import {
  access,
  lstat,
  mkdir,
  readFile,
  readdir,
  realpath,
  stat,
} from "node:fs/promises";
import { dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { z } from "zod";
import { runLoggedProcess } from "./process-runner.js";
import { writeFileAtomic, writeJsonAtomic } from "./artifacts.js";

const execFileAsync = promisify(execFile);

export const NANOCHAT_LIFECYCLE_STAGES = [
  "data",
  "tokenizer",
  "tokenizer_eval",
  "pretrain",
  "base_eval",
  "sft",
  "chat_eval",
  "rl",
  "inference",
  "package",
] as const;

export type NanochatLifecycleStageId = (typeof NANOCHAT_LIFECYCLE_STAGES)[number];

const sourceSchema = z.object({
  checkout: z.string().min(1),
  revision: z.string().regex(/^[a-f0-9]{40}$/i, "revision must be a full 40-character Git commit"),
  python: z.string().min(1),
  allowDirty: z.boolean().default(false),
}).strict();

const dataSchema = z.object({
  trainShards: z.number().int().min(1).max(6542).default(1),
  workers: z.number().int().min(1).max(32).default(2),
}).strict().default({ trainShards: 1, workers: 2 });

const tokenizerSchema = z.object({
  maxChars: z.number().int().min(10_000).max(2_000_000_000).default(2_000_000),
  docCap: z.number().int().min(128).max(100_000).default(10_000),
  vocabSize: z.number().int().min(265).max(131_072).default(512),
}).strict().default({ maxChars: 2_000_000, docCap: 10_000, vocabSize: 512 });

const pretrainSchema = z.object({
  depth: z.number().int().min(1).max(64).default(2),
  headDim: z.number().int().min(16).max(256).default(64),
  maxSeqLen: z.number().int().min(32).max(8192).default(64),
  windowPattern: z.string().regex(/^[SL]+$/).default("L"),
  deviceBatchSize: z.number().int().min(1).max(1024).default(2),
  totalBatchSize: z.number().int().min(32).default(128),
  numIterations: z.number().int().min(1).max(10_000_000).default(2),
  modelTag: z.string().regex(/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$/).default("tt-smoke"),
}).strict().default({
  depth: 2,
  headDim: 64,
  maxSeqLen: 64,
  windowPattern: "L",
  deviceBatchSize: 2,
  totalBatchSize: 128,
  numIterations: 2,
  modelTag: "tt-smoke",
});

const baseEvalSchema = z.object({
  enabled: z.boolean().default(true),
  splitTokens: z.number().int().min(32).max(100_000_000).default(512),
  deviceBatchSize: z.number().int().min(1).max(1024).default(1),
}).strict().default({ enabled: true, splitTokens: 512, deviceBatchSize: 1 });

const sftSchema = z.object({
  enabled: z.boolean().default(true),
  numIterations: z.number().int().min(1).max(10_000_000).default(1),
  deviceBatchSize: z.number().int().min(1).max(1024).default(1),
  totalBatchSize: z.number().int().min(32).default(64),
  mmluEpochs: z.number().int().min(0).max(100).default(0),
  gsm8kEpochs: z.number().int().min(0).max(100).default(0),
}).strict().default({
  enabled: true,
  numIterations: 1,
  deviceBatchSize: 1,
  totalBatchSize: 64,
  mmluEpochs: 0,
  gsm8kEpochs: 0,
});

const chatEvalSchema = z.object({
  enabled: z.boolean().default(true),
  task: z.enum(["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval"]).default("ARC-Easy"),
  maxProblems: z.number().int().min(1).max(100_000).default(1),
  maxNewTokens: z.number().int().min(1).max(4096).default(8),
  batchSize: z.number().int().min(1).max(1024).default(1),
}).strict().default({
  enabled: true,
  task: "ARC-Easy",
  maxProblems: 1,
  maxNewTokens: 8,
  batchSize: 1,
});

const rlSchema = z.object({
  enabled: z.boolean().default(false),
  numEpochs: z.number().int().min(1).max(100).default(1),
  deviceBatchSize: z.number().int().min(1).max(1024).default(1),
  examplesPerStep: z.number().int().min(1).max(100_000).default(1),
  numSamples: z.number().int().min(1).max(1024).default(1),
  maxNewTokens: z.number().int().min(1).max(4096).default(32),
}).strict().default({
  enabled: false,
  numEpochs: 1,
  deviceBatchSize: 1,
  examplesPerStep: 1,
  numSamples: 1,
  maxNewTokens: 32,
});

const inferenceSchema = z.object({
  enabled: z.boolean().default(true),
  promptTokens: z.number().int().min(16).max(8192).default(16),
  decodeTokens: z.number().int().min(2).max(4096).default(4),
  batchSizes: z.array(z.number().int().min(1).max(1024)).min(1).max(16).default([1]),
  temperature: z.number().min(0).max(2).default(0),
}).strict().default({
  enabled: true,
  promptTokens: 16,
  decodeTokens: 4,
  batchSizes: [1],
  temperature: 0,
});

export const nanochatLifecycleConfigSchema = z.object({
  version: z.literal(1),
  name: z.string().min(1).max(120).default("nanochat-lifecycle"),
  artifactRoot: z.string().min(1).default(".tt-local/nanochat"),
  source: sourceSchema,
  data: dataSchema,
  tokenizer: tokenizerSchema,
  pretrain: pretrainSchema,
  baseEval: baseEvalSchema,
  sft: sftSchema,
  chatEval: chatEvalSchema,
  rl: rlSchema,
  inference: inferenceSchema,
  timeoutMs: z.number().int().min(1_000).max(30 * 24 * 60 * 60 * 1000).default(24 * 60 * 60 * 1000),
}).strict().superRefine((config, context) => {
  const tokensPerMicroBatch = config.pretrain.deviceBatchSize * config.pretrain.maxSeqLen;
  if (config.pretrain.totalBatchSize % tokensPerMicroBatch !== 0) {
    context.addIssue({
      code: "custom",
      path: ["pretrain", "totalBatchSize"],
      message: `must be divisible by deviceBatchSize * maxSeqLen (${tokensPerMicroBatch})`,
    });
  }
  if (config.sft.enabled && config.sft.totalBatchSize % (config.sft.deviceBatchSize * config.pretrain.maxSeqLen) !== 0) {
    context.addIssue({
      code: "custom",
      path: ["sft", "totalBatchSize"],
      message: `must be divisible by deviceBatchSize * pretrain.maxSeqLen (${config.sft.deviceBatchSize * config.pretrain.maxSeqLen})`,
    });
  }
  if (config.rl.enabled && !config.sft.enabled) {
    context.addIssue({ code: "custom", path: ["rl", "enabled"], message: "RL requires the SFT stage" });
  }
  if (config.chatEval.enabled && !config.sft.enabled) {
    context.addIssue({ code: "custom", path: ["chatEval", "enabled"], message: "chat evaluation requires the SFT stage" });
  }
  if (config.inference.promptTokens + config.inference.decodeTokens > config.pretrain.maxSeqLen) {
    context.addIssue({
      code: "custom",
      path: ["inference", "decodeTokens"],
      message: "promptTokens + decodeTokens must not exceed pretrain.maxSeqLen",
    });
  }
}).transform((config) => ({
  ...config,
  source: { ...config.source, revision: config.source.revision.toLowerCase() },
}));

export type NanochatLifecycleConfig = z.infer<typeof nanochatLifecycleConfigSchema>;

export interface NanochatSourceIdentity {
  checkout: string;
  revision: string;
  remote: string | null;
  dirty: boolean;
}

export interface NanochatArtifactFile {
  path: string;
  size_bytes: number;
  sha256: string;
}

export interface NanochatStageAudit {
  id: NanochatLifecycleStageId;
  status: "pending" | "running" | "completed" | "skipped" | "failed";
  depends_on: NanochatLifecycleStageId[];
  started_at?: string;
  completed_at?: string;
  elapsed_ms?: number;
  command?: string[];
  log_path?: string;
  reason?: string;
  exit_code?: number;
  outputs: NanochatArtifactFile[];
}

export interface NanochatLifecycleAudit {
  schema_version: 1;
  workflow: "nanochat";
  run_id: string;
  name: string;
  status: "running" | "completed" | "failed";
  created_at: string;
  updated_at: string;
  completed_at?: string;
  config_sha256: string;
  config_path: string;
  source: NanochatSourceIdentity;
  dataset: {
    provider: "huggingface";
    repository: "karpathy/climbmix-400b-shuffle";
    train_shards: number;
    immutable_by: "downloaded_file_sha256";
    task_repositories: string[];
  };
  artifact_root: string;
  nanochat_base_dir: string;
  stages: NanochatStageAudit[];
  error?: string;
}

interface LifecycleRuntime {
  inspectSource?: (checkout: string) => Promise<NanochatSourceIdentity>;
  runProcess?: typeof runLoggedProcess;
  now?: () => Date;
}

export interface RunNanochatLifecycleOptions {
  runId?: string;
  runtime?: LifecycleRuntime;
}

function resolveLocalPath(value: string, base: string): string {
  if (isAbsolute(value)) return resolve(value);
  return resolve(base, value);
}

export async function loadNanochatLifecycleConfig(path: string): Promise<NanochatLifecycleConfig> {
  const absolute = resolve(path);
  const parsed = nanochatLifecycleConfigSchema.parse(JSON.parse(await readFile(absolute, "utf8")));
  const base = dirname(absolute);
  return {
    ...parsed,
    artifactRoot: resolveLocalPath(parsed.artifactRoot, base),
    source: {
      ...parsed.source,
      checkout: resolveLocalPath(parsed.source.checkout, base),
      python: resolveLocalPath(parsed.source.python, base),
    },
  };
}

export async function inspectNanochatSource(checkout: string): Promise<NanochatSourceIdentity> {
  const canonical = await realpath(resolve(checkout));
  for (const required of [
    "nanochat/dataset.py",
    "scripts/tok_train.py",
    "scripts/tok_eval.py",
    "scripts/base_train.py",
    "scripts/base_eval.py",
    "scripts/chat_sft.py",
    "scripts/chat_eval.py",
    "scripts/chat_rl.py",
    "scripts/infer_bench.py",
  ]) {
    await access(join(canonical, required)).catch(() => {
      throw new Error(`nanochat checkout is missing ${required}: ${canonical}`);
    });
  }
  const git = async (args: string[]) => (await execFileAsync("git", ["-C", canonical, ...args], {
    encoding: "utf8",
    maxBuffer: 1024 * 1024,
  })).stdout.trim();
  const [revision, status, remote] = await Promise.all([
    git(["rev-parse", "HEAD"]),
    git(["status", "--porcelain", "--untracked-files=no"]),
    git(["remote", "get-url", "origin"]).catch(() => ""),
  ]);
  return {
    checkout: canonical,
    revision: revision.toLowerCase(),
    remote: remote || null,
    dirty: Boolean(status),
  };
}

async function sha256File(path: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}

async function listFiles(current: string): Promise<string[]> {
  const files: string[] = [];
  for (const entry of await readdir(current, { withFileTypes: true })) {
    const absolute = join(current, entry.name);
    const metadata = await lstat(absolute);
    if (metadata.isSymbolicLink()) throw new Error(`nanochat artifacts must not contain symlinks: ${absolute}`);
    if (metadata.isDirectory()) files.push(...await listFiles(absolute));
    else if (metadata.isFile()) files.push(absolute);
  }
  return files.sort();
}

async function describeOutputs(runDir: string, paths: string[]): Promise<NanochatArtifactFile[]> {
  const files = new Set<string>();
  for (const path of paths) {
    const metadata = await lstat(path).catch(() => null);
    if (!metadata) throw new Error(`nanochat stage did not create expected output: ${path}`);
    if (metadata.isSymbolicLink()) throw new Error(`nanochat artifacts must not contain symlinks: ${path}`);
    if (metadata.isDirectory()) {
      const nestedFiles = await listFiles(path);
      if (nestedFiles.length === 0) throw new Error(`nanochat stage created an empty expected output: ${path}`);
      for (const file of nestedFiles) files.add(file);
    } else if (metadata.isFile()) {
      files.add(path);
    }
  }
  return await Promise.all([...files].sort().map(async (path) => {
    const containment = relative(runDir, path);
    if (containment === ".." || containment.startsWith(`..${sep}`) || isAbsolute(containment)) {
      throw new Error(`nanochat output escapes its run directory: ${path}`);
    }
    return {
      path: containment.split(sep).join("/"),
      size_bytes: (await stat(path)).size,
      sha256: await sha256File(path),
    };
  }));
}

function lifecycleEnvironment(baseDir: string): NodeJS.ProcessEnv {
  const inherited = Object.fromEntries([
    "PATH",
    "HOME",
    "USER",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "CUDA_HOME",
    "LD_LIBRARY_PATH",
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "HF_HUB_CACHE",
    "TORCH_HOME",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
  ].flatMap((name) => process.env[name] === undefined ? [] : [[name, process.env[name]!]]));
  return {
    ...inherited,
    NANOCHAT_BASE_DIR: baseDir,
    WANDB_RUN: "dummy",
    WANDB_MODE: "disabled",
    OMP_NUM_THREADS: "1",
    PYTHONUNBUFFERED: "1",
  };
}

function taskRepositories(config: NanochatLifecycleConfig): string[] {
  const repositories = new Set<string>();
  if (config.sft.enabled) {
    repositories.add("HuggingFaceTB/smol-smoltalk");
    repositories.add("cais/mmlu");
    repositories.add("openai/gsm8k");
  }
  if (config.chatEval.enabled) {
    if (config.chatEval.task.startsWith("ARC-")) repositories.add("allenai/ai2_arc");
    if (config.chatEval.task === "MMLU") repositories.add("cais/mmlu");
    if (config.chatEval.task === "GSM8K") repositories.add("openai/gsm8k");
    if (config.chatEval.task === "HumanEval") repositories.add("openai/openai_humaneval");
  }
  return [...repositories];
}

interface StageDefinition {
  id: NanochatLifecycleStageId;
  dependsOn: NanochatLifecycleStageId[];
  enabled: boolean;
  args?: string[];
  outputPaths: (baseDir: string, runDir: string, modelTag: string) => string[];
}

function stageDefinitions(config: NanochatLifecycleConfig): StageDefinition[] {
  const tag = config.pretrain.modelTag;
  const finalModelStage: NanochatLifecycleStageId = config.rl.enabled ? "rl" : config.sft.enabled ? "sft" : "pretrain";
  const finalModelSource = config.rl.enabled ? "rl" : config.sft.enabled ? "sft" : "base";
  const inferenceDependency: NanochatLifecycleStageId = config.rl.enabled
    ? "rl"
    : config.chatEval.enabled
      ? "chat_eval"
      : config.sft.enabled
        ? "sft"
        : config.baseEval.enabled
          ? "base_eval"
          : "pretrain";
  return [
    {
      id: "data",
      dependsOn: [],
      enabled: true,
      args: ["-m", "nanochat.dataset", "-n", String(config.data.trainShards), "-w", String(config.data.workers)],
      outputPaths: (baseDir) => [join(baseDir, "base_data_climbmix")],
    },
    {
      id: "tokenizer",
      dependsOn: ["data"],
      enabled: true,
      args: [
        "-m", "scripts.tok_train",
        `--max-chars=${config.tokenizer.maxChars}`,
        `--doc-cap=${config.tokenizer.docCap}`,
        `--vocab-size=${config.tokenizer.vocabSize}`,
      ],
      outputPaths: (baseDir) => [join(baseDir, "tokenizer")],
    },
    {
      id: "tokenizer_eval",
      dependsOn: ["tokenizer"],
      enabled: true,
      args: ["-m", "scripts.tok_eval"],
      outputPaths: (_baseDir, runDir) => [join(runDir, "logs", "tokenizer_eval.log")],
    },
    {
      id: "pretrain",
      dependsOn: ["tokenizer_eval"],
      enabled: true,
      args: [
        "-m", "scripts.base_train",
        `--depth=${config.pretrain.depth}`,
        `--head-dim=${config.pretrain.headDim}`,
        `--window-pattern=${config.pretrain.windowPattern}`,
        `--max-seq-len=${config.pretrain.maxSeqLen}`,
        `--device-batch-size=${config.pretrain.deviceBatchSize}`,
        `--total-batch-size=${config.pretrain.totalBatchSize}`,
        `--num-iterations=${config.pretrain.numIterations}`,
        `--model-tag=${tag}`,
        "--eval-every=-1",
        "--core-metric-every=-1",
        "--sample-every=-1",
        "--save-every=-1",
        "--run=dummy",
      ],
      outputPaths: (baseDir) => [join(baseDir, "base_checkpoints", tag)],
    },
    {
      id: "base_eval",
      dependsOn: ["pretrain"],
      enabled: config.baseEval.enabled,
      args: [
        "-m", "scripts.base_eval",
        "--eval=bpb,sample",
        `--model-tag=${tag}`,
        `--device-batch-size=${config.baseEval.deviceBatchSize}`,
        `--split-tokens=${config.baseEval.splitTokens}`,
      ],
      outputPaths: (_baseDir, runDir) => [join(runDir, "logs", "base_eval.log")],
    },
    {
      id: "sft",
      dependsOn: config.baseEval.enabled ? ["base_eval"] : ["pretrain"],
      enabled: config.sft.enabled,
      args: [
        "-m", "scripts.chat_sft",
        `--model-tag=${tag}`,
        "--load-optimizer=0",
        `--num-iterations=${config.sft.numIterations}`,
        `--max-seq-len=${config.pretrain.maxSeqLen}`,
        `--device-batch-size=${config.sft.deviceBatchSize}`,
        `--total-batch-size=${config.sft.totalBatchSize}`,
        `--mmlu-epochs=${config.sft.mmluEpochs}`,
        `--gsm8k-epochs=${config.sft.gsm8kEpochs}`,
        "--eval-every=-1",
        "--chatcore-every=-1",
        "--run=dummy",
      ],
      outputPaths: (baseDir) => [
        join(baseDir, "chatsft_checkpoints", tag),
        join(baseDir, "task_data"),
      ],
    },
    {
      id: "chat_eval",
      dependsOn: ["sft"],
      enabled: config.chatEval.enabled,
      args: [
        "-m", "scripts.chat_eval",
        "-i", "sft",
        "-g", tag,
        "-a", config.chatEval.task,
        "-x", String(config.chatEval.maxProblems),
        "-m", String(config.chatEval.maxNewTokens),
        "-b", String(config.chatEval.batchSize),
      ],
      outputPaths: (baseDir, runDir) => [
        join(baseDir, "task_data"),
        join(runDir, "logs", "chat_eval.log"),
      ],
    },
    {
      id: "rl",
      dependsOn: ["sft"],
      enabled: config.rl.enabled,
      args: [
        "-m", "scripts.chat_rl",
        `--model-tag=${tag}`,
        `--num-epochs=${config.rl.numEpochs}`,
        `--device-batch-size=${config.rl.deviceBatchSize}`,
        `--examples-per-step=${config.rl.examplesPerStep}`,
        `--num-samples=${config.rl.numSamples}`,
        `--max-new-tokens=${config.rl.maxNewTokens}`,
        "--eval-every=10000000",
        "--save-every=10000000",
        "--run=dummy",
      ],
      outputPaths: (baseDir) => [join(baseDir, "chatrl_checkpoints", tag)],
    },
    {
      id: "inference",
      dependsOn: [inferenceDependency],
      enabled: config.inference.enabled,
      args: [
        "-m", "scripts.infer_bench",
        "-i", finalModelSource,
        "-g", tag,
        `--prompt-tokens=${config.inference.promptTokens}`,
        `--decode-tokens=${config.inference.decodeTokens}`,
        `--batch-sizes=${config.inference.batchSizes.join(",")}`,
        `--temperature=${config.inference.temperature}`,
      ],
      outputPaths: (_baseDir, runDir) => [join(runDir, "logs", "inference.log")],
    },
    {
      id: "package",
      dependsOn: config.inference.enabled ? ["inference"] : [finalModelStage],
      enabled: true,
      outputPaths: (_baseDir, runDir) => [
        join(runDir, "config.json"),
        join(runDir, "package.json"),
      ],
    },
  ];
}

function newStageAudit(definition: StageDefinition): NanochatStageAudit {
  return {
    id: definition.id,
    status: "pending",
    depends_on: definition.dependsOn,
    outputs: [],
  };
}

async function writePackage(args: {
  path: string;
  config: NanochatLifecycleConfig;
  audit: NanochatLifecycleAudit;
  baseDir: string;
}): Promise<void> {
  const source = args.config.rl.enabled ? "rl" : args.config.sft.enabled ? "sft" : "base";
  const checkpointDirectory = join(
    args.baseDir,
    source === "rl" ? "chatrl_checkpoints" : source === "sft" ? "chatsft_checkpoints" : "base_checkpoints",
    args.config.pretrain.modelTag,
  );
  await access(checkpointDirectory).catch(() => {
    throw new Error(`Final nanochat checkpoint directory is missing: ${checkpointDirectory}`);
  });
  await writeJsonAtomic(args.path, {
    schema_version: 1,
    framework: "nanochat",
    source,
    model_tag: args.config.pretrain.modelTag,
    checkpoint_directory: checkpointDirectory,
    tokenizer_directory: join(args.baseDir, "tokenizer"),
    source_revision: args.audit.source.revision,
    lifecycle_manifest: join(args.audit.artifact_root, "lifecycle.json"),
  });
}

export async function runNanochatLifecycle(
  rawConfig: unknown,
  options: RunNanochatLifecycleOptions = {},
): Promise<NanochatLifecycleAudit> {
  const config = nanochatLifecycleConfigSchema.parse(rawConfig);
  const inspectSource = options.runtime?.inspectSource ?? inspectNanochatSource;
  const runProcess = options.runtime?.runProcess ?? runLoggedProcess;
  const now = options.runtime?.now ?? (() => new Date());
  const source = await inspectSource(config.source.checkout);
  if (source.revision !== config.source.revision) {
    throw new Error(`nanochat revision mismatch: expected ${config.source.revision}, found ${source.revision}`);
  }
  if (source.dirty && !config.source.allowDirty) {
    throw new Error("nanochat checkout has tracked modifications; commit them or set source.allowDirty explicitly");
  }
  await access(config.source.python, constants.X_OK).catch(() => {
    throw new Error(`nanochat Python executable does not exist or is not executable: ${config.source.python}`);
  });

  const runId = options.runId ?? randomUUID();
  if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(runId)) {
    throw new Error("nanochat lifecycle runId must be a UUID");
  }
  const artifactRoot = resolve(config.artifactRoot);
  const runDir = join(artifactRoot, runId);
  const baseDir = join(runDir, "nanochat");
  const manifestPath = join(runDir, "lifecycle.json");
  await mkdir(artifactRoot, { recursive: true });
  await mkdir(runDir, { recursive: false }).catch((error: NodeJS.ErrnoException) => {
    if (error.code === "EEXIST") throw new Error(`nanochat lifecycle run already exists: ${runId}`);
    throw error;
  });
  await mkdir(join(runDir, "logs"), { recursive: true });
  await mkdir(baseDir, { recursive: true });
  const configPath = join(runDir, "config.json");
  await writeJsonAtomic(configPath, config);

  const createdAt = now().toISOString();
  const definitions = stageDefinitions(config);
  const audit: NanochatLifecycleAudit = {
    schema_version: 1,
    workflow: "nanochat",
    run_id: runId,
    name: config.name,
    status: "running",
    created_at: createdAt,
    updated_at: createdAt,
    config_sha256: await sha256File(configPath),
    config_path: configPath,
    source,
    dataset: {
      provider: "huggingface",
      repository: "karpathy/climbmix-400b-shuffle",
      train_shards: config.data.trainShards,
      immutable_by: "downloaded_file_sha256",
      task_repositories: taskRepositories(config),
    },
    artifact_root: runDir,
    nanochat_base_dir: baseDir,
    stages: definitions.map(newStageAudit),
  };
  const writeAudit = async () => {
    audit.updated_at = now().toISOString();
    await writeJsonAtomic(manifestPath, audit);
  };
  await writeAudit();

  let active: NanochatStageAudit | undefined;
  try {
    for (const definition of definitions) {
      active = audit.stages.find((stage) => stage.id === definition.id)!;
      if (!definition.enabled) {
        active.status = "skipped";
        active.reason = "disabled by lifecycle configuration";
        active.completed_at = now().toISOString();
        await writeAudit();
        continue;
      }
      for (const dependency of active.depends_on) {
        const dependencyStatus = audit.stages.find((stage) => stage.id === dependency)?.status;
        if (dependencyStatus !== "completed") {
          throw new Error(`Stage ${active.id} requires completed stage ${dependency}, found ${dependencyStatus ?? "missing"}`);
        }
      }
      const started = now();
      active.status = "running";
      active.started_at = started.toISOString();
      const logPath = join(runDir, "logs", `${active.id}.log`);
      active.log_path = logPath;
      await writeAudit();

      if (definition.id === "package") {
        await writePackage({ path: join(runDir, "package.json"), config, audit, baseDir });
        await writeFileAtomic(logPath, "Packaged audited nanochat checkpoint references.\n");
        active.command = ["internal:package"];
        active.exit_code = 0;
      } else {
        const command = [config.source.python, ...definition.args!];
        active.command = command;
        await writeAudit();
        const result = await runProcess({
          command: config.source.python,
          commandArgs: definition.args!,
          cwd: source.checkout,
          env: lifecycleEnvironment(baseDir),
          logPath,
          timeoutMs: config.timeoutMs,
          timeoutMessage: `nanochat stage ${active.id} timed out after ${config.timeoutMs}ms`,
          stage: `nanochat:${active.id}`,
          terminateProcessGroupOnExit: true,
        });
        active.exit_code = result.exitCode;
        if (result.exitCode !== 0) {
          throw new Error(`nanochat stage ${active.id} exited with code ${result.exitCode}; see ${logPath}`);
        }
      }
      active.outputs = await describeOutputs(
        runDir,
        [...definition.outputPaths(baseDir, runDir, config.pretrain.modelTag), logPath],
      );
      active.status = "completed";
      active.completed_at = now().toISOString();
      active.elapsed_ms = Math.max(0, now().getTime() - started.getTime());
      await writeAudit();
    }
    audit.status = "completed";
    audit.completed_at = now().toISOString();
    await writeAudit();
    return audit;
  } catch (error) {
    if (active?.status === "running") {
      active.status = "failed";
      active.completed_at = now().toISOString();
      active.reason = error instanceof Error ? error.message : String(error);
    }
    audit.status = "failed";
    audit.error = error instanceof Error ? error.message : String(error);
    audit.completed_at = now().toISOString();
    await writeAudit();
    throw error;
  }
}

export function defaultNanochatLifecycleConfig(args: {
  checkout: string;
  revision: string;
  python: string;
  artifactRoot?: string;
}): NanochatLifecycleConfig {
  return nanochatLifecycleConfigSchema.parse({
    version: 1,
    name: "nanochat-smoke",
    artifactRoot: args.artifactRoot ?? ".tt-local/nanochat",
    source: {
      checkout: args.checkout,
      revision: args.revision,
      python: args.python,
    },
  });
}
