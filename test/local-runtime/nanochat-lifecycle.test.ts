import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { chmod, mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import {
  defaultNanochatLifecycleConfig,
  NANOCHAT_LIFECYCLE_STAGES,
  nanochatLifecycleConfigSchema,
  runNanochatLifecycle,
  type NanochatSourceIdentity,
} from "../../src/local-runtime/nanochat-lifecycle.js";

const revision = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd";

function source(checkout: string, overrides: Partial<NanochatSourceIdentity> = {}): NanochatSourceIdentity {
  return {
    checkout,
    revision,
    remote: "https://github.com/karpathy/nanochat.git",
    dirty: false,
    ...overrides,
  };
}

test("runs the bounded nanochat lifecycle and hashes every stage artifact", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-nanochat-lifecycle-"));
  try {
    const checkout = join(root, "checkout");
    const python = join(root, "python");
    const artifactRoot = join(root, "artifacts");
    await mkdir(checkout);
    await writeFile(python, "fake python\n");
    await chmod(python, 0o700);
    const defaults = defaultNanochatLifecycleConfig({ checkout, revision, python, artifactRoot });
    assert.equal(defaults.rl.enabled, false);
    const config = nanochatLifecycleConfigSchema.parse({
      ...defaults,
      rl: { ...defaults.rl, enabled: true },
    });
    const commands: Array<{ stage: string; args: string[] }> = [];
    const audit = await runNanochatLifecycle(config, {
      runId: "11111111-1111-4111-8111-111111111111",
      runtime: {
        inspectSource: async () => source(checkout),
        runProcess: async (args) => {
          commands.push({ stage: args.stage, args: args.commandArgs });
          const baseDir = args.env?.NANOCHAT_BASE_DIR!;
          const stage = args.stage.replace("nanochat:", "");
          await mkdir(join(baseDir, "base_data_climbmix"), { recursive: true });
          if (stage === "data") {
            await writeFile(join(baseDir, "base_data_climbmix", "shard_00000.parquet"), "train shard");
            await writeFile(join(baseDir, "base_data_climbmix", "shard_06542.parquet"), "validation shard");
          }
          if (stage === "tokenizer") {
            await mkdir(join(baseDir, "tokenizer"), { recursive: true });
            await writeFile(join(baseDir, "tokenizer", "tokenizer.pkl"), "tokenizer");
          }
          if (stage === "pretrain") {
            await mkdir(join(baseDir, "base_checkpoints", config.pretrain.modelTag), { recursive: true });
            await writeFile(join(baseDir, "base_checkpoints", config.pretrain.modelTag, "model_000002.pt"), "base");
          }
          if (stage === "sft") {
            await mkdir(join(baseDir, "chatsft_checkpoints", config.pretrain.modelTag), { recursive: true });
            await writeFile(join(baseDir, "chatsft_checkpoints", config.pretrain.modelTag, "model_000001.pt"), "sft");
            await mkdir(join(baseDir, "task_data", "HuggingFaceTB--smol-smoltalk", "default", "train"), { recursive: true });
            await writeFile(
              join(baseDir, "task_data", "HuggingFaceTB--smol-smoltalk", "default", "train", "00000.parquet"),
              "sft dataset",
            );
          }
          if (stage === "chat_eval") {
            await mkdir(join(baseDir, "task_data", "allenai--ai2_arc", "ARC-Easy", "test"), { recursive: true });
            await writeFile(
              join(baseDir, "task_data", "allenai--ai2_arc", "ARC-Easy", "test", "00000.parquet"),
              "evaluation dataset",
            );
          }
          if (stage === "rl") {
            await mkdir(join(baseDir, "chatrl_checkpoints", config.pretrain.modelTag), { recursive: true });
            await writeFile(join(baseDir, "chatrl_checkpoints", config.pretrain.modelTag, "model_000001.pt"), "rl");
            await mkdir(join(baseDir, "task_data", "openai--gsm8k", "main", "train"), { recursive: true });
            await writeFile(
              join(baseDir, "task_data", "openai--gsm8k", "main", "train", "00000.parquet"),
              "rl dataset",
            );
          }
          if (args.logPath) await writeFile(args.logPath, `${stage} completed\n`);
          return { exitCode: 0, stderr: "" };
        },
      },
    });

    assert.equal(audit.status, "completed");
    assert.deepEqual(audit.dataset.task_repositories, [
      "HuggingFaceTB/smol-smoltalk",
      "cais/mmlu",
      "openai/gsm8k",
      "allenai/ai2_arc",
    ]);
    assert.deepEqual(audit.stages.map((stage) => stage.id), [...NANOCHAT_LIFECYCLE_STAGES]);
    assert.equal(audit.stages.find((stage) => stage.id === "rl")?.status, "completed");
    assert.ok(audit.stages.find((stage) => stage.id === "data")?.outputs.some((file) => file.path.endsWith("shard_00000.parquet")));
    assert.ok(audit.stages.find((stage) => stage.id === "sft")?.outputs.some((file) => file.path.includes("smol-smoltalk")));
    assert.ok(audit.stages.find((stage) => stage.id === "chat_eval")?.outputs.some((file) => file.path.includes("ai2_arc")));
    assert.ok(audit.stages.find((stage) => stage.id === "rl")?.outputs.some((file) => file.path.includes("gsm8k/main/train")));
    assert.ok(audit.stages.find((stage) => stage.id === "sft")?.outputs.every((file) => file.sha256.length === 64));
    assert.deepEqual(commands.map((command) => command.stage), [
      "nanochat:data",
      "nanochat:tokenizer",
      "nanochat:tokenizer_eval",
      "nanochat:pretrain",
      "nanochat:base_eval",
      "nanochat:sft",
      "nanochat:chat_eval",
      "nanochat:rl",
      "nanochat:inference",
    ]);
    assert.deepEqual(commands.find((command) => command.stage === "nanochat:inference")?.args.slice(2, 6), [
      "-i", "rl", "-g", config.pretrain.modelTag,
    ]);
    const stored = JSON.parse(await readFile(join(audit.artifact_root, "lifecycle.json"), "utf8"));
    assert.equal(stored.status, "completed");
    assert.equal(stored.source.revision, revision);
    const storedConfig = await readFile(stored.config_path);
    assert.deepEqual(JSON.parse(storedConfig.toString("utf8")), config);
    assert.equal(stored.config_sha256, createHash("sha256").update(storedConfig).digest("hex"));
    const packaged = JSON.parse(await readFile(join(audit.artifact_root, "package.json"), "utf8"));
    assert.equal(packaged.source, "rl");
    assert.equal(packaged.source_revision, revision);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("fails revision validation before creating run artifacts", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-nanochat-revision-"));
  try {
    const checkout = join(root, "checkout");
    const python = join(root, "python");
    const artifactRoot = join(root, "artifacts");
    await mkdir(checkout);
    await writeFile(python, "fake python\n");
    await chmod(python, 0o700);
    const config = defaultNanochatLifecycleConfig({ checkout, revision, python, artifactRoot });
    await assert.rejects(
      runNanochatLifecycle(config, {
        runtime: { inspectSource: async () => source(checkout, { revision: "a".repeat(40) }) },
      }),
      /revision mismatch/i,
    );
    await assert.rejects(readFile(join(artifactRoot, "lifecycle.json")));
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("rejects invalid batch geometry before source inspection", async () => {
  const base = defaultNanochatLifecycleConfig({ checkout: "/tmp/nanochat", revision, python: "/tmp/python" });
  const raw = {
    ...base,
    pretrain: {
      ...base.pretrain,
      totalBatchSize: 127,
    },
  };
  await assert.rejects(runNanochatLifecycle(raw), /totalBatchSize|divisible/i);
});

test("rejects a non-UUID run identity before creating artifacts", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-nanochat-run-id-"));
  try {
    const checkout = join(root, "checkout");
    const python = join(root, "python");
    const artifactRoot = join(root, "artifacts");
    await mkdir(checkout);
    await writeFile(python, "fake python\n");
    await chmod(python, 0o700);
    const config = defaultNanochatLifecycleConfig({ checkout, revision, python, artifactRoot });
    await assert.rejects(
      runNanochatLifecycle(config, {
        runId: "../../escape",
        runtime: { inspectSource: async () => source(checkout) },
      }),
      /runId must be a UUID/i,
    );
    await assert.rejects(readFile(join(root, "escape", "lifecycle.json")));
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
