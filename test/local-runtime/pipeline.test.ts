import assert from "node:assert/strict";
import { mkdir, mkdtemp, readdir, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fineTuneRunRequestSchema, localRunnerConfigSchema } from "../../src/local-runtime/contracts.js";
import {
  canonicalLocalPipeline,
  runLocalPipeline,
  validateLocalPipeline,
} from "../../src/local-runtime/orchestrator.js";

const behaviorSpecId = "22222222-2222-4222-8222-222222222222";

function requestFixture(runId: string) {
  return fineTuneRunRequestSchema.parse({
    run_id: runId,
    user_id: "local-user",
    behavior_spec_id: behaviorSpecId,
    run_number: 1,
    spec_snapshot: {
      name: "Local SFT",
      description: "",
      system_prompt: "Return labels.",
      guidelines: [],
      constraints: [],
      base_model: "Qwen/Qwen3.5-2B",
      examples: [
        { input: "Classify: good", output: "positive" },
        { input: "Classify: bad", output: "negative" },
      ],
    },
    hyperparameters: { n_epochs: 1 },
  });
}

function configFixture(root: string) {
  return localRunnerConfigSchema.parse({
    artifactRoot: join(root, "artifacts"),
    storeRoot: join(root, "store"),
    dryRun: true,
  });
}

test("validates local pipeline references before any store or artifact state exists", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-pipeline-invalid-"));
  try {
    const pipeline = {
      version: 1 as const,
      steps: [{ id: "candidate", uses: "evaluate" as const, target: "local" as const, with: { model: { from: "train.model" }, evaluator: "behavior" } }],
    };
    assert.throws(() => validateLocalPipeline(pipeline), /must reference an earlier train step/);
    await assert.rejects(
      runLocalPipeline({ request: requestFixture("81818181-8181-4818-8818-818181818181"), config: configFixture(root), pipeline }),
      /must reference an earlier train step/,
    );
    await assert.rejects(readdir(join(root, "artifacts")));
    await assert.rejects(readdir(join(root, "store")));
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("rejects multiple local train steps before execution", () => {
  assert.throws(() => validateLocalPipeline({
    version: 1,
    steps: [
      { id: "train_a", uses: "train", target: "local" },
      { id: "train_b", uses: "train", target: "local" },
    ],
  }), /at most one train/i);
});

test("refuses a preserved symlinked run directory before reading its workflow lock", {
  skip: process.platform === "win32",
}, async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-pipeline-lock-symlink-"));
  try {
    const runId = "85858585-8585-4858-8858-858585858585";
    const config = configFixture(root);
    const externalRun = join(root, "external-run");
    await mkdir(join(config.storeRoot!, "runs"), { recursive: true });
    await mkdir(externalRun);
    await writeFile(join(externalRun, "workflow.lock"), JSON.stringify({
      pid: process.pid,
      created_at: new Date().toISOString(),
    }));
    await symlink(externalRun, join(config.storeRoot!, "runs", runId));

    await assert.rejects(
      runLocalPipeline({
        request: requestFixture(runId),
        config,
        pipeline: canonicalLocalPipeline(),
      }),
      /symbolic link/i,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("runs a reordered partial local pipeline and finishes without a synthetic full report", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-pipeline-partial-"));
  try {
    const result = await runLocalPipeline({
      request: requestFixture("82828282-8282-4828-8828-828282828282"),
      config: configFixture(root),
      pipeline: {
        version: 1,
        steps: [
          { id: "train", uses: "train", target: "local" },
          { id: "baseline", uses: "evaluate", target: "local", with: { model: "base", evaluator: "behavior" } },
          { id: "candidate", uses: "evaluate", target: "local", with: { model: { from: "train.model" }, evaluator: "behavior" } },
        ],
      },
    });
    assert.equal(result.status, "stage_completed");
    assert.equal(result.report, undefined);
    assert.deepEqual(Object.keys(result.outputs), ["train", "baseline", "candidate"]);
    assert.ok(result.outputs.candidate?.reportPath.endsWith("evaluations/candidate.json"));
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("allows repeated behavior evaluations with distinct step artifacts", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-pipeline-repeat-"));
  try {
    const result = await runLocalPipeline({
      request: requestFixture("84848484-8484-4848-8848-848484848484"),
      config: configFixture(root),
      pipeline: {
        version: 1,
        steps: [
          { id: "first", uses: "evaluate", target: "local", with: { model: "base", evaluator: "behavior" } },
          { id: "second", uses: "evaluate", target: "local", with: { model: "base", evaluator: "behavior" } },
        ],
      },
    });
    assert.equal(result.status, "stage_completed");
    assert.notEqual(result.outputs.first?.reportPath, result.outputs.second?.reportPath);
    assert.equal(result.outputs.first?.report?.avg_score, result.outputs.second?.report?.avg_score);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("runs the canonical full dry-run pipeline with a comparison and compatibility report", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-pipeline-full-"));
  try {
    const result = await runLocalPipeline({
      request: requestFixture("83838383-8383-4838-8838-838383838383"),
      config: configFixture(root),
      pipeline: canonicalLocalPipeline(),
    });
    assert.equal(result.status, "completed");
    assert.equal(result.report?.status, "completed");
    assert.ok(result.outputs.compare?.comparison);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
