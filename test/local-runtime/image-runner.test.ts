import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { imageSpecSchema, prepareImageRun, runImagePipeline } from "../../src/local-runtime/image-runner.js";

async function fixture() {
  const dir = await mkdtemp(join(tmpdir(), "tt-image-"));
  await mkdir(join(dir, "data"));
  // Input byte/provenance boundary; real image decoding is exercised by Python tests.
  await writeFile(join(dir, "data", "train.png"), "training bytes");
  await writeFile(join(dir, "data", "heldout.png"), "held-out bytes");
  await writeFile(join(dir, "data", "train.jsonl"), JSON.stringify({ image: "train.png", caption: "red square" }) + "\n");
  await writeFile(join(dir, "data", "val.jsonl"), JSON.stringify({ image: "heldout.png", caption: "blue square" }) + "\n");
  const spec = {
    engine: "image", name: "image smoke", base_model: "MiniMaxAI/MiniMax-H3-Image",
    base_model_revision: "a".repeat(40), text_encoder_revision: "b".repeat(40),
    dataset: { training: "data/train.jsonl", validation: "data/val.jsonl" },
    hyperparameters: { steps: 1 },
  };
  const specPath = join(dir, "tunedtensor.json");
  await writeFile(specPath, JSON.stringify(spec));
  return { dir, spec, specPath, outputDir: join(dir, "run") };
}

test("image dry-run resolves manifest-relative images, pins sources and has no process or output side effects", async () => {
  const f = await fixture();
  try {
    const before = await readdir(f.dir);
    const result = await runImagePipeline({ ...f, dryRun: true }, async () => { throw new Error("must not spawn"); });
    assert.equal("dry_run" in result && result.dry_run, true);
    assert.deepEqual(await readdir(f.dir), before);
    const prepared = await prepareImageRun(f.specPath);
    assert.equal(prepared.training.rows[0]!.image, join(f.dir, "data/train.png"));
    assert.equal(prepared.spec.hyperparameters.width, 512);
  } finally { await rm(f.dir, { recursive: true, force: true }); }
});

test("image split leakage is rejected even when identical bytes use different filenames", async () => {
  const f = await fixture();
  try {
    await writeFile(join(f.dir, "data/heldout.png"), "training bytes");
    await assert.rejects(prepareImageRun(f.specPath), /Duplicate image content/);
  } finally { await rm(f.dir, { recursive: true, force: true }); }
});

test("image runner records failed execution and refuses reusing the run directory", async () => {
  const f = await fixture();
  try {
    await assert.rejects(runImagePipeline(f, async () => ({ exitCode: 7 })), /exited with code 7/);
    assert.match(await readFile(join(f.outputDir, "failure.json"), "utf8"), /failed/);
    await assert.rejects(runImagePipeline(f, async () => { throw new Error("must not spawn"); }), /EEXIST/);
  } finally { await rm(f.dir, { recursive: true, force: true }); }
});

test("image success requires the requested steps and matching adapter bytes", async () => {
  const f = await fixture();
  try {
    const result = await runImagePipeline(f, async (args) => {
      assert.equal(args.command, "uv");
      assert.ok(args.commandArgs.includes("--frozen"));
      assert.ok(args.env?.UV_PROJECT_ENVIRONMENT?.includes("uv-image"));
      const output = dirname(args.commandArgs.at(-1)!);
      await mkdir(join(output, "adapter"));
      const bytes = Buffer.from("fixture adapter");
      await writeFile(join(output, "adapter/adapter_model.safetensors"), bytes);
      await writeFile(join(output, "report.json"), JSON.stringify({
        status: "succeeded", optimizer_steps: 1, baseline_loss: 1, candidate_loss: 0.9,
        adapter_sha256: createHash("sha256").update(bytes).digest("hex"),
      }));
      return { exitCode: 0 };
    });
    assert.equal("status" in result && result.status, "succeeded");
    assert.equal("model_dir" in result && result.model_dir, join(f.outputDir, "adapter"));
  } finally { await rm(f.dir, { recursive: true, force: true }); }
});

test("successful process exit cannot conceal a corrupt adapter", async () => {
  const f = await fixture();
  try {
    await assert.rejects(runImagePipeline(f, async () => {
      await mkdir(join(f.outputDir, "adapter"));
      await writeFile(join(f.outputDir, "adapter/adapter_model.safetensors"), "wrong bytes");
      await writeFile(join(f.outputDir, "report.json"), JSON.stringify({
        status: "succeeded", optimizer_steps: 1, baseline_loss: 1, candidate_loss: 1, adapter_sha256: "0".repeat(64),
      }));
      return { exitCode: 0 };
    }), /checksum/);
    assert.match(await readFile(join(f.outputDir, "failure.json"), "utf8"), /checksum/);
  } finally { await rm(f.dir, { recursive: true, force: true }); }
});

test("image specs reject unsupported configuration instead of silently ignoring it", async () => {
  const f = await fixture();
  try {
    for (const change of [
      { base_model: "MiniMaxAI/MiniMax-H3" }, { base_model_revision: "main" },
      { text_encoder_revision: undefined }, { runtime: { gpu: {} } },
      { hyperparameters: { width: 513 } }, { hyperparameters: { steps: 0 } },
    ]) assert.equal(imageSpecSchema.safeParse({ ...f.spec, ...change }).success, false);
  } finally { await rm(f.dir, { recursive: true, force: true }); }
});
