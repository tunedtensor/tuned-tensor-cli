import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, realpath, stat, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { z } from "zod";
import { pythonEnvironmentPath } from "../paths.js";
import { localRuntimePackageRoot } from "./package-root.js";
import { minimalMachineLearningEnvironment } from "./huggingface-cache.js";
import { runLoggedProcess } from "./process-runner.js";
import type { LocalRunReporter } from "./run-reporter.js";

const revision = z.string().regex(/^[a-f0-9]{40}$/i, "Pin a 40-character Hugging Face commit SHA");
const dimension = z.number().int().min(256).max(4096).multipleOf(32);
export const imageSpecSchema = z.object({
  engine: z.literal("image"),
  name: z.string().trim().min(1),
  base_model: z.literal("MiniMaxAI/MiniMax-H3-Image"),
  base_model_revision: revision,
  text_encoder_revision: revision,
  dataset: z.object({ training: z.string().min(1), validation: z.string().min(1) }).strict(),
  hyperparameters: z.object({
    steps: z.number().int().min(1).max(1_000_000).default(100),
    learning_rate: z.number().positive().max(0.01).default(0.00001),
    lora_rank: z.number().int().min(1).max(256).default(16),
    lora_alpha: z.number().int().min(1).max(512).default(16),
    gradient_accumulation_steps: z.number().int().min(1).max(1024).default(1),
    gradient_checkpointing: z.boolean().default(true),
    width: dimension.default(512),
    height: dimension.default(512),
    seed: z.number().int().min(0).max(2_147_483_647).default(42),
  }).strict().prefault({}),
}).strict();
const rowSchema = z.object({ image: z.string().min(1), caption: z.string().trim().min(1) }).strict();
const digest = (bytes: string | Buffer) => createHash("sha256").update(bytes).digest("hex");

/** Local captioned images only: paths are relative to their manifest, never the shell cwd. */
export async function prepareImageRun(specPath: string) {
  const path = resolve(specPath);
  const source = await readFile(path, "utf8");
  const spec = imageSpecSchema.parse(JSON.parse(source));
  const seen = new Set<string>();
  const readSplit = async (value: string) => {
    const manifest = await realpath(resolve(dirname(path), value));
    const raw = await readFile(manifest, "utf8");
    const rows = [];
    for (const [index, line] of raw.split(/\r?\n/).entries()) {
      if (!line.trim()) continue;
      const row = rowSchema.parse(JSON.parse(line));
      const image = await realpath(resolve(dirname(manifest), row.image));
      if (!(await stat(image)).isFile()) throw new Error(`Not an image file: ${image}`);
      const sha256 = digest(await readFile(image));
      if (seen.has(sha256)) throw new Error(`Duplicate image content in training/validation: ${manifest}:${index + 1}`);
      seen.add(sha256);
      rows.push({ ...row, image, sha256 });
    }
    if (!rows.length) throw new Error(`Image manifest is empty: ${manifest}`);
    return { path: manifest, sha256: digest(raw), rows };
  };
  const training = await readSplit(spec.dataset.training);
  const validation = await readSplit(spec.dataset.validation);
  return { spec, source: { path, sha256: digest(source) }, training, validation };
}

export async function runImagePipeline(args: {
  specPath: string;
  outputDir?: string;
  dryRun?: boolean;
  reporter?: LocalRunReporter;
}, spawn = runLoggedProcess) {
  const prepared = await prepareImageRun(args.specPath);
  const preview = {
    engine: "image", experimental: true, base_model: prepared.spec.base_model,
    spec: prepared.source, training_images: prepared.training.rows.length,
    validation_images: prepared.validation.rows.length,
    steps: ["encode", "baseline", "train", "candidate", "export", "verify_export"],
    hyperparameters: prepared.spec.hyperparameters,
  };
  if (args.dryRun) return { ...preview, dry_run: true };
  const output = resolve(args.outputDir ?? join(dirname(prepared.source.path), ".tuned-tensor", "image-runs", randomUUID()));
  await mkdir(dirname(output), { recursive: true });
  await mkdir(output); // Refuse existing runs, including symlinks; never overwrite an adapter.
  const configPath = join(output, "config.json");
  await writeFile(configPath, `${JSON.stringify(prepared, null, 2)}\n`, { flag: "wx" });
  const project = join(localRuntimePackageRoot(import.meta.url), "training/image");
  const hash = createHash("sha256");
  for (const name of ["pyproject.toml", "uv.lock"]) hash.update(await readFile(join(project, name)));
  const env = {
    ...minimalMachineLearningEnvironment(process.env),
    UV_PROJECT_ENVIRONMENT: pythonEnvironmentPath("uv-image", hash.digest("hex").slice(0, 20)),
    PYTHONUNBUFFERED: "1",
  };
  const reportPath = join(output, "report.json");
  try {
    await args.reporter?.onEvent?.({ stage: "training", status: "running",
      message: "Starting H3-Image encoding and LoRA training; missing model components will be downloaded.",
      details: { log_path: join(output, "training.log") },
    });
    const result = await spawn({
      command: "uv", commandArgs: ["run", "--frozen", "--project", project, "python", join(project, "src/train.py"), "--config", configPath],
      env, logPath: join(output, "training.log"), stage: "training", reporter: args.reporter,
    });
    if (result.exitCode !== 0) throw new Error(`Image training exited with code ${result.exitCode}; see ${join(output, "training.log")}`);
    const report = z.object({
      status: z.literal("succeeded"), optimizer_steps: z.number().int().positive(),
      baseline_loss: z.number().finite().nonnegative(), candidate_loss: z.number().finite().nonnegative(),
      adapter_sha256: z.string().regex(/^[a-f0-9]{64}$/),
    }).passthrough().parse(JSON.parse(await readFile(reportPath, "utf8")));
    if (report.optimizer_steps !== prepared.spec.hyperparameters.steps) throw new Error("Image trainer did not complete the requested steps");
    if (digest(await readFile(join(output, "adapter", "adapter_model.safetensors"))) !== report.adapter_sha256) {
      throw new Error("Image adapter checksum does not match its report");
    }
    return { ...report, engine: "image", experimental: true, run_dir: output, report_path: reportPath, model_dir: join(output, "adapter") };
  } catch (error) {
    await writeFile(join(output, "failure.json"), JSON.stringify({ status: "failed", error: String(error) }, null, 2));
    throw error;
  }
}
