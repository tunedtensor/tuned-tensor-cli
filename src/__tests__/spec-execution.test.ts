import { mkdtemp, readFile, rm, writeFile, access } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createProgram } from "../cli.js";
import { setJsonMode } from "../output.js";
import { applySpecUpdate, inspectLocalSpec, prepareSpecUpdate } from "../spec-workspace.js";
import { prepareLocalPipelineAction } from "../local-pipeline-action.js";
import { createExecutionPlan, pipelineFromFoundationHyperparameters } from "../pipeline.js";
import { runFoundationPipeline } from "../local-runtime/foundation-runner.js";
import { loadLocalRunInput } from "../local-runtime/local-project.js";
import { reviewSpec } from "../commands/spec.js";

let root: string;
let originalCwd: string;
beforeEach(async () => { originalCwd = process.cwd(); root = await mkdtemp(join(tmpdir(), "tt-spec-execution-")); });
afterEach(async () => { process.chdir(originalCwd); await rm(root, { recursive: true, force: true }); setJsonMode(false); process.exitCode = 0; vi.restoreAllMocks(); });
async function foundation() {
  const source = await readFile(new URL("../../examples/local-runtime/foundation-spec.json", import.meta.url), "utf8");
  await writeFile(join(root, "tunedtensor.json"), source);
  const input = await loadLocalRunInput(join(root, "tunedtensor.json"));
  if (input.kind !== "foundation-spec") throw new Error("Expected foundation fixture");
  return input.spec;
}

describe("spec is the source of execution settings", () => {
  it("rejects the same invalid spec in review, planning, validation, execution and agent proposals", async () => {
    const spec = await foundation();
    const invalid = { ...spec, system_prompt: "   " };
    const path = join(root, "tunedtensor.json");
    await writeFile(path, JSON.stringify(invalid));
    expect(await reviewSpec(root, "validate")).toMatchObject({ valid: false });
    for (const args of [["plan"], ["validate"], ["run", "--dry-run"]]) {
      await expect(createProgram("test").parseAsync(
        ["pipeline", ...args, "--spec", path], { from: "user" },
      )).rejects.toThrow(/system_prompt must not be blank/);
    }
    await expect(prepareLocalPipelineAction({ workspaceRoot: root })).rejects.toThrow(/system_prompt must not be blank/);

    const log = vi.spyOn(console, "log").mockImplementation(() => {});
    await createProgram("test").parseAsync(["--json", "pipeline", "validate", "--spec", path], { from: "user" });
    expect(JSON.parse(String(log.mock.calls.at(-1)![0]))).toMatchObject({ valid: false });
    expect(process.exitCode).toBe(1);
  });

  it("uses the same spec-derived recipe for direct CLI planning and agent preview", async () => {
    await foundation();
    const prepared = await prepareLocalPipelineAction({ workspaceRoot: root });
    const log = vi.spyOn(console, "log").mockImplementation(() => {});
    await createProgram("test").parseAsync(
      ["--json", "pipeline", "plan", "--spec", join(root, "tunedtensor.json")], { from: "user" },
    );
    const plan = JSON.parse(String(log.mock.calls.at(-1)![0]));
    expect(plan.spec.sha256).toBe(prepared.specSha256);
    expect(plan.steps).toEqual(prepared.plan.steps);
  });

  it("derives current settings even with a stale default-named recipe present; rejects explicit conflicts", async () => {
    const spec = await foundation();
    const recipe = join(root, "tunedtensor.pipeline.json");
    await writeFile(recipe, JSON.stringify(pipelineFromFoundationHyperparameters(spec.name, spec.foundation)));
    const prepared = await prepareSpecUpdate(root, "tunedtensor.json", (await inspectLocalSpec(root)).sha256, { foundation: { pretrain_steps: 7 } });
    await applySpecUpdate(root, prepared.update);
    const log = vi.spyOn(console, "log").mockImplementation(() => {});
    process.chdir(root);
    await createProgram("test").parseAsync(["--json", "pipeline", "run", "--dry-run", "--spec", join(root, "tunedtensor.json")], { from: "user" });
    const plan = JSON.parse(String(log.mock.calls.at(-1)![0]));
    expect(plan.spec.sha256).toBe((await inspectLocalSpec(root)).sha256);
    expect(plan.steps.find((step: { id: string }) => step.id === "pretrain").with.steps).toBe(7);
    await expect(createProgram("test").parseAsync(["pipeline", "run", "--dry-run", "--spec", join(root, "tunedtensor.json"), "--file", recipe], { from: "user" })).rejects.toThrow(/conflicts with the behavior spec/);
    await expect(createProgram("test").parseAsync(["pipeline", "run", "--dry-run", "--spec", join(root, "tunedtensor.json"), "--file", join(root, "typo.json")], { from: "user" })).rejects.toThrow(/not found/);
  });

  it("rejects mismatched model-proposed foundation settings before approval", async () => {
    const spec = await foundation();
    const pipeline = pipelineFromFoundationHyperparameters(spec.name, { ...spec.foundation, pretrain_steps: 99 });
    await expect(prepareLocalPipelineAction({ workspaceRoot: root, pipeline })).rejects.toThrow(/conflicts/);
  });

  it("rejects an explicit null recipe instead of silently deriving a default", async () => {
    await foundation();
    const recipe = join(root, "invalid.pipeline.json");
    await writeFile(recipe, "null");
    await expect(createProgram("test").parseAsync([
      "pipeline", "plan", "--spec", join(root, "tunedtensor.json"), "--file", recipe,
    ], { from: "user" })).rejects.toThrow();
    await expect(prepareLocalPipelineAction({ workspaceRoot: root, pipeline: null })).rejects.toThrow();
  });

  it.each(["plan", "validate", "run"])("%s refuses a missing explicit spec even with an existing recipe", async operation => {
    const spec = await foundation();
    const recipe = join(root, "pipeline.json");
    await writeFile(recipe, JSON.stringify(pipelineFromFoundationHyperparameters(spec.name, spec.foundation)));
    await expect(createProgram("test").parseAsync([
      "pipeline", operation, ...(operation === "run" ? ["--dry-run"] : []),
      "--spec", join(root, "typo.json"), "--file", recipe,
    ], { from: "user" })).rejects.toThrow(/Behavior spec not found/);
  });

  it("the runtime itself rejects conflicting settings before creating artifacts or launching Python", async () => {
    const spec = await foundation();
    const plan = createExecutionPlan(pipelineFromFoundationHyperparameters(spec.name, { ...spec.foundation, depth: 3 }));
    const spawn = vi.fn(); const outputDir = join(root, "run");
    await expect(runFoundationPipeline({ spec, plan, specPath: join(root, "tunedtensor.json"), outputDir, spawnStep: spawn })).rejects.toThrow(/conflicts/);
    expect(spawn).not.toHaveBeenCalled(); await expect(access(outputDir)).rejects.toThrow();
  });

  it("passes edited instructions, examples and training settings to every foundation stage", async () => {
    await foundation();
    const edit = await prepareSpecUpdate(root, "tunedtensor.json", (await inspectLocalSpec(root)).sha256, {
      system_prompt: "Answer arithmetic directly.", guidelines: ["Use digits."], constraints: ["No commentary."],
      foundation: { pretrain_steps: 4, finetune_steps: 3 },
    });
    await applySpecUpdate(root, edit.update);
    const input = await loadLocalRunInput(join(root, "tunedtensor.json"));
    if (input.kind !== "foundation-spec") throw new Error("Expected foundation spec");
    const configs: Array<{ entrypoint: string; config: Record<string, any> }> = [];
    await runFoundationPipeline({ spec: input.spec, specPath: input.path, outputDir: join(root, "run"), plan: createExecutionPlan(pipelineFromFoundationHyperparameters(input.spec.name, input.spec.foundation)),
      spawnStep: async args => {
        const config = JSON.parse(await readFile(args.configPath, "utf8")); configs.push({ entrypoint: args.entrypoint, config });
        if (args.entrypoint === "train_tokenizer.py") await writeFile(join(config.output_dir, "tokenizer.json"), "{}");
        else if (args.entrypoint === "evaluate.py") await writeFile(join(config.output_dir, "report.json"), '{"ok":true}');
        else { await writeFile(join(config.output_dir, "model.safetensors"), "fixture weights"); await writeFile(join(config.output_dir, "config.json"), '{"depth":2,"width":64,"heads":4,"vocab_size":256,"sequence_length":64}'); }
        await writeFile(join(config.output_dir, "metrics.json"), '{"ok":true}');
      } });
    expect(configs).toHaveLength(6);
    for (const { config } of configs) {
      expect(config.system_prompt).toContain("Answer arithmetic directly.");
      expect(config.system_prompt).toContain("Use digits.");
      expect(config.system_prompt).toContain("No commentary.");
      expect(config.examples).toEqual(input.spec.examples);
    }
    expect(configs.find(item => item.entrypoint === "pretrain.py")!.config.steps).toBe(4);
    expect(configs.find(item => item.entrypoint === "finetune.py")!.config.steps).toBe(3);
  });
});
