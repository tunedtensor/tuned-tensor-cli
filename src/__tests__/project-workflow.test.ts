import { discoverShellContext } from "../shell-context.js";
import { mkdtemp, readFile, writeFile, rm, access, symlink } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { resolveProjectConfig } from "../local-runtime/project-workflow.js";
import { migrateProjectWorkflow } from "../local-runtime/project-migration.js";
import { canonicalPipeline } from "../pipeline.js";
import { prepareLocalPipelineAction, validatePreparedLocalPipelineAction } from "../local-pipeline-action.js";
import { loadLocalRunInput } from "../local-runtime/local-project.js";
import { pipelineForRunInput } from "../pipeline.js";
import { projectCloudSpec, projectLocalSpec } from "../project-spec.js";
import { createProgram } from "../cli.js";
import { setJsonMode } from "../output.js";
import { resolvePublishStoreRoot } from "../commands/publish.js";

async function resolveProjectWorkflow(path: string, options: { pipelinePath?: string } = {}) {
  const input = await loadLocalRunInput(path);
  const pipeline = pipelineForRunInput(input, options.pipelinePath ? await json(options.pipelinePath) : undefined);
  return { input, pipeline, ...await resolveProjectConfig(path, undefined, undefined, input) };
}
const spec = { name: "Sentiment", base_model: "Qwen/Qwen3.5-2B", system_prompt: "Return a sentiment label.", guidelines: ["Use lowercase."], examples: [{ input: "great", output: "positive" }, { input: "bad", output: "negative" }], hyperparameters: { n_epochs: 2 } };
let root: string;
let file: string;
const json = async (path: string) => JSON.parse(await readFile(path, "utf8"));
beforeEach(async () => { root = await mkdtemp(join(tmpdir(), "tt-single-spec-")); file = join(root, "tunedtensor.json"); await writeFile(file, JSON.stringify(spec)); });
afterEach(async () => { setJsonMode(false); vi.restoreAllMocks(); await rm(root, { recursive: true, force: true }); });

describe("single spec workflows", () => {
  it("keeps shell context and publish lookup aligned with the core runtime section", async () => {
    await writeFile(file, JSON.stringify({ ...spec, runtime: { storeRoot: "my-state", artifactRoot: "my-artifacts" } }));
    const context = await discoverShellContext({ cwd: root, env: { HOME: root } });
    expect(context.local.storeRoot).toBe(join(root, "my-state"));
    expect(context.local.artifactRoot).toBe(join(root, "my-artifacts"));
    expect(context.local.configPath).toBeUndefined();
    expect(resolvePublishStoreRoot({ cwd: root, env: { HOME: root } })).toBe(context.local.storeRoot);
  });

  it("materializes an advanced recipe in the existing spec without a sidecar", async () => {
    vi.spyOn(console, "log").mockImplementation(() => undefined);
    await createProgram("test").parseAsync(["node", "tt", "pipeline", "init", "--spec", file]);
    expect((await json(file)).pipeline.steps).toHaveLength(4);
    await expect(access(join(root, "tunedtensor.pipeline.json"))).rejects.toThrow();
    expect((await json(file)).system_prompt).toBe(spec.system_prompt);
  });

  it("resolves foundation artifact placement and rejects unsupported settings", async () => {
    const foundation = { name: "tiny", system_prompt: "Answer directly.", engine: "foundation", examples: spec.examples, foundation: {}, runtime: { artifactRoot: "output" } };
    await writeFile(file, JSON.stringify(foundation));
    const result = await resolveProjectWorkflow(file);
    expect(result.input.kind).toBe("foundation-spec");
    expect(result.config.artifactRoot).toBe(join(root, "output"));
    expect(result.pipeline.steps[0]?.uses).toBe("tokenize");
    expect((await prepareLocalPipelineAction({ workspaceRoot: root })).engine).toBe("foundation");
    await writeFile(file, JSON.stringify({ ...foundation, evaluation: {} }));
    await expect(resolveProjectWorkflow(file)).rejects.toThrow(/Foundation uses/);
  });

  it("resolves runtime and evaluation paths relative to the spec, independently of cwd", async () => {
    await writeFile(file, JSON.stringify({ ...spec, runtime: { artifactRoot: "output", storeRoot: "state", paths: { modelCache: "cache" } }, evaluation: { scoring: { mode: "json_fields", fields: ["label"] }, generalRegression: { dataset: "regression.jsonl" } } }));
    const result = await resolveProjectWorkflow(file);
    expect(result.config).toMatchObject({ artifactRoot: join(root, "output"), storeRoot: join(root, "state"), paths: { modelCache: join(root, "cache") }, evaluation: { scoring: { mode: "json_fields", fields: ["label"] }, generalRegression: { dataset: join(root, "regression.jsonl") } } });
    expect(result.pipeline.steps.map(s => s.id)).toEqual(["baseline", "train", "candidate", "compare"]);
    expect(result.input.kind === "spec" && result.input.request.spec_snapshot).not.toHaveProperty("runtime");
    expect(resolvePublishStoreRoot({ cwd: root })).toBe(join(root, "state"));
    expect((await resolveProjectConfig(file)).config).toEqual(result.config);
  });

  it("keeps advanced recipes in the spec and produces the same agent and CLI plan", async () => {
    const pipeline = canonicalPipeline("local"); pipeline.steps = pipeline.steps.slice(0, 1);
    await writeFile(file, JSON.stringify({ ...spec, pipeline }));
    const cli = await resolveProjectWorkflow(file);
    const agent = await prepareLocalPipelineAction({ workspaceRoot: root });
    expect(cli.pipeline).toEqual(agent.pipeline);
    expect(agent.plan.steps).toHaveLength(1);
    await expect(prepareLocalPipelineAction({ workspaceRoot: root, pipeline: canonicalPipeline("local") })).rejects.toThrow(/Conflicting pipeline/);
  });

  it("invalidates pipeline approvals when only evaluation settings change", async () => {
    const p = await prepareLocalPipelineAction({ workspaceRoot: root });
    await writeFile(file, JSON.stringify({ ...spec, evaluation: { scoring: { mode: "json_fields", fields: ["label"] } } }));
    await expect(validatePreparedLocalPipelineAction({ workspaceRoot: root, pipeline: p.pipeline, specPath: p.specPath, expectedSpecSha256: p.specSha256, expectedWorkspaceFingerprint: p.workspaceFingerprint, dryRun: true })).rejects.toThrow(/spec changed/i);
  });

  it("rejects misspelled settings and conflicting legacy sources before execution", async () => {
    await writeFile(file, JSON.stringify({ ...spec, runtime: { artifactRot: "typo" } }));
    await expect(resolveProjectWorkflow(file)).rejects.toThrow();
    await writeFile(file, JSON.stringify({ ...spec, evaluation: {} }));
    await writeFile(join(root, "local-runner.json"), "{}");
    await expect(resolveProjectWorkflow(file)).rejects.toThrow(/Conflicting/);
  });

  it("rejects an explicit missing recipe instead of silently running defaults", async () => {
    await expect(resolveProjectWorkflow(file, { pipelinePath: join(root, "absent.json") })).rejects.toThrow(/ENOENT/);
  });

  it("migrates settings and custom pipeline with recoverable backups and identical resolution", async () => {
    const pipeline = canonicalPipeline("local"); pipeline.steps = pipeline.steps.slice(0, 1);
    await writeFile(join(root, "tunedtensor.pipeline.json"), JSON.stringify(pipeline));
    await writeFile(join(root, "local-runner.json"), JSON.stringify({ artifactRoot: "artifacts", storeRoot: "state", evaluation: { inference: { maxNewTokens: 9 } } }));
    const before = await resolveProjectWorkflow(file, { pipelinePath: join(root, "tunedtensor.pipeline.json") });
    const migrated = await migrateProjectWorkflow(file);
    const after = await resolveProjectWorkflow(file);
    expect(after.config).toEqual(before.config);
    expect(after.pipeline).toEqual(before.pipeline);
    expect(migrated.backups).toHaveLength(3);
    await expect(access(join(root, "local-runner.json"))).rejects.toThrow();
    expect(await json(file + ".bak")).toEqual(spec);
    expect((await prepareLocalPipelineAction({ workspaceRoot: root })).plan.steps).toHaveLength(1);
    expect((await migrateProjectWorkflow(file)).migrated).toBe(false);
  });

  it("refuses conflicting migrations without modifying originals", async () => {
    await writeFile(file, JSON.stringify({ ...spec, runtime: { storeRoot: "new" } }));
    await writeFile(join(root, "local-runner.json"), JSON.stringify({ storeRoot: "old" }));
    const before = await readFile(file, "utf8");
    await expect(migrateProjectWorkflow(file)).rejects.toThrow(/Conflicting/);
    expect(await readFile(file, "utf8")).toBe(before);
    await access(join(root, "local-runner.json"));
  });

  it("validates nested runtime settings even for CLI dry-runs", async () => {
    await writeFile(file, JSON.stringify({ ...spec, evaluation: { scoring: { mode: "typo" } } }));
    await expect(createProgram("test").parseAsync(["node", "tt", "pipeline", "run", "--dry-run", "--spec", file])).rejects.toThrow();
  });

  it("keeps local workflow fields out of the cloud behavior projection", () => {
    const project = { ...spec, runtime: { storeRoot: "private" }, evaluation: {}, pipeline: canonicalPipeline("local") };
    expect(projectCloudSpec(project).body).not.toHaveProperty("runtime");
    expect(projectCloudSpec(project).body).not.toHaveProperty("evaluation");
    expect(projectCloudSpec(project).body).not.toHaveProperty("pipeline");
    expect(projectLocalSpec(project).body).toMatchObject(project);
  });
});


describe("single-spec review regressions", () => {
  it("requires conversation recipe changes to be saved in the spec first", async () => {
    const pipeline = canonicalPipeline("local"); pipeline.steps = pipeline.steps.slice(0, 1);
    await expect(prepareLocalPipelineAction({ workspaceRoot: root, pipeline })).rejects.toThrow(/saved spec drives execution/);
    await writeFile(file, JSON.stringify({ ...spec, pipeline }));
    expect((await prepareLocalPipelineAction({ workspaceRoot: root, pipeline })).plan.steps).toHaveLength(1);
  });

  it("preserves AWS GPU placement and resolves its identity relative to the spec", async () => {
    const gpu = { provider: "aws", instanceId: "i-0123456789abcdef0", user: "ubuntu", identityFile: "keys/gpu.pem" };
    await writeFile(file, JSON.stringify({ ...spec, runtime: { gpu } }));
    const result = await resolveProjectWorkflow(file);
    expect(result.config.gpu).toMatchObject({ ...gpu, identityFile: join(root, "keys/gpu.pem") });
    expect((await prepareLocalPipelineAction({ workspaceRoot: root })).resolvedConfig?.gpu).toEqual(result.config.gpu);
  });

  it("migrates foundation AWS placement without adding unsupported adapter settings", async () => {
    await writeFile(file, JSON.stringify({ engine: "foundation", name: "tiny", system_prompt: "Answer directly.", examples: spec.examples, foundation: {} }));
    await writeFile(join(root, "local-runner.json"), JSON.stringify({ gpu: { provider: "aws", instanceId: "i-0123456789abcdef0", user: "ubuntu" } }));
    await migrateProjectWorkflow(file);
    expect((await resolveProjectWorkflow(file)).config.gpu?.provider).toBe("aws");
    expect(await json(file)).not.toHaveProperty("evaluation");
  });

  it("leaves no new backups or changes when a later backup already exists", async () => {
    await writeFile(join(root, "local-runner.json"), "{}");
    await writeFile(join(root, "local-runner.json.bak"), "existing backup");
    await expect(migrateProjectWorkflow(file)).rejects.toThrow(/EEXIST/);
    expect(await json(file)).toEqual(spec);
    await expect(access(file + ".bak")).rejects.toThrow();
    expect(await readFile(join(root, "local-runner.json.bak"), "utf8")).toBe("existing backup");
  });

  it("rejects symlink migration and preserves both the source and legacy inputs", async () => {
    await writeFile(join(root, "local-runner.json"), "{}");
    const link = join(root, "linked.json");
    await symlink(file, link);
    await expect(migrateProjectWorkflow(link)).rejects.toThrow(/regular spec/);
    expect(await json(file)).toEqual(spec);
    await access(join(root, "local-runner.json"));
  });
});
