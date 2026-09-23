import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { Command } from "commander";
import {
  canonicalFoundationPipeline,
  canonicalPipeline,
  createExecutionPlan,
  isFoundationPipeline,
  parsePipeline,
  pipelineFromFoundationHyperparameters,
  pipelineForRunInput,
  type Pipeline,
} from "../pipeline.js";
import { isJsonMode, printJson, printSuccess, printWarning } from "../output.js";
import {
  loadLocalRunInput,
  parseLocalRunInput,
  specHash,
  type LocalRunInput,
} from "../local-runtime/local-project.js";
import { loadLocalRunnerConfig, runLocalPipeline, type LocalPipeline } from "../local-runtime/orchestrator.js";
import { runFoundationPipeline } from "../local-runtime/foundation-runner.js";
import { warningsFromSnapshot } from "../local-runtime/capability.js";
import { readHardwareSnapshot } from "../local-runtime/hardware-snapshot.js";

const DEFAULT_PIPELINE_FILE = "tunedtensor.pipeline.json";
const DEFAULT_SPEC_FILE = "tunedtensor.json";

export function localConfigPath(explicitPath: string | undefined, specPath: string): string | undefined {
  if (explicitPath) return resolve(explicitPath);
  const adjacent = join(dirname(resolve(specPath)), "local-runner.json");
  return existsSync(adjacent) ? adjacent : undefined;
}

function parseList(value?: string): string[] | undefined {
  if (!value) return undefined;
  const ids = value.split(",").map((id) => id.trim()).filter(Boolean);
  if (!ids.length) throw new Error("Step selection must name at least one step.");
  return ids;
}

function loadPipelineFile(path: string): unknown {
  const resolved = resolve(path);
  if (!existsSync(resolved)) throw new Error(`Pipeline file not found: ${path}. Run \`tt pipeline init\` to create one.`);
  try {
    return JSON.parse(readFileSync(resolved, "utf8"));
  } catch (error) {
    throw new Error(`Pipeline file must be JSON: ${(error as Error).message}`);
  }
}

function pipelineFromSpec(input: Extract<LocalRunInput, { kind: "foundation-spec" }>): Pipeline {
  return pipelineFromFoundationHyperparameters(input.spec.name, input.spec.foundation);
}

function resolvePipelineDocument(options: { file?: string; spec: string }, specRequired: boolean) {
  const specPath = resolve(options.spec);
  const requested = options.file ? loadPipelineFile(options.file) : undefined;
  if (!existsSync(specPath)) {
    // Recipe-only inspection is allowed only when the default spec was omitted.
    if (specRequired) throw new Error(`Behavior spec not found: ${options.spec}.`);
    if (requested !== undefined) return { document: parsePipeline(requested) };
    throw new Error(`Behavior spec not found: ${options.spec}. Create tunedtensor.json or pass an explicit --file for a recipe-only preview.`);
  }

  // All downstream consumers use this snapshot, including hardware warnings.
  const source = readFileSync(specPath, "utf8");
  const input = parseLocalRunInput(JSON.parse(source), specPath);
  const document = pipelineForRunInput(input, requested);
  const spec = {
    path: specPath,
    sha256: specHash(source),
    engine: input.kind === "foundation-spec" ? "foundation" : "adapter",
  };
  return { document, input, spec };
}

function outputPlan(plan: unknown, hostWarnings: string[] = []): void {
  if (isJsonMode()) {
    return printJson(hostWarnings.length ? { ...plan as object, host_warnings: hostWarnings } : plan);
  }
  const identity = (plan as { spec?: { path: string; sha256: string } }).spec;
  if (identity) console.log(`Behavior spec: ${identity.path} (sha256 ${identity.sha256})`);
  const typed = plan as { steps: Array<{ id: string; uses: string; target: string; transfers: Array<{ from: string; from_target: string; to_target: string }> }> };
  for (const step of typed.steps) {
    console.log(`${step.id.padEnd(16)} ${step.uses.padEnd(10)} ${step.target}`);
    for (const transfer of step.transfers) console.log(`  transfer ${transfer.from}: ${transfer.from_target} -> ${transfer.to_target}`);
  }
  for (const warning of hostWarnings) printWarning(warning);
}

async function hostWarningsForPipeline(document: Pipeline, input?: LocalRunInput): Promise<string[]> {
  const snapshot = await readHardwareSnapshot();
  if (!snapshot) return [];
  const engine = isFoundationPipeline(parsePipeline(document)) ? "foundation" : "adapter";
  const baseModel = input && input.kind !== "foundation-spec"
    ? input.request.spec_snapshot.base_model
    : undefined;
  return warningsFromSnapshot(snapshot.capabilities, { engine, baseModel });
}

export function registerPipelineCommands(parent: Command): void {
  const pipeline = parent.command("pipeline").description("Create, validate, inspect, and safely dry-run ordered pipeline recipes");

  pipeline.command("init")
    .description("Write a canonical v1 pipeline recipe")
    .option("-f, --file <path>", "Output file", DEFAULT_PIPELINE_FILE)
    .option("--engine <engine>", "adapter (default) or foundation")
    .option("--spec <path>", "Foundation spec whose hyperparameters stamp the DAG")
    .action(async (options: { file: string; engine?: string; spec?: string }) => {
      const engine = options.engine ?? "adapter";
      if (engine !== "adapter" && engine !== "foundation") {
        throw new Error(`--engine must be adapter or foundation, got: ${engine}`);
      }
      if (engine !== "foundation" && options.spec) {
        throw new Error("--spec is only valid with --engine foundation.");
      }
      const path = resolve(options.file);
      if (existsSync(path)) throw new Error(`Pipeline file already exists: ${options.file}.`);
      let recipe: Pipeline;
      if (engine === "foundation") {
        if (options.spec) {
          const input = await loadLocalRunInput(resolve(options.spec));
          if (input.kind !== "foundation-spec") {
            throw new Error("--spec must be a foundation tunedtensor.json.");
          }
          recipe = pipelineFromSpec(input);
        } else {
          recipe = canonicalFoundationPipeline();
        }
      } else {
        recipe = canonicalPipeline("local");
      }
      writeFileSync(path, `${JSON.stringify(recipe, null, 2)}\n`);
      if (isJsonMode()) return printJson({ created: true, path, pipeline: recipe });
      printSuccess(`Created ${options.file}`);
    });

  pipeline.command("validate")
    .description("Validate a pipeline without any execution or transfer")
    .option("-f, --file <path>", "Explicit pipeline recipe (default: derive from behavior spec)")
    .option("--spec <path>", "Local behavior spec to validate and plan against", DEFAULT_SPEC_FILE)
    .action(async (options: { file?: string; spec: string }, command: Command) => {
      try {
        const { document, input, spec } = resolvePipelineDocument(options, command.getOptionValueSource("spec") !== "default");
        const host_warnings = await hostWarningsForPipeline(document, input);
        if (isJsonMode()) {
          return printJson({ valid: true, errors: [], spec, host_warnings });
        }
        printSuccess("Pipeline is valid.");
        for (const warning of host_warnings) printWarning(warning);
      } catch (error) {
        if (!isJsonMode()) throw error;
        process.exitCode = 1;
        printJson({ valid: false, errors: [(error as Error).message] });
      }
    });

  pipeline.command("plan")
    .description("Resolve step targets and required artifact transfers")
    .option("-f, --file <path>", "Explicit pipeline recipe (default: derive from behavior spec)")
    .option("--spec <path>", "Local behavior spec to validate and plan against", DEFAULT_SPEC_FILE)
    .option("--only <ids>", "Comma-separated step IDs to include")
    .option("--skip <ids>", "Comma-separated step IDs to omit")
    .action(async (options: { file?: string; spec: string; only?: string; skip?: string }, command: Command) => {
      const { document, input, spec } = resolvePipelineDocument(options, command.getOptionValueSource("spec") !== "default");
      const plan = createExecutionPlan(document, { only: parseList(options.only), skip: parseList(options.skip) });
      outputPlan({ ...plan, ...(spec ? { spec } : {}) }, await hostWarningsForPipeline(document, input));
    });

  pipeline.command("run")
    .description("Run an ordered local pipeline, or safely preview any pipeline")
    .option("--dry-run", "Resolve and display only; never execute, transfer, or reserve credits")
    .option("-f, --file <path>", "Explicit pipeline recipe (default: derive from behavior spec)")
    .option("--spec <path>", "Local behavior spec", DEFAULT_SPEC_FILE)
    .option("--config <path>", "Local runtime config")
    .option("--output <path>", "Foundation run directory (must not already exist)")
    .option("--resume <path>", "Resume a foundation run directory")
    .option("--only <ids>", "Comma-separated step IDs to include")
    .option("--skip <ids>", "Comma-separated step IDs to omit")
    .action(async (options: { file?: string; spec: string; config?: string; output?: string; resume?: string; dryRun?: boolean; only?: string; skip?: string }, command: Command) => {
      if (options.output && options.resume) {
        throw new Error("--output and --resume are mutually exclusive.");
      }
      const config = await loadLocalRunnerConfig(localConfigPath(options.config, options.spec));
      const { document, input, spec } = resolvePipelineDocument(options, command.getOptionValueSource("spec") !== "default");
      const plan = createExecutionPlan(document, { only: parseList(options.only), skip: parseList(options.skip) });
      const hostWarnings = config.gpu ? [] : await hostWarningsForPipeline(document, input);
      if (options.dryRun || config.dryRun) {
        if (isJsonMode()) {
          return printJson({
            dry_run: true,
            ...(spec ? { spec } : {}),
            ...plan,
            ...(config.gpu ? { gpu: { provider: config.gpu.provider, instanceId: config.gpu.instanceId, region: config.gpu.region, profile: config.gpu.profile } } : {}),
            ...(hostWarnings.length ? { host_warnings: hostWarnings } : {}),
          });
        }
        console.log("Dry run only — no execution, artifact transfer, or credit reservation will occur.");
        if (config.gpu) console.log(`GPU processes: AWS instance ${config.gpu.instanceId}. Orchestration stays local.`);
        return outputPlan({ ...plan, ...(spec ? { spec } : {}) }, hostWarnings);
      }
      if (spec && !isJsonMode()) console.log(`Behavior spec: ${spec.path} (sha256 ${spec.sha256})`);
      for (const warning of hostWarnings) printWarning(warning);
      const remote = plan.steps.find((step) => step.target !== "local");
      if (remote) {
        throw new Error(`Step "${remote.id}" targets cloud execution. Pipeline execution requires local targets; set targets to local and configure gpu in local-runner.json to use your AWS instance.`);
      }
      if (!input) throw new Error("Pipeline execution requires a behavior spec. Pass --spec tunedtensor.json.");
      if (input.kind === "foundation-spec") {
        const result = await runFoundationPipeline({
          spec: input.spec,
          plan,
          specPath: resolve(options.spec),
          ...(options.output || options.resume ? { outputDir: resolve(options.resume ?? options.output!) } : {}),
          resume: Boolean(options.resume),
          gpu: config.gpu,
        });
        if (isJsonMode()) return printJson({ ...result, spec });
        printSuccess(`Foundation pipeline completed. Report: ${result.report_path}`);
        return;
      }
      if (options.output || options.resume) {
        throw new Error("--output and --resume are only valid for foundation pipelines.");
      }
      const localPipeline: LocalPipeline = {
        version: 1,
        ...(plan.name ? { name: plan.name } : {}),
        steps: plan.steps.map(({ transfers: _transfers, ...step }) => step) as LocalPipeline["steps"],
      };
      const result = await runLocalPipeline({ request: input.request, config, pipeline: localPipeline });
      if (isJsonMode()) return printJson({ ...result, spec });
      printSuccess(`Pipeline completed with status ${result.status}.`);
    });
}

export { DEFAULT_PIPELINE_FILE };
