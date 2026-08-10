import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { Command } from "commander";
import { canonicalPipeline, createExecutionPlan, PIPELINE_TARGETS, validatePipeline, type Pipeline, type PipelineTarget } from "../pipeline.js";
import { isJsonMode, printJson, printSuccess } from "../output.js";
import { loadLocalRunInput, assertLocalRunInputReady } from "../local-runtime/local-project.js";
import { loadLocalRunnerConfig, runLocalPipeline, type LocalPipeline } from "../local-runtime/orchestrator.js";

const DEFAULT_PIPELINE_FILE = "tunedtensor.pipeline.json";

function parseList(value?: string): string[] | undefined {
  if (!value) return undefined;
  const ids = value.split(",").map((id) => id.trim()).filter(Boolean);
  if (!ids.length) throw new Error("Step selection must name at least one step.");
  return ids;
}

function loadPipeline(path: string): unknown {
  const resolved = resolve(path);
  if (!existsSync(resolved)) throw new Error(`Pipeline file not found: ${path}. Run \`tt pipeline init\` to create one.`);
  try {
    return JSON.parse(readFileSync(resolved, "utf8"));
  } catch (error) {
    throw new Error(`Pipeline file must be JSON: ${(error as Error).message}`);
  }
}

function outputPlan(plan: unknown): void {
  if (isJsonMode()) return printJson(plan);
  const typed = plan as { steps: Array<{ id: string; uses: string; target: string; transfers: Array<{ from: string; from_target: string; to_target: string }> }> };
  for (const step of typed.steps) {
    console.log(`${step.id.padEnd(16)} ${step.uses.padEnd(10)} ${step.target}`);
    for (const transfer of step.transfers) console.log(`  transfer ${transfer.from}: ${transfer.from_target} -> ${transfer.to_target}`);
  }
}

export function registerPipelineCommands(parent: Command): void {
  const pipeline = parent.command("pipeline").description("Create, validate, inspect, and safely dry-run ordered pipeline recipes");

  pipeline.command("init")
    .description("Write a canonical v1 pipeline recipe")
    .option("-f, --file <path>", "Output file", DEFAULT_PIPELINE_FILE)
    .option("--target <target>", "Default target: local or cloud", "local")
    .action((options: { file: string; target: string }) => {
      if (!(PIPELINE_TARGETS as readonly string[]).includes(options.target)) throw new Error("--target must be local or cloud.");
      const path = resolve(options.file);
      if (existsSync(path)) throw new Error(`Pipeline file already exists: ${options.file}.`);
      const recipe = canonicalPipeline(options.target as PipelineTarget);
      writeFileSync(path, `${JSON.stringify(recipe, null, 2)}\n`);
      if (isJsonMode()) return printJson({ created: true, path, pipeline: recipe });
      printSuccess(`Created ${options.file}`);
    });

  pipeline.command("validate")
    .description("Validate a pipeline without any execution or transfer")
    .option("-f, --file <path>", "Pipeline file", DEFAULT_PIPELINE_FILE)
    .action((options: { file: string }) => {
      const errors = validatePipeline(loadPipeline(options.file));
      const result = { valid: errors.length === 0, errors };
      if (isJsonMode()) return printJson(result);
      if (errors.length) throw new Error(`Invalid pipeline:\n- ${errors.join("\n- ")}`);
      printSuccess("Pipeline is valid.");
    });

  pipeline.command("plan")
    .description("Resolve step targets and required artifact transfers")
    .option("-f, --file <path>", "Pipeline file", DEFAULT_PIPELINE_FILE)
    .option("--only <ids>", "Comma-separated step IDs to include")
    .option("--skip <ids>", "Comma-separated step IDs to omit")
    .action((options: { file: string; only?: string; skip?: string }) => {
      outputPlan(createExecutionPlan(loadPipeline(options.file) as Pipeline, { only: parseList(options.only), skip: parseList(options.skip) }));
    });

  pipeline.command("run")
    .description("Run an ordered local pipeline, or safely preview any pipeline")
    .option("--dry-run", "Resolve and display only; never execute, transfer, or reserve credits")
    .option("-f, --file <path>", "Pipeline file", DEFAULT_PIPELINE_FILE)
    .option("--spec <path>", "Local behavior spec", "tunedtensor.json")
    .option("--config <path>", "Local runtime config")
    .option("--only <ids>", "Comma-separated step IDs to include")
    .option("--skip <ids>", "Comma-separated step IDs to omit")
    .action(async (options: { file: string; spec: string; config?: string; dryRun?: boolean; only?: string; skip?: string }) => {
      const plan = createExecutionPlan(loadPipeline(options.file) as Pipeline, { only: parseList(options.only), skip: parseList(options.skip) });
      if (options.dryRun) {
        if (isJsonMode()) return printJson({ dry_run: true, ...plan });
        console.log("Dry run only — no execution, artifact transfer, or credit reservation will occur.");
        return outputPlan(plan);
      }
      const remote = plan.steps.find((step) => step.target !== "local");
      if (remote) {
        throw new Error(`Step "${remote.id}" targets cloud execution. Hosted and mixed pipeline execution is not enabled by this CLI build; use --dry-run.`);
      }
      const input = await loadLocalRunInput(resolve(options.spec));
      assertLocalRunInputReady(input.request);
      const config = await loadLocalRunnerConfig(options.config ? resolve(options.config) : undefined);
      const localPipeline: LocalPipeline = {
        version: 1,
        ...(plan.name ? { name: plan.name } : {}),
        steps: plan.steps.map(({ transfers: _transfers, ...step }) => step) as LocalPipeline["steps"],
      };
      const result = await runLocalPipeline({ request: input.request, config, pipeline: localPipeline });
      if (isJsonMode()) return printJson(result);
      printSuccess(`Pipeline completed with status ${result.status}.`);
    });
}

export { DEFAULT_PIPELINE_FILE };
