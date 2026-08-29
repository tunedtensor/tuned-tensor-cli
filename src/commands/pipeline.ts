import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { Command } from "commander";
import {
  canonicalFoundationPipeline,
  canonicalPipeline,
  createExecutionPlan,
  isFoundationPipeline,
  parsePipeline,
  pipelineFromFoundationHyperparameters,
  validatePipeline,
  type Pipeline,
} from "../pipeline.js";
import { isJsonMode, printJson, printSuccess, printWarning } from "../output.js";
import {
  assertFoundationSpecReady,
  assertLocalRunInputReady,
  loadLocalRunInput,
  type LocalRunInput,
} from "../local-runtime/local-project.js";
import { loadLocalRunnerConfig, runLocalPipeline, type LocalPipeline } from "../local-runtime/orchestrator.js";
import { runFoundationPipeline } from "../local-runtime/foundation-runner.js";
import { warningsFromSnapshot } from "../local-runtime/capability.js";
import { readHardwareSnapshot } from "../local-runtime/hardware-snapshot.js";

const DEFAULT_PIPELINE_FILE = "tunedtensor.pipeline.json";
const DEFAULT_SPEC_FILE = "tunedtensor.json";

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

function isParsedFoundationPipeline(document: unknown): boolean {
  try {
    return isFoundationPipeline(parsePipeline(document));
  } catch {
    return false;
  }
}

async function resolvePipelineDocument(options: { file: string; spec?: string }): Promise<unknown> {
  const filePath = resolve(options.file);
  const fileExists = existsSync(filePath);
  const specPath = options.spec ? resolve(options.spec) : undefined;
  const specInput = specPath && existsSync(specPath) ? await loadLocalRunInput(specPath) : undefined;

  if (specInput?.kind === "foundation-spec") {
    if (fileExists) {
      const document = loadPipelineFile(options.file);
      if (isParsedFoundationPipeline(document)) return document;
      throw new Error(
        `Pipeline file ${options.file} is an adapter recipe, but ${options.spec} is a foundation spec. Use a foundation pipeline, omit --file, or pass an adapter spec.`,
      );
    }
    return pipelineFromSpec(specInput);
  }

  if (fileExists) {
    const document = loadPipelineFile(options.file);
    if (specInput && isParsedFoundationPipeline(document)) {
      throw new Error(
        `Pipeline file ${options.file} is a foundation recipe, but ${options.spec} is an adapter spec. Use a foundation spec, omit --file, or pass an adapter pipeline.`,
      );
    }
    return document;
  }
  if (specPath && !existsSync(specPath)) await loadLocalRunInput(specPath);
  throw new Error(
    `Pipeline file not found: ${options.file}. Run \`tt pipeline init\` or pass a foundation --spec.`,
  );
}

function outputPlan(plan: unknown, hostWarnings: string[] = []): void {
  if (isJsonMode()) {
    return printJson(hostWarnings.length ? { ...plan as object, host_warnings: hostWarnings } : plan);
  }
  const typed = plan as { steps: Array<{ id: string; uses: string; target: string; transfers: Array<{ from: string; from_target: string; to_target: string }> }> };
  for (const step of typed.steps) {
    console.log(`${step.id.padEnd(16)} ${step.uses.padEnd(10)} ${step.target}`);
    for (const transfer of step.transfers) console.log(`  transfer ${transfer.from}: ${transfer.from_target} -> ${transfer.to_target}`);
  }
  for (const warning of hostWarnings) printWarning(warning);
}

async function hostWarningsForPipeline(document: unknown, specPath?: string): Promise<string[]> {
  const snapshot = await readHardwareSnapshot();
  if (!snapshot) return [];
  let engine: "adapter" | "foundation" = "adapter";
  let baseModel: string | undefined;
  try {
    if (isFoundationPipeline(parsePipeline(document))) engine = "foundation";
  } catch {
    // Use the spec below when the document is not a portable pipeline yet.
  }
  if (specPath && existsSync(specPath)) {
    try {
      const input = await loadLocalRunInput(specPath);
      if (input.kind === "foundation-spec") {
        engine = "foundation";
      } else if (input.kind === "spec") {
        engine = "adapter";
        baseModel = input.request.spec_snapshot.base_model;
      }
    } catch {
      // Spec parse errors are reported by the existing validate/run paths.
    }
  }
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
    .option("-f, --file <path>", "Pipeline file", DEFAULT_PIPELINE_FILE)
    .option("--spec <path>", "Local behavior spec used to detect leftover engine mismatches", DEFAULT_SPEC_FILE)
    .action(async (options: { file: string; spec?: string }) => {
      const document = await resolvePipelineDocument(options);
      const errors = validatePipeline(document);
      const host_warnings = await hostWarningsForPipeline(document, options.spec);
      const result = { valid: errors.length === 0, errors, ...(host_warnings.length ? { host_warnings } : {}) };
      if (isJsonMode()) return printJson(result);
      if (errors.length) throw new Error(`Invalid pipeline:\n- ${errors.join("\n- ")}`);
      printSuccess("Pipeline is valid.");
      for (const warning of host_warnings) printWarning(warning);
    });

  pipeline.command("plan")
    .description("Resolve step targets and required artifact transfers")
    .option("-f, --file <path>", "Pipeline file", DEFAULT_PIPELINE_FILE)
    .option("--spec <path>", "Local behavior spec used to detect leftover engine mismatches", DEFAULT_SPEC_FILE)
    .option("--only <ids>", "Comma-separated step IDs to include")
    .option("--skip <ids>", "Comma-separated step IDs to omit")
    .action(async (options: { file: string; spec?: string; only?: string; skip?: string }) => {
      const document = await resolvePipelineDocument(options);
      const plan = createExecutionPlan(document as Pipeline, { only: parseList(options.only), skip: parseList(options.skip) });
      outputPlan(plan, await hostWarningsForPipeline(document, options.spec));
    });

  pipeline.command("run")
    .description("Run an ordered local pipeline, or safely preview any pipeline")
    .option("--dry-run", "Resolve and display only; never execute, transfer, or reserve credits")
    .option("-f, --file <path>", "Pipeline file", DEFAULT_PIPELINE_FILE)
    .option("--spec <path>", "Local behavior spec", DEFAULT_SPEC_FILE)
    .option("--config <path>", "Local runtime config")
    .option("--only <ids>", "Comma-separated step IDs to include")
    .option("--skip <ids>", "Comma-separated step IDs to omit")
    .action(async (options: { file: string; spec: string; config?: string; dryRun?: boolean; only?: string; skip?: string }) => {
      const document = await resolvePipelineDocument(options);
      const plan = createExecutionPlan(document as Pipeline, { only: parseList(options.only), skip: parseList(options.skip) });
      const hostWarnings = await hostWarningsForPipeline(document, options.spec);
      if (options.dryRun) {
        if (isJsonMode()) {
          return printJson({
            dry_run: true,
            ...plan,
            ...(hostWarnings.length ? { host_warnings: hostWarnings } : {}),
          });
        }
        console.log("Dry run only — no execution, artifact transfer, or credit reservation will occur.");
        return outputPlan(plan, hostWarnings);
      }
      for (const warning of hostWarnings) printWarning(warning);
      const remote = plan.steps.find((step) => step.target !== "local");
      if (remote) {
        throw new Error(`Step "${remote.id}" targets cloud execution. This CLI is local-only; rewrite that step to local or use --dry-run.`);
      }
      const input = await loadLocalRunInput(resolve(options.spec));
      if (isParsedFoundationPipeline(document) || input.kind === "foundation-spec") {
        if (input.kind !== "foundation-spec") {
          throw new Error("Foundation pipelines require a foundation tunedtensor.json --spec.");
        }
        assertFoundationSpecReady(input.spec);
        const result = await runFoundationPipeline({
          spec: input.spec,
          plan,
          specPath: resolve(options.spec),
        });
        if (isJsonMode()) return printJson(result);
        printSuccess(`Foundation pipeline completed. Report: ${result.report_path}`);
        return;
      }
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
