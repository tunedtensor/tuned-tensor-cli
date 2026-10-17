import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
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
import { formatDate, formatStatus, isJsonMode, printDetail, printJson, printSuccess, printTable, printWarning, shortId, truncate } from "../output.js";
import {
  createBackgroundRunExecutor,
  executeBackgroundRunCancel,
  executeBackgroundRunStart,
  prepareBackgroundRunStart,
} from "../local-background-run-action.js";
import {
  backgroundRunsRoot,
  createBackgroundRunStore,
  currentRequester,
  describeBackgroundRun,
  isActiveBackgroundRun,
  superviseBackgroundRun,
} from "../local-runtime/background-runs.js";
import {
  loadLocalRunInput,
  parseLocalRunInput,
  specHash,
  type LocalRunInput,
} from "../local-runtime/local-project.js";
import { runLocalPipeline, type LocalPipeline } from "../local-runtime/orchestrator.js";
import { resolveProjectConfig } from "../local-runtime/project-workflow.js";
import { migrateProjectWorkflow } from "../local-runtime/project-migration.js";
import { runFoundationPipeline } from "../local-runtime/foundation-runner.js";
import { runDecisionPipeline } from "../local-runtime/decision-runner.js";
import { warningsFromSnapshot } from "../local-runtime/capability.js";
import { readHardwareSnapshot } from "../local-runtime/hardware-snapshot.js";
import { ensureDatasetCached } from "../local-runtime/dataset-prefetch.js";
import { applySpecLock, enforceSpecLock, readSpecLock } from "../local-runtime/assets.js";

import { createRunConsoleReporter } from "../local-runtime/console-reporter.js";

const DEFAULT_PIPELINE_FILE = "tunedtensor.pipeline.json";
const DEFAULT_SPEC_FILE = "tunedtensor.json";

function printPipelineWarning(message: string): void {
  // Background supervisors parse stdout as one JSON result. Keep advisory
  // output in the log, including warnings from legacy config and host sizing.
  if (isJsonMode()) console.error(`Warning: ${message}`);
  else printWarning(message);
}

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
    engine: specEngine(input),
  };
  return { document, input, spec };
}

function specEngine(input: LocalRunInput): "adapter" | "foundation" | "decision" {
  if (input.kind === "foundation-spec") return "foundation";
  return input.kind === "decision-spec" ? "decision" : "adapter";
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
  if (input?.kind === "decision-spec") return warningsFromSnapshot(snapshot.capabilities, { engine: "decision" });
  const engine = isFoundationPipeline(parsePipeline(document)) ? "foundation" : "adapter";
  const baseModel = input && input.kind !== "foundation-spec"
    ? input.request.spec_snapshot.base_model
    : undefined;
  return warningsFromSnapshot(snapshot.capabilities, { engine, baseModel });
}

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function printDecisionSummary(result: Awaited<ReturnType<typeof runDecisionPipeline>>): void {
  for (const step of result.steps) {
    if (step.decision_metrics) {
      const metrics = step.decision_metrics;
      console.log(`${step.id.padEnd(12)} accuracy ${percent(metrics.accuracy)}  log-loss ${metrics.log_loss.toFixed(3)}  brier ${metrics.brier.toFixed(3)}  (${step.report?.total ?? 0} held-out)`);
    }
    if (step.comparison) {
      const comparison = step.comparison;
      const sign = (value: number) => (value >= 0 ? "+" : "");
      console.log(`${step.id.padEnd(12)} accuracy ${sign(comparison.pass_rate_delta)}${percent(comparison.pass_rate_delta)}  log-loss ${sign(comparison.log_loss_delta)}${comparison.log_loss_delta.toFixed(3)}  improved ${comparison.improvements}, regressed ${comparison.regressions}`);
    }
  }
  if (result.model_dir) console.log(`Tuned decision model: ${result.model_dir}`);
  printSuccess(`Decision pipeline completed. Report: ${result.report_path}`);
}

export function registerPipelineCommands(parent: Command): void {
  const pipeline = parent.command("pipeline").description("Create, validate, dry-run, run, and start background runs of ordered pipeline recipes");

  pipeline.command("init")
    .description("Write a canonical v1 pipeline recipe")
    .option("-f, --file <path>", "Legacy standalone recipe output (default: embed in tunedtensor.json)")
    .option("--engine <engine>", "adapter (default) or foundation")
    .option("--spec <path>", "Foundation spec whose hyperparameters stamp the DAG")
    .action(async (options: { file?: string; engine?: string; spec?: string }) => {
      if (!options.file) {
        const path = resolve(options.spec ?? DEFAULT_SPEC_FILE);
        const raw = JSON.parse(readFileSync(path, "utf8"));
        const input = parseLocalRunInput(raw, path);
        if (input.kind === "request") throw new Error("Pipeline init requires a behavior spec.");
        if (options.engine && options.engine !== (input.spec.engine ?? "adapter")) throw new Error("--engine must match the spec.");
        if (raw.pipeline) throw new Error("The spec already contains a pipeline.");
        raw.pipeline = pipelineForRunInput(input);
        writeFileSync(path, `${JSON.stringify(raw, null, 2)}\n`);
        if (isJsonMode()) return printJson({ created: true, path, pipeline: raw.pipeline });
        return printSuccess(`Saved pipeline in ${path}`);
      }
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

  pipeline.command("migrate")
    .description("Move legacy configuration into tunedtensor.json, retaining .bak copies")
    .option("--spec <path>", "Local behavior spec", DEFAULT_SPEC_FILE)
    .action(async (options: { spec: string }) => {
      const result = await migrateProjectWorkflow(resolve(options.spec));
      if (isJsonMode()) return printJson(result);
      printSuccess(result.migrated ? `Migrated ${result.path}; originals retained as .bak files.` : "No legacy configuration found.");
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
    .option("--output <path>", "Foundation or decision run directory (must not already exist)")
    .option("--resume <path>", "Resume a foundation run directory")
    .option("--only <ids>", "Comma-separated step IDs to include")
    .option("--skip <ids>", "Comma-separated step IDs to omit")
    .option("--verbose", "Also stream training and evaluation process logs")
    .option("--quiet", "Do not print live stage progress")
    .action(async (options: { file?: string; spec: string; config?: string; output?: string; resume?: string; dryRun?: boolean; only?: string; skip?: string; verbose?: boolean; quiet?: boolean }, command: Command) => {
      if (options.output && options.resume) {
        throw new Error("--output and --resume are mutually exclusive.");
      }
      const { document, input, spec } = resolvePipelineDocument(options, command.getOptionValueSource("spec") !== "default");
      const { config } = await resolveProjectConfig(options.spec, options.config, printPipelineWarning, input);
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
      for (const warning of hostWarnings) printPipelineWarning(warning);
      const remote = plan.steps.find((step) => step.target !== "local");
      if (remote) {
        throw new Error(`Step "${remote.id}" targets cloud execution. Pipeline execution requires local targets; set targets to local and configure runtime.gpu in tunedtensor.json to use your AWS instance.`);
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
      if (input.kind === "decision-spec") {
        if (options.resume) throw new Error("--resume is only valid for foundation pipelines.");
        const result = await runDecisionPipeline({
          spec: input.spec,
          plan,
          specPath: resolve(options.spec),
          ...(options.output ? { outputDir: resolve(options.output) } : {}),
          onProgress: isJsonMode() ? undefined : (message) => console.error(message),
        });
        if (isJsonMode()) return printJson({ ...result, spec });
        printDecisionSummary(result);
        return;
      }
      if (options.output || options.resume) {
        throw new Error("--output is only valid for foundation and decision pipelines; --resume only for foundation.");
      }
      const localPipeline: LocalPipeline = {
        version: 1,
        ...(plan.name ? { name: plan.name } : {}),
        steps: plan.steps.map(({ transfers: _transfers, ...step }) => step) as LocalPipeline["steps"],
      };
      // Download a missing dataset before starting the recorded pipeline.
      const reporter = options.quiet ? undefined : createRunConsoleReporter({ verbose: options.verbose });
      // The spec's lock pins dataset revisions and file digests (tunedtensor.lock.json).
      const lock = spec ? await readSpecLock(spec.path) : undefined;
      const { request } = await ensureDatasetCached({ request: applySpecLock(input.request, lock), config, reporter });
      if (spec) await enforceSpecLock({ specPath: spec.path, request, config, reporter });
      const result = await runLocalPipeline({
        request,
        config,
        pipeline: localPipeline,
        ...(input.kind === "spec" ? { projectSpec: input.spec } : {}),
        ...(spec ? { specFile: { path: spec.path, sha256: spec.sha256 } } : {}),
        ...(reporter ? { reporter } : {}),
      });
      if (!isJsonMode()) {
        console.error(`Run ${result.request.run_id}: \`tt runs audit ${result.request.run_id}\` verifies its record and shows timing and provenance.`);
      }
      if (isJsonMode()) return printJson({ ...result, spec });
      printSuccess(`Pipeline completed with status ${result.status}.`);
    });

  registerBackgroundRunCommands(pipeline);
}

function registerBackgroundRunCommands(pipeline: Command): void {
  const store = () => createBackgroundRunStore(backgroundRunsRoot());

  pipeline.command("start")
    .description("Start a real run in the background with an audit record; check it with tt pipeline status")
    .requiredOption("--reason <text>", "Why this run is being done (kept in its audit record)")
    .option("--spec <path>", "Local behavior spec", DEFAULT_SPEC_FILE)
    .action(async (options: { reason: string; spec: string }) => {
      const specPath = resolve(options.spec);
      const executor = createBackgroundRunExecutor({ env: process.env, entrypoint: process.argv[1] });
      const workspaceRoot = dirname(specPath);
      const prepared = await prepareBackgroundRunStart({
        workspaceRoot,
        specPath: basename(specPath),
        reason: options.reason,
        executor,
      });
      for (const warning of (prepared.preview.warnings as string[] | undefined) ?? []) printPipelineWarning(warning);
      const started = await executeBackgroundRunStart({
        workspaceRoot,
        input: prepared.arguments,
        executor,
        requestedBy: currentRequester("cli"),
      });
      if (isJsonMode()) return printJson(started);
      const id = String(started.background_run_id);
      printSuccess(`Started background run ${shortId(id)} (${prepared.engine}, ${String(prepared.preview.target)}).`);
      console.log(`Status: tt pipeline status ${shortId(id)}   Logs: tt pipeline logs ${shortId(id)}   Cancel: tt pipeline cancel ${shortId(id)} --reason "..."`);
    });

  pipeline.command("list")
    .description("List background runs on this machine")
    .option("--active", "Only runs that are starting, running or stopping")
    .action(async (options: { active?: boolean }) => {
      const runs = (await store().list()).filter((run) => !options.active || isActiveBackgroundRun(run));
      if (isJsonMode()) return printJson(runs.map(describeBackgroundRun));
      if (runs.length === 0) return printWarning("No background runs found.");
      printTable(
        ["ID", "Status", "Engine", "Target", "Requested by", "Reason", "Started", "Finished"],
        runs.map((run) => [
          shortId(run.id),
          formatStatus(run.status),
          run.choices.engine,
          run.choices.target,
          `${run.requested_by.user} (${run.requested_by.via})`,
          truncate(run.reason, 40),
          formatDate(run.started_at),
          formatDate(run.finished_at),
        ]),
      );
    });

  pipeline.command("status")
    .description("Show a background run's record, audit-chain check and recent log lines")
    .argument("<id>", "Background run ID or prefix")
    .option("--lines <n>", "Log lines to show", "20")
    .action(async (id: string, options: { lines: string }) => {
      const runs = store();
      const run = await runs.get(id);
      const [{ events, chain }, log] = await Promise.all([runs.events(run.id), runs.logTail(run.id, Number(options.lines) || 20)]);
      if (isJsonMode()) return printJson({ ...run, audit: { chain, events }, log_tail: log });
      printDetail([
        ["ID", run.id],
        ["Status", formatStatus(run.status)],
        ["Reason", run.reason],
        ["Requested by", `${run.requested_by.user}@${run.requested_by.host} via ${run.requested_by.via}`],
        ["Approved", formatDate(run.approved_at)],
        ["Spec", `${run.choices.spec_path} (sha256 ${run.choices.spec_sha256.slice(0, 12)})`],
        ["Engine", run.choices.engine],
        ["Steps", run.choices.steps.map((step) => step.id).join(" → ")],
        ["Target", run.choices.target],
        ["Started", formatDate(run.started_at)],
        ["Finished", run.finished_at ? formatDate(run.finished_at) : undefined],
        ["Cancel reason", run.cancel?.reason],
        ["Model", run.outcome?.model_id],
        ["Report", run.outcome?.report_path],
        ["Error", run.outcome?.error],
        ["Audit chain", `${chain.status} (${chain.events} events)`],
        ["Record", runs.paths(run.id).record],
      ]);
      if (run.outcome?.metrics) console.log(`\nMetrics:\n${JSON.stringify(run.outcome.metrics, null, 2)}`);
      if (log) console.log(`\nRecent log:\n${log}`);
    });

  pipeline.command("logs")
    .description("Print a background run's log")
    .argument("<id>", "Background run ID or prefix")
    .option("--lines <n>", "Lines to show", "200")
    .action(async (id: string, options: { lines: string }) => {
      const log = await store().logTail(id, Number(options.lines) || 200);
      if (isJsonMode()) return printJson({ log });
      console.log(log);
    });

  pipeline.command("cancel")
    .description("Stop an active background run")
    .argument("<id>", "Background run ID or prefix")
    .requiredOption("--reason <text>", "Why the run is being stopped (kept in its audit record)")
    .action(async (id: string, options: { reason: string }) => {
      const runs = store();
      const run = await runs.get(id);
      const result = await executeBackgroundRunCancel({
        input: { background_run_id: run.id, reason: options.reason },
        store: runs,
        requestedBy: currentRequester("cli"),
      });
      if (isJsonMode()) return printJson(result);
      printSuccess(`Cancellation requested for ${shortId(run.id)}. It records cancelled once its process exits.`);
    });

  pipeline.command("supervise", { hidden: true })
    .description("Internal: own one background run's pipeline process")
    .argument("<id>", "Background run ID")
    .action(async (id: string) => {
      await superviseBackgroundRun({ store: store(), id, entrypoint: process.argv[1]! });
    });
}

export { DEFAULT_PIPELINE_FILE };
