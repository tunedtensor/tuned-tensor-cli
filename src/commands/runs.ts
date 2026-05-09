import { Command } from "commander";
import ora from "ora";
import { get, post, type ClientOpts } from "../client.js";
import { resolveSpecId, resolveRunId, resolveDatasetId } from "../resolve.js";
import {
  printTable,
  printDetail,
  printSuccess,
  printJson,
  isJsonMode,
  formatDate,
  formatStatus,
  shortId,
  truncate,
} from "../output.js";

interface Run {
  id: string;
  behavior_spec_id: string;
  run_number: number;
  status: string;
  current_stage?: string | null;
  stage_label?: string | null;
  progress_pct?: number | null;
  status_message?: string | null;
  hyperparameters: Record<string, unknown> | null;
  eval_summary: { mean_score?: number; pass_rate?: number } | null;
  error: string | null;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
  updated_at: string;
  _spec_name?: string;
  _evals?: RunEval[];
  _events?: RunEvent[];
}

interface RunEval {
  id: string;
  prompt: string;
  expected: string;
  actual: string;
  passed: boolean;
  score: number | null;
  reasoning: string | null;
  latency_ms: number | null;
}

interface RunEvent {
  id: string;
  stage: string;
  label: string;
  status: string;
  message: string | null;
  occurred_at: string;
}

function formatProgress(run: Run): string | undefined {
  if (run.progress_pct == null && !run.stage_label) return undefined;
  const label = run.stage_label ?? run.current_stage ?? "Progress";
  return run.progress_pct == null ? label : `${label} (${run.progress_pct}%)`;
}

export function registerRunsCommands(parent: Command) {
  const runs = parent.command("runs").description("Manage runs");

  runs
    .command("list")
    .description("List runs")
    .option("-s, --spec <id>", "Filter by spec ID (full UUID or 8+ char prefix)")
    .option("-p, --page <n>", "Page number", "1")
    .option("--per-page <n>", "Results per page", "20")
    .action(async (cmdOpts) => {
      const opts = parent.opts() as ClientOpts;

      let path = "/runs";
      const query: Record<string, string | number> = {
        page: cmdOpts.page,
        per_page: cmdOpts.perPage,
      };

      if (cmdOpts.spec) {
        const fullSpecId = await resolveSpecId(cmdOpts.spec, opts);
        path = `/behavior-specs/${fullSpecId}/runs`;
      }

      const { data, meta } = await get<Run[]>(path, query, opts);

      if (isJsonMode()) return printJson({ data, meta });

      printTable(
        ["ID", "Spec", "#", "Status", "Score", "Started", "Completed"],
        data.map((r) => [
          shortId(r.id),
          r._spec_name ? truncate(r._spec_name, 20) : shortId(r.behavior_spec_id),
          String(r.run_number),
          formatStatus(r.status),
          r.eval_summary?.mean_score != null
            ? (r.eval_summary.mean_score * 100).toFixed(1) + "%"
            : "—",
          formatDate(r.started_at),
          formatDate(r.completed_at),
        ]),
        meta,
      );
    });

  runs
    .command("get")
    .description("Show run details and eval results")
    .argument("<id>", "Run ID (full UUID or 8+ char prefix)")
    .action(async (id: string) => {
      const opts = parent.opts() as ClientOpts;
      const fullId = await resolveRunId(id, opts);
      const { data } = await get<Run>(`/runs/${fullId}`, undefined, opts);

      if (isJsonMode()) return printJson(data);

      printDetail([
        ["ID", data.id],
        ["Spec", data._spec_name ?? shortId(data.behavior_spec_id)],
        ["Run #", String(data.run_number)],
        ["Status", formatStatus(data.status)],
        ["Stage", formatProgress(data)],
        ["Message", data.status_message ?? undefined],
        ["Score", data.eval_summary?.mean_score != null
          ? (data.eval_summary.mean_score * 100).toFixed(1) + "%"
          : undefined],
        ["Pass Rate", data.eval_summary?.pass_rate != null
          ? (data.eval_summary.pass_rate * 100).toFixed(1) + "%"
          : undefined],
        ["Error", data.error ?? undefined],
        ["Started", formatDate(data.started_at)],
        ["Completed", formatDate(data.completed_at)],
        ["Created", formatDate(data.created_at)],
      ]);

      if (data.hyperparameters && Object.keys(data.hyperparameters).length) {
        console.log("\nHyperparameters:");
        for (const [k, v] of Object.entries(data.hyperparameters)) {
          console.log(`  ${k}: ${v}`);
        }
      }

      if (data._events?.length) {
        console.log("\nLatest Updates:");
        for (const event of data._events.slice(-5)) {
          const time = formatDate(event.occurred_at);
          console.log(
            `  ${time}  ${formatStatus(event.status)} ${event.label}${event.message ? ` — ${event.message}` : ""}`,
          );
        }
      }

      if (data._evals?.length) {
        console.log(`\nEval Results (${data._evals.length}):`);
        printTable(
          ["#", "Passed", "Score", "Prompt", "Latency"],
          data._evals.map((e, i) => [
            String(i + 1),
            e.passed ? "✓" : "✗",
            e.score != null ? e.score.toFixed(2) : "—",
            truncate(e.prompt, 40),
            e.latency_ms != null ? `${e.latency_ms}ms` : "—",
          ]),
        );
      }
    });

  runs
    .command("start")
    .description("Start a new run for a behaviour spec")
    .argument("<spec-id>", "Behaviour spec ID (full UUID or 8+ char prefix)")
    .option("--no-augment", "Disable data augmentation")
    .option("--no-llm-judge", "Disable Bedrock LLM judging")
    .option("--epochs <n>", "Number of training epochs")
    .option("--lr <rate>", "Learning rate")
    .option("--batch-size <n>", "Batch size")
    .option("--dataset <id>", "Dataset ID to use instead of inline spec examples (full UUID or 4+ char prefix)")
    .option("--train-ratio <ratio>", "Dataset training split ratio (default: 0.8 when any split ratio is set)")
    .option("--validation-ratio <ratio>", "Dataset validation split ratio (default: 0.1 when any split ratio is set)")
    .option("--test-ratio <ratio>", "Dataset test split ratio (default: 0.1 when any split ratio is set)")
    .option("--lora-rank <n>", "LoRA rank")
    .option("--lora-alpha <n>", "LoRA alpha")
    .action(async (specId: string, cmdOpts) => {
      const opts = parent.opts() as ClientOpts;
      const body: Record<string, unknown> = {};

      if (cmdOpts.augment === false) body.augment = false;
      if (cmdOpts.dataset) body.dataset_id = await resolveDatasetId(cmdOpts.dataset, opts);

      const splitRatioOptions = [
        cmdOpts.trainRatio,
        cmdOpts.validationRatio,
        cmdOpts.testRatio,
      ];
      if (splitRatioOptions.some((value) => value != null)) {
        body.split_ratios = {
          train: cmdOpts.trainRatio != null ? Number(cmdOpts.trainRatio) : 0.8,
          validation:
            cmdOpts.validationRatio != null ? Number(cmdOpts.validationRatio) : 0.1,
          test: cmdOpts.testRatio != null ? Number(cmdOpts.testRatio) : 0.1,
        };
      }

      const hp: Record<string, unknown> = {};
      if (cmdOpts.epochs) hp.n_epochs = Number(cmdOpts.epochs);
      if (cmdOpts.lr) hp.learning_rate = Number(cmdOpts.lr);
      if (cmdOpts.batchSize) hp.batch_size = Number(cmdOpts.batchSize);
      if (cmdOpts.loraRank) hp.lora_rank = Number(cmdOpts.loraRank);
      if (cmdOpts.loraAlpha) hp.lora_alpha = Number(cmdOpts.loraAlpha);
      if (cmdOpts.llmJudge === false) body.use_llm_judge = false;
      if (Object.keys(hp).length) body.hyperparameters = hp;

      const fullSpecId = await resolveSpecId(specId, opts);
      const { data } = await post<Run>(
        `/behavior-specs/${fullSpecId}/runs`,
        body,
        opts,
      );

      if (isJsonMode()) return printJson(data);
      printSuccess(
        `Run #${data.run_number} started (${shortId(data.id)}). ${formatProgress(data) ?? `Status: ${formatStatus(data.status)}`}`,
      );
    });

  runs
    .command("cancel")
    .description("Cancel a running run")
    .argument("<id>", "Run ID (full UUID or 8+ char prefix)")
    .action(async (id: string) => {
      const opts = parent.opts() as ClientOpts;
      const fullId = await resolveRunId(id, opts);
      const { data } = await post<Run>(`/runs/${fullId}/cancel`, undefined, opts);

      if (isJsonMode()) return printJson(data);
      printSuccess(`Run ${shortId(fullId)} cancelled.`);
    });

  runs
    .command("watch")
    .description("Watch a run until it completes")
    .argument("<id>", "Run ID (full UUID or 8+ char prefix)")
    .option("--interval <ms>", "Poll interval in milliseconds", "5000")
    .action(async (id: string, cmdOpts) => {
      const opts = parent.opts() as ClientOpts;
      const interval = Number(cmdOpts.interval);
      const fullId = await resolveRunId(id, opts);
      const spinner = ora(`Watching run ${shortId(fullId)}...`).start();

      const terminalStates = new Set([
        "completed",
        "failed",
        "cancelled",
      ]);

      let run: Run;
      while (true) {
        const { data } = await get<Run>(`/runs/${fullId}`, undefined, opts);
        run = data;
        spinner.text = `Run ${shortId(fullId)} — ${formatProgress(run) ?? formatStatus(run.status)}`;

        if (terminalStates.has(run.status)) break;
        await new Promise((r) => setTimeout(r, interval));
      }

      spinner.stop();

      if (isJsonMode()) return printJson(run);

      printDetail([
        ["Run", shortId(run.id)],
        ["Status", formatStatus(run.status)],
        ["Stage", formatProgress(run)],
        ["Message", run.status_message ?? undefined],
        ["Score", run.eval_summary?.mean_score != null
          ? (run.eval_summary.mean_score * 100).toFixed(1) + "%"
          : undefined],
        ["Error", run.error ?? undefined],
        ["Completed", formatDate(run.completed_at)],
      ]);
    });
}
