import { Command } from "commander";
import ora from "ora";
import { get, post, type ClientOpts } from "../client.js";
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
  hyperparameters: Record<string, unknown> | null;
  eval_summary: { mean_score?: number; pass_rate?: number } | null;
  error: string | null;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
  updated_at: string;
  _spec_name?: string;
  _evals?: RunEval[];
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

export function registerRunsCommands(parent: Command) {
  const runs = parent.command("runs").description("Manage runs");

  runs
    .command("list")
    .description("List runs")
    .option("-s, --spec <id>", "Filter by spec ID")
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
        path = `/behavior-specs/${cmdOpts.spec}/runs`;
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
    .argument("<id>", "Run ID")
    .action(async (id: string) => {
      const opts = parent.opts() as ClientOpts;
      const { data } = await get<Run>(`/runs/${id}`, undefined, opts);

      if (isJsonMode()) return printJson(data);

      printDetail([
        ["ID", data.id],
        ["Spec", data._spec_name ?? shortId(data.behavior_spec_id)],
        ["Run #", String(data.run_number)],
        ["Status", formatStatus(data.status)],
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
    .argument("<spec-id>", "Behaviour spec ID")
    .option("--no-augment", "Disable data augmentation")
    .option("--epochs <n>", "Number of training epochs")
    .option("--lr <rate>", "Learning rate")
    .option("--batch-size <n>", "Batch size")
    .option("--lora-rank <n>", "LoRA rank")
    .option("--lora-alpha <n>", "LoRA alpha")
    .action(async (specId: string, cmdOpts) => {
      const opts = parent.opts() as ClientOpts;
      const body: Record<string, unknown> = {};

      if (cmdOpts.augment === false) body.augment = false;

      const hp: Record<string, unknown> = {};
      if (cmdOpts.epochs) hp.n_epochs = Number(cmdOpts.epochs);
      if (cmdOpts.lr) hp.learning_rate = Number(cmdOpts.lr);
      if (cmdOpts.batchSize) hp.batch_size = Number(cmdOpts.batchSize);
      if (cmdOpts.loraRank) hp.lora_rank = Number(cmdOpts.loraRank);
      if (cmdOpts.loraAlpha) hp.lora_alpha = Number(cmdOpts.loraAlpha);
      if (Object.keys(hp).length) body.hyperparameters = hp;

      const { data } = await post<Run>(
        `/behavior-specs/${specId}/runs`,
        body,
        opts,
      );

      if (isJsonMode()) return printJson(data);
      printSuccess(
        `Run #${data.run_number} started (${shortId(data.id)}). Status: ${formatStatus(data.status)}`,
      );
    });

  runs
    .command("cancel")
    .description("Cancel a running run")
    .argument("<id>", "Run ID")
    .action(async (id: string) => {
      const opts = parent.opts() as ClientOpts;
      const { data } = await post<Run>(`/runs/${id}/cancel`, undefined, opts);

      if (isJsonMode()) return printJson(data);
      printSuccess(`Run ${shortId(id)} cancelled.`);
    });

  runs
    .command("watch")
    .description("Watch a run until it completes")
    .argument("<id>", "Run ID")
    .option("--interval <ms>", "Poll interval in milliseconds", "5000")
    .action(async (id: string, cmdOpts) => {
      const opts = parent.opts() as ClientOpts;
      const interval = Number(cmdOpts.interval);
      const spinner = ora(`Watching run ${shortId(id)}...`).start();

      const terminalStates = new Set([
        "completed",
        "failed",
        "cancelled",
      ]);

      let run: Run;
      while (true) {
        const { data } = await get<Run>(`/runs/${id}`, undefined, opts);
        run = data;
        spinner.text = `Run ${shortId(id)} — ${formatStatus(run.status)}`;

        if (terminalStates.has(run.status)) break;
        await new Promise((r) => setTimeout(r, interval));
      }

      spinner.stop();

      if (isJsonMode()) return printJson(run);

      printDetail([
        ["Run", shortId(run.id)],
        ["Status", formatStatus(run.status)],
        ["Score", run.eval_summary?.mean_score != null
          ? (run.eval_summary.mean_score * 100).toFixed(1) + "%"
          : undefined],
        ["Error", run.error ?? undefined],
        ["Completed", formatDate(run.completed_at)],
      ]);
    });
}
