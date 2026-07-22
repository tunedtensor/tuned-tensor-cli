import { Command } from "commander";
import ora from "ora";
import { get, post, type ClientOpts } from "../client.js";
import { resolveSpecId, resolveRunId, resolveDatasetId, resolveModelId } from "../resolve.js";
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
  eval_summary: RunEvalSummary | null;
  error: string | null;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
  updated_at: string;
  _spec_name?: string;
  _evals?: RunEval[];
  _events?: RunEvent[];
}

interface RunEvalSummary {
  avg_score?: number;
  mean_score?: number;
  pass_rate?: number;
  output_diagnostics?: RunOutputDiagnostics;
}

interface EvalOutputDiagnostics {
  total: number;
  avg_output_chars?: number;
  max_output_chars?: number;
  starts_json_count: number;
  starts_json_rate: number;
  valid_json_count: number;
  valid_json_rate: number;
  strict_json_count: number;
  strict_json_rate: number;
  expected_schema_keys_count: number;
  expected_schema_keys_rate: number;
  non_json_prefix_count: number;
  non_json_prefix_rate: number;
  visible_reasoning_prefix_count: number;
  visible_reasoning_prefix_rate: number;
}

interface RunOutputDiagnostics {
  baseline: EvalOutputDiagnostics;
  candidate: EvalOutputDiagnostics;
  test?: {
    baseline: EvalOutputDiagnostics;
    candidate: EvalOutputDiagnostics;
  };
  insights?: string[];
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

interface RunDiagnostics {
  run_id: string;
  status: string;
  stage: string | null;
  stage_label: string | null;
  progress_pct: number | null;
  status_message: string | null;
  summary: string;
  insights: string[];
  output_diagnostics?: RunOutputDiagnostics;
  training: {
    state: string;
    phase: string | null;
    started_at: string | null;
    completed_at: string | null;
    last_updated_at: string | null;
    curve: {
      target_epochs: number | null;
      latest_epoch: number | null;
      latest_loss: number | null;
      previous_loss: number | null;
      latest_token_accuracy: number | null;
      epoch_rate_per_minute: number | null;
      estimated_minutes_remaining: number | null;
      latest_log_at: string | null;
      samples: {
        timestamp: string;
        epoch: number;
        loss?: number;
        learning_rate?: number;
        token_accuracy?: number;
      }[];
    };
  };
  generated_at: string;
}

interface RunReportResult {
  prompt: string;
  expected?: string | null;
  actual?: string | null;
  passed?: boolean | null;
  score?: number | null;
  reasoning?: string | null;
}

interface RunReportEval {
  total?: number;
  eval_examples_used?: number;
  avg_score?: number;
  pass_rate?: number;
  exact_match_rate?: number;
  results?: RunReportResult[];
}

interface RunReportComparisonExample {
  prompt: string;
  old_score?: number | null;
  new_score?: number | null;
}

interface RunReportComparison {
  avg_score_delta?: number;
  pass_rate_delta?: number;
  exact_match_rate_delta?: number;
  regressions?: number;
  improvements?: number;
  regressed_examples?: RunReportComparisonExample[];
}

interface RunReportSplit {
  baseline?: RunReportEval;
  candidate?: RunReportEval;
  comparison?: RunReportComparison;
}

interface RunReport extends RunReportSplit {
  run_id?: string;
  status?: string;
  fine_tuned_model_id?: string | null;
  test?: RunReportSplit;
}

interface RunEstimate {
  estimated_training_tokens: number;
  estimated_cost_cents: number;
  estimated_epochs: number;
  billing?: {
    plan: string;
    free_run_eligible: boolean;
    free_run_ineligibility: string[];
    free_runs_used: number;
    free_runs_remaining: number;
    free_runs_monthly_limit: number;
    billing_source: "free_quota" | "credits";
  };
  duration: {
    estimated_minutes: number;
    range_minutes: {
      low: number;
      high: number;
    };
    confidence: "low" | "medium" | "high";
    sample_count: number;
    basis: "matched_model" | "all_completed_runs" | "fallback";
  };
}

const LONG_EXAMPLE_POLICIES = ["error", "truncate", "skip"] as const;

function parseLongExamplesPolicy(value: string): (typeof LONG_EXAMPLE_POLICIES)[number] {
  if ((LONG_EXAMPLE_POLICIES as readonly string[]).includes(value)) {
    return value as (typeof LONG_EXAMPLE_POLICIES)[number];
  }
  throw new Error(
    `--long-examples must be one of: ${LONG_EXAMPLE_POLICIES.join(", ")}`,
  );
}

function formatProgress(run: Run): string | undefined {
  if (run.progress_pct == null && !run.stage_label) return undefined;
  const label = run.stage_label ?? run.current_stage ?? "Progress";
  return run.progress_pct == null ? label : `${label} (${run.progress_pct}%)`;
}

function getEvalScore(run: Run): number | undefined {
  return run.eval_summary?.avg_score ?? run.eval_summary?.mean_score;
}

function formatEvalScore(run: Run): string | undefined {
  const score = getEvalScore(run);
  return score == null ? undefined : (score * 100).toFixed(1) + "%";
}

function formatPercent(rate: number | undefined): string | undefined {
  return rate == null ? undefined : (rate * 100).toFixed(1) + "%";
}

function formatPointDelta(delta: number | undefined): string | undefined {
  if (delta == null) return undefined;
  const points = delta * 100;
  return `${points >= 0 ? "+" : ""}${points.toFixed(1)} pp`;
}

function formatScore(score: number | null | undefined): string {
  return score == null ? "—" : score.toFixed(2);
}

function formatCountRate(count: number | undefined, total: number | undefined, rate: number | undefined): string | undefined {
  if (count == null || total == null || rate == null) return undefined;
  return `${formatPercent(rate)} (${count}/${total})`;
}

function formatMinutes(minutes?: number | null): string | undefined {
  if (minutes == null) return undefined;
  if (minutes < 60) return `${Math.max(1, Math.round(minutes))}m`;
  return `${(minutes / 60).toFixed(1)}h`;
}

function formatCents(cents: number): string {
  const dollars = cents / 100;
  if (Math.abs(dollars) >= 100) return `$${dollars.toFixed(0)}`;
  return `$${dollars.toFixed(2)}`;
}

function formatEstimateRange(estimate: RunEstimate): string {
  const duration = estimate.duration;
  return `${formatMinutes(duration.estimated_minutes)} (${formatMinutes(duration.range_minutes.low)}-${formatMinutes(duration.range_minutes.high)})`;
}

function printRunEstimate(estimate: RunEstimate) {
  const billing = estimate.billing;
  printDetail([
    ["Estimated Time", formatEstimateRange(estimate)],
    ["Confidence", estimate.duration.confidence],
    ["History Samples", String(estimate.duration.sample_count)],
    ["Basis", estimate.duration.basis.replaceAll("_", " ")],
    ["Estimated Cost", formatCents(estimate.estimated_cost_cents)],
    ["Training Tokens", `${(estimate.estimated_training_tokens / 1000).toFixed(1)}k`],
    ["Epochs", String(estimate.estimated_epochs)],
    ["Plan", billing?.plan],
    [
      "Billing Source",
      billing?.billing_source === "free_quota"
        ? "Free monthly quota"
        : billing?.billing_source === "credits"
          ? "Credits"
          : undefined,
    ],
    [
      "Free Runs",
      billing
        ? `${billing.free_runs_remaining}/${billing.free_runs_monthly_limit} remaining`
        : undefined,
    ],
    [
      "Free Eligible",
      billing
        ? billing.free_run_eligible
          ? "yes"
          : `no (${billing.free_run_ineligibility.join(", ") || "not eligible"})`
        : undefined,
    ],
  ]);

  console.log(
    "\nDuration is a rough historical range. Final cost uses provider-reported training tokens.",
  );
}

function formatEpochProgress(diagnostics: RunDiagnostics): string | undefined {
  const curve = diagnostics.training.curve;
  if (curve.latest_epoch == null) return undefined;
  if (curve.target_epochs == null) return curve.latest_epoch.toFixed(4);
  return `${curve.latest_epoch.toFixed(4)} / ${curve.target_epochs.toFixed(2)}`;
}

function printDiagnostics(diagnostics: RunDiagnostics) {
  const curve = diagnostics.training.curve;
  const perFive = curve.epoch_rate_per_minute == null
    ? undefined
    : (curve.epoch_rate_per_minute * 5).toFixed(4);

  printDetail([
    ["Run", shortId(diagnostics.run_id)],
    ["Status", formatStatus(diagnostics.status)],
    ["Stage", diagnostics.stage_label ?? diagnostics.stage ?? undefined],
    ["Summary", diagnostics.summary],
    ["Training", diagnostics.training.state],
    ["Phase", diagnostics.training.phase ?? undefined],
    ["Epoch", formatEpochProgress(diagnostics)],
    ["Latest Loss", curve.latest_loss == null ? undefined : curve.latest_loss.toFixed(4)],
    [
      "Token Accuracy",
      curve.latest_token_accuracy == null
        ? undefined
        : (curve.latest_token_accuracy * 100).toFixed(1) + "%",
    ],
    ["Pace", perFive ? `${perFive} epoch / 5m` : undefined],
    ["ETA", formatMinutes(curve.estimated_minutes_remaining)],
    ["Latest Update", formatDate(diagnostics.training.last_updated_at)],
  ]);

  if (diagnostics.insights.length) {
    console.log("\nInsights:");
    for (const insight of diagnostics.insights) {
      console.log(`  - ${insight}`);
    }
  }

  printEvalOutputDiagnostics(diagnostics.output_diagnostics);
}

function printEvalOutputDiagnostics(diagnostics: RunOutputDiagnostics | undefined) {
  if (!diagnostics) return;
  const candidate = diagnostics.candidate;
  const baseline = diagnostics.baseline;

  console.log("\nOutput Diagnostics:");
  printDetail([
    ["Tuned Valid JSON", formatCountRate(candidate.valid_json_count, candidate.total, candidate.valid_json_rate)],
    ["Tuned Strict JSON", formatCountRate(candidate.strict_json_count, candidate.total, candidate.strict_json_rate)],
    [
      "Tuned Schema Keys",
      formatCountRate(candidate.expected_schema_keys_count, candidate.total, candidate.expected_schema_keys_rate),
    ],
    [
      "Tuned Non-JSON Prefix",
      formatCountRate(candidate.non_json_prefix_count, candidate.total, candidate.non_json_prefix_rate),
    ],
    [
      "Tuned Reasoning Prefix",
      formatCountRate(
        candidate.visible_reasoning_prefix_count,
        candidate.total,
        candidate.visible_reasoning_prefix_rate,
      ),
    ],
    ["Tuned Avg Output", candidate.avg_output_chars == null ? undefined : `${candidate.avg_output_chars} chars`],
    ["Base Valid JSON", formatCountRate(baseline.valid_json_count, baseline.total, baseline.valid_json_rate)],
    [
      "Base Reasoning Prefix",
      formatCountRate(
        baseline.visible_reasoning_prefix_count,
        baseline.total,
        baseline.visible_reasoning_prefix_rate,
      ),
    ],
  ]);

  if (diagnostics.test) {
    console.log("\nTest Output Diagnostics:");
    printDetail([
      [
        "Tuned Valid JSON",
        formatCountRate(
          diagnostics.test.candidate.valid_json_count,
          diagnostics.test.candidate.total,
          diagnostics.test.candidate.valid_json_rate,
        ),
      ],
      [
        "Tuned Strict JSON",
        formatCountRate(
          diagnostics.test.candidate.strict_json_count,
          diagnostics.test.candidate.total,
          diagnostics.test.candidate.strict_json_rate,
        ),
      ],
      [
        "Tuned Reasoning Prefix",
        formatCountRate(
          diagnostics.test.candidate.visible_reasoning_prefix_count,
          diagnostics.test.candidate.total,
          diagnostics.test.candidate.visible_reasoning_prefix_rate,
        ),
      ],
    ]);
  }

  if (diagnostics.insights?.length) {
    console.log("\nEvaluation Insights:");
    for (const insight of diagnostics.insights) {
      console.log(`  - ${insight}`);
    }
  }
}

function compactWhitespace(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function parseJsonish(value: string | null | undefined): unknown {
  if (!value) return value;
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

function formatJsonish(value: string | null | undefined, max = 360): string {
  const parsed = parseJsonish(value);
  const rendered = typeof parsed === "string"
    ? compactWhitespace(parsed)
    : JSON.stringify(parsed);
  return truncate(rendered ?? "—", max);
}

function formatPromptSummary(prompt: string, max = 260): string {
  const subject = prompt.match(/^Subject:\s*(.+)$/m)?.[1]?.trim();
  const body = prompt.match(/^Body:\s*([\s\S]+)$/m)?.[1];
  if (subject) {
    const bodySummary = body ? ` | Body: ${compactWhitespace(body)}` : "";
    return truncate(`Subject: ${subject}${bodySummary}`, max);
  }
  return truncate(compactWhitespace(prompt), max);
}

function indexResults(results: RunReportResult[] | undefined): Map<string, RunReportResult> {
  return new Map((results ?? []).map((result) => [result.prompt, result]));
}

function printReportMetrics(label: string, split: RunReportSplit | undefined) {
  if (!split) return;
  const comparison = split.comparison;
  const baseline = split.baseline;
  const candidate = split.candidate;

  console.log(`\n${label} Metrics:`);
  printDetail([
    ["Base Avg", formatPercent(baseline?.avg_score)],
    ["Tuned Avg", formatPercent(candidate?.avg_score)],
    ["Avg Delta", formatPointDelta(comparison?.avg_score_delta)],
    ["Base Pass", formatPercent(baseline?.pass_rate)],
    ["Tuned Pass", formatPercent(candidate?.pass_rate)],
    ["Pass Delta", formatPointDelta(comparison?.pass_rate_delta)],
    ["Regressions", comparison?.regressions == null ? undefined : String(comparison.regressions)],
    ["Improvements", comparison?.improvements == null ? undefined : String(comparison.improvements)],
    ["Eval Rows", candidate?.eval_examples_used == null
      ? candidate?.total == null ? undefined : String(candidate.total)
      : String(candidate.eval_examples_used)],
  ]);
}

function printComparedExample(input: {
  index: number;
  prompt: string;
  baseline?: RunReportResult;
  candidate?: RunReportResult;
  oldScore?: number | null;
  newScore?: number | null;
}) {
  const { index, prompt, baseline, candidate, oldScore, newScore } = input;
  const expected = candidate?.expected ?? baseline?.expected;
  const scoreLine = oldScore != null || newScore != null
    ? oldScore == null
      ? `tuned score ${formatScore(newScore)}`
      : `score ${formatScore(oldScore)} -> ${formatScore(newScore)}`
    : `tuned score ${formatScore(candidate?.score)}`;

  console.log(`\n${index}. ${scoreLine}`);
  console.log(`   Prompt: ${formatPromptSummary(prompt)}`);
  console.log(`   Expected: ${formatJsonish(expected)}`);
  if (baseline?.actual != null) {
    console.log(`   Base: ${formatJsonish(baseline.actual)}`);
  }
  if (candidate?.actual != null) {
    console.log(`   Tuned: ${formatJsonish(candidate.actual)}`);
  }
  const reasoning = candidate?.reasoning ?? baseline?.reasoning;
  if (reasoning) {
    console.log(`   Judge: ${truncate(compactWhitespace(reasoning), 420)}`);
  }
}

function printReportExamples(
  label: string,
  split: RunReportSplit | undefined,
  mode: "regressions" | "failures",
  limit: number,
) {
  if (!split || limit <= 0) return;
  const baselineByPrompt = indexResults(split.baseline?.results);
  const candidateByPrompt = indexResults(split.candidate?.results);

  if (mode === "regressions") {
    const regressions = split.comparison?.regressed_examples ?? [];
    console.log(`\n${label} Regressions:`);
    if (!regressions.length) {
      console.log("  No regressed examples in the report.");
      return;
    }
    regressions.slice(0, limit).forEach((example, i) => {
      printComparedExample({
        index: i + 1,
        prompt: example.prompt,
        baseline: baselineByPrompt.get(example.prompt),
        candidate: candidateByPrompt.get(example.prompt),
        oldScore: example.old_score,
        newScore: example.new_score,
      });
    });
    return;
  }

  const failures = (split.candidate?.results ?? [])
    .filter((result) => result.passed === false || (result.score ?? 1) < 1)
    .sort((a, b) => (a.score ?? 0) - (b.score ?? 0));

  console.log(`\n${label} Tuned Failures:`);
  if (!failures.length) {
    console.log("  No tuned failures in the report.");
    return;
  }
  failures.slice(0, limit).forEach((candidate, i) => {
    printComparedExample({
      index: i + 1,
      prompt: candidate.prompt,
      baseline: baselineByPrompt.get(candidate.prompt),
      candidate,
      newScore: candidate.score,
    });
  });
}

function printRunReport(
  report: RunReport,
  options: { split: "primary" | "test" | "all"; limit: number; mode: "regressions" | "failures" },
) {
  printDetail([
    ["Run", report.run_id ? shortId(report.run_id) : undefined],
    ["Status", report.status ? formatStatus(report.status) : undefined],
    ["Model", report.fine_tuned_model_id ? shortId(report.fine_tuned_model_id) : undefined],
  ]);

  const splits: Array<[string, RunReportSplit | undefined]> = [];
  if (options.split === "primary" || options.split === "all") {
    splits.push(["Primary", report]);
  }
  if (options.split === "test" || options.split === "all") {
    splits.push(["Test", report.test]);
  }

  for (const [label, split] of splits) {
    printReportMetrics(label, split);
    printReportExamples(label, split, options.mode, options.limit);
  }
}

function addRunConfigurationOptions(command: Command): Command {
  return command
    .option("--no-augment", "Disable data augmentation")
    .option("--no-llm-judge", "Disable LLM judging")
    .option("--epochs <n>", "Number of training epochs")
    .option("--lr <rate>", "Learning rate")
    .option("--batch-size <n>", "Batch size")
    .option("--dataset <id>", "Dataset ID to use instead of inline spec examples (full UUID or 4+ char prefix)")
    .option("--parent-model <id>", "Fine-tuned model ID to continue training from (full UUID or 4+ char prefix)")
    .option("--train-ratio <ratio>", "Dataset training split ratio (default: 0.8 when any split ratio is set)")
    .option("--validation-ratio <ratio>", "Dataset validation split ratio (default: 0.1 when any split ratio is set)")
    .option("--test-ratio <ratio>", "Dataset test split ratio (default: 0.1 when any split ratio is set)")
    .option("--lora-rank <n>", "LoRA rank")
    .option("--lora-alpha <n>", "LoRA alpha")
    .option("--max-eval-examples <n>", "Max examples for the primary eval pass")
    .option("--max-test-eval-examples <n>", "Max examples for the secondary test eval pass")
    .option("--long-examples <policy>", "Long training row policy: error, truncate, or skip")
    .option("--max-seq-length <tokens>", "Maximum training sequence length in tokens")
    .option("--max-output-tokens <tokens>", "Desired evaluation output budget in tokens")
    .option("--eval-reserved-output-tokens <tokens>", "Minimum evaluation output tokens reserved per row");
}

async function buildRunRequestBody(
  cmdOpts: Record<string, unknown>,
  opts: ClientOpts,
): Promise<Record<string, unknown>> {
  const body: Record<string, unknown> = {};

  if (cmdOpts.augment === false) body.augment = false;
  if (cmdOpts.dataset) body.dataset_id = await resolveDatasetId(String(cmdOpts.dataset), opts);
  if (cmdOpts.parentModel) {
    body.parent_model_id = await resolveModelId(String(cmdOpts.parentModel), opts);
  }

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
  if (cmdOpts.maxEvalExamples) hp.max_eval_examples = Number(cmdOpts.maxEvalExamples);
  if (cmdOpts.maxTestEvalExamples) hp.max_test_eval_examples = Number(cmdOpts.maxTestEvalExamples);
  if (cmdOpts.longExamples) hp.long_examples = parseLongExamplesPolicy(String(cmdOpts.longExamples));
  if (cmdOpts.maxSeqLength) hp.max_seq_length = Number(cmdOpts.maxSeqLength);
  if (cmdOpts.maxOutputTokens) hp.max_output_tokens = Number(cmdOpts.maxOutputTokens);
  if (cmdOpts.evalReservedOutputTokens) {
    hp.eval_reserved_output_tokens = Number(cmdOpts.evalReservedOutputTokens);
  }
  if (cmdOpts.llmJudge === false) body.use_llm_judge = false;
  if (Object.keys(hp).length) body.hyperparameters = hp;

  return body;
}

export function registerRunsCommands(parent: Command) {
  const runs = parent.command("runs").description("Manage runs");

  runs
    .command("list")
    .description("List runs")
    .option("-s, --spec <id>", "Filter by spec ID (full UUID or 8+ char prefix)")
    .option("-p, --page <n>", "Page number", "1")
    .option("--per-page <n>", "Results per page", "20")
    .option("--summary", "Request compact run summaries without detailed eval payloads or events")
    .action(async (cmdOpts) => {
      const opts = parent.opts() as ClientOpts;

      let path = "/runs";
      const query: Record<string, string | number> = {
        page: cmdOpts.page,
        per_page: cmdOpts.perPage,
      };
      if (cmdOpts.summary) query.view = "summary";

      if (cmdOpts.spec) {
        const fullSpecId = await resolveSpecId(cmdOpts.spec, opts);
        path = `/behavior-specs/${fullSpecId}/runs`;
      }

      const response = await get<Run[]>(path, query, opts);
      const { data, meta } = response;

      if (isJsonMode()) {
        return printJson(cmdOpts.summary ? response : { data, meta });
      }

      printTable(
        ["ID", "Spec", "#", "Status", "Score", "Started", "Completed"],
        data.map((r) => [
          shortId(r.id),
          r._spec_name ? truncate(r._spec_name, 20) : shortId(r.behavior_spec_id),
          String(r.run_number),
          formatStatus(r.status),
          formatEvalScore(r) ?? "—",
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
        ["Score", formatEvalScore(data)],
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

      printEvalOutputDiagnostics(data.eval_summary?.output_diagnostics);

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

  addRunConfigurationOptions(
    runs
      .command("estimate")
      .description("Estimate run cost and duration before starting")
      .argument("<spec-id>", "Behaviour spec ID (full UUID or 8+ char prefix)"),
  )
    .action(async (specId: string, cmdOpts) => {
      const opts = parent.opts() as ClientOpts;
      const body = await buildRunRequestBody(cmdOpts, opts);
      const fullSpecId = await resolveSpecId(specId, opts);
      const { data } = await post<RunEstimate>(
        `/behavior-specs/${fullSpecId}/runs/estimate`,
        body,
        opts,
      );

      if (isJsonMode()) return printJson(data);
      printRunEstimate(data);
    });

  addRunConfigurationOptions(
    runs
      .command("start")
      .description("Start a new run for a behaviour spec")
      .argument("<spec-id>", "Behaviour spec ID (full UUID or 8+ char prefix)"),
  )
    .action(async (specId: string, cmdOpts) => {
      const opts = parent.opts() as ClientOpts;
      const body = await buildRunRequestBody(cmdOpts, opts);
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
        ["Score", formatEvalScore(run)],
        ["Error", run.error ?? undefined],
        ["Completed", formatDate(run.completed_at)],
      ]);
    });

  runs
    .command("diagnose")
    .description("Show live run diagnostics")
    .argument("<id>", "Run ID (full UUID or 8+ char prefix)")
    .action(async (id: string) => {
      const opts = parent.opts() as ClientOpts;
      const fullId = await resolveRunId(id, opts);
      const { data } = await get<RunDiagnostics>(
        `/runs/${fullId}/diagnostics`,
        undefined,
        opts,
      );

      if (isJsonMode()) return printJson(data);
      printDiagnostics(data);
    });

  runs
    .command("report")
    .description("Show run metrics and side-by-side eval output insights")
    .argument("<id>", "Run ID (full UUID or 8+ char prefix)")
    .option("--split <split>", "Split to show: primary, test, or all", "primary")
    .option("--limit <n>", "Number of examples to show", "5")
    .option("--mode <mode>", "Example mode: regressions or failures", "regressions")
    .action(async (id: string, cmdOpts) => {
      const opts = parent.opts() as ClientOpts;
      const fullId = await resolveRunId(id, opts);
      const { data } = await get<RunReport>(
        `/runs/${fullId}/report`,
        undefined,
        opts,
      );

      if (isJsonMode()) return printJson(data);

      const split = String(cmdOpts.split);
      if (!["primary", "test", "all"].includes(split)) {
        throw new Error("--split must be one of: primary, test, all");
      }
      const mode = String(cmdOpts.mode);
      if (!["regressions", "failures"].includes(mode)) {
        throw new Error("--mode must be one of: regressions, failures");
      }
      const limit = Number(cmdOpts.limit);
      if (!Number.isFinite(limit) || limit < 0) {
        throw new Error("--limit must be a non-negative number");
      }

      printRunReport(data, {
        split: split as "primary" | "test" | "all",
        mode: mode as "regressions" | "failures",
        limit: Math.floor(limit),
      });
    });
}
