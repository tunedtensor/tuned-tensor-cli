import { Command } from "commander";
import ora from "ora";
import chalk from "chalk";
import {
  printTable,
  printJson,
  isJsonMode,
  printWarning,
  truncate,
} from "../output.js";
import type { ClientOpts } from "../client.js";
import { loadSpec, DEFAULT_SPEC_FILE } from "./init.js";
import { runEvals } from "../eval/runner.js";
import type { EvalSummary, ValidationCheck } from "../eval/types.js";

export function registerEvalCommand(parent: Command) {
  parent
    .command("eval")
    .description("Evaluate a behaviour spec against rule-based assertions, optionally calling a model")
    .option("-f, --file <path>", "Spec file path", DEFAULT_SPEC_FILE)
    .option("-m, --model <model>", "Model ID to evaluate (uses Tuned Tensor Playground API)")
    .action(async (cmdOpts) => {
      const spec = loadSpec(cmdOpts.file);
      const clientOpts = cmdOpts.model ? parent.opts() as ClientOpts : undefined;

      const mode = cmdOpts.model ? `model: ${cmdOpts.model}` : "offline (spec only)";
      if (!isJsonMode()) {
        console.log(chalk.dim(`\nSpec: ${cmdOpts.file}  |  Mode: ${mode}\n`));
      }

      let spinner: ReturnType<typeof ora> | null = null;
      if (cmdOpts.model) {
        spinner = ora("Running evals...").start();
      }

      const summary = await runEvals(spec, {
        model: cmdOpts.model,
        clientOpts,
        onProgress: spinner
          ? (done, total) => { spinner!.text = `Running evals... ${done}/${total}`; }
          : undefined,
      });

      spinner?.stop();

      if (isJsonMode()) return printJson(summary);

      printValidation(summary.spec_validation.checks);
      console.log();
      printResults(summary);
    });
}

function printValidation(checks: ValidationCheck[]) {
  console.log(chalk.bold("Spec Validation"));
  for (const check of checks) {
    const icon = check.passed ? chalk.green("✓") : chalk.red("✗");
    const msg = check.message
      ? chalk.dim(` — ${check.message}`)
      : "";
    console.log(`  ${icon} ${check.name}${msg}`);
  }
}

function printResults(summary: EvalSummary) {
  if (!summary.results.length) {
    printWarning("No eval cases to run. Add examples or eval_cases to your spec.");
    return;
  }

  console.log(chalk.bold(`Eval Results (${summary.total}):`));

  const hasModel = summary.model !== null;

  printTable(
    ["#", "Passed", "Input", ...(hasModel ? ["Latency"] : [])],
    summary.results.map((r, i) => [
      String(i + 1),
      r.passed ? chalk.green("✓") : chalk.red("✗"),
      truncate(r.input, hasModel ? 40 : 55),
      ...(hasModel ? [r.latency_ms != null ? `${r.latency_ms}ms` : "—"] : []),
    ]),
  );

  console.log();

  const passStr = (summary.pass_rate * 100).toFixed(1) + "%";
  const color = summary.pass_rate >= 0.8
    ? chalk.green
    : summary.pass_rate >= 0.5
      ? chalk.yellow
      : chalk.red;

  console.log(
    `  Pass Rate: ${color(chalk.bold(passStr))}  (${summary.passed}/${summary.total} passed)`,
  );

  const failedResults = summary.results.filter((r) => !r.passed);
  if (failedResults.length) {
    console.log(chalk.dim(`\nFailed eval details:`));
    for (const r of failedResults) {
      console.log(chalk.dim(`  • "${truncate(r.input, 50)}":`));
      if (hasModel && r.actual) {
        console.log(chalk.dim(`    Response: ${truncate(r.actual, 80)}`));
      }
      for (const a of r.assertions.filter((a) => !a.passed)) {
        console.log(chalk.red(`    ✗ ${a.assertion}: ${a.message}`));
      }
    }
  }
}
