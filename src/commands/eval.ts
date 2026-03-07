import { Command } from "commander";
import chalk from "chalk";
import {
  printTable,
  printJson,
  isJsonMode,
  printWarning,
  truncate,
} from "../output.js";
import { loadSpec, DEFAULT_SPEC_FILE } from "./init.js";
import { runEvals } from "../eval/runner.js";
import type { EvalSummary, ValidationCheck } from "../eval/types.js";

export function registerEvalCommand(parent: Command) {
  parent
    .command("eval")
    .description("Run rule-based evals against a behaviour spec")
    .option("-f, --file <path>", "Spec file path", DEFAULT_SPEC_FILE)
    .action(async (cmdOpts) => {
      const spec = loadSpec(cmdOpts.file);

      if (!isJsonMode()) {
        console.log(chalk.dim(`\nSpec: ${cmdOpts.file}\n`));
      }

      const summary = runEvals(spec);

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

  printTable(
    ["#", "Passed", "Input"],
    summary.results.map((r, i) => [
      String(i + 1),
      r.passed ? chalk.green("✓") : chalk.red("✗"),
      truncate(r.input, 55),
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
      for (const a of r.assertions.filter((a) => !a.passed)) {
        console.log(chalk.red(`    ✗ ${a.assertion}: ${a.message}`));
      }
    }
  }
}
