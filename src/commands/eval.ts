import { Command } from "commander";
import ora from "ora";
import chalk from "chalk";
import {
  printTable,
  printJson,
  isJsonMode,
  printError,
  printWarning,
  truncate,
} from "../output.js";
import { loadSpec, DEFAULT_SPEC_FILE } from "./init.js";
import { runEvals } from "../eval/runner.js";
import { checkProviderAvailability } from "../eval/providers.js";
import type { ProviderConfig, EvalSummary, ValidationCheck } from "../eval/types.js";

export function registerEvalCommand(parent: Command) {
  parent
    .command("eval")
    .description("Evaluate a model's performance against a behaviour spec")
    .requiredOption(
      "-p, --provider <provider>",
      "Model provider: ollama, openai, or custom",
    )
    .option("-m, --model <model>", "Model name to evaluate")
    .option("-f, --file <path>", "Spec file path", DEFAULT_SPEC_FILE)
    .option("--base-url <url>", "Custom provider base URL")
    .option("--provider-api-key <key>", "Custom provider API key")
    .action(async (cmdOpts) => {
      const spec = loadSpec(cmdOpts.file);

      const provider: ProviderConfig = {
        provider: cmdOpts.provider as ProviderConfig["provider"],
        model: cmdOpts.model || inferDefaultModel(cmdOpts.provider),
        baseUrl: cmdOpts.baseUrl,
        apiKey: cmdOpts.providerApiKey,
      };

      const spinner = ora(`Checking ${provider.provider} availability...`).start();
      const check = await checkProviderAvailability(provider);
      if (!check.available) {
        spinner.fail(check.error);
        process.exit(1);
      }
      spinner.succeed(`${provider.provider} ready (${provider.model})`);

      if (!isJsonMode()) {
        console.log(
          chalk.dim(`\nSpec: ${cmdOpts.file}  |  Model: ${provider.model}\n`),
        );
      }

      const evalSpinner = ora("Running evals...").start();

      const summary = await runEvals(spec, provider, (done, total) => {
        evalSpinner.text = `Running evals... ${done}/${total}`;
      });

      evalSpinner.stop();

      if (isJsonMode()) return printJson(summary);

      printValidation(summary.spec_validation.checks);
      console.log();
      printResults(summary);
    });
}

function inferDefaultModel(provider: string): string {
  switch (provider) {
    case "ollama":
      return "llama3.2";
    case "openai":
      return "gpt-4o-mini";
    default:
      return "default";
  }
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
    ["#", "Passed", "Score", "Input", "Latency"],
    summary.results.map((r, i) => [
      String(i + 1),
      r.passed ? chalk.green("✓") : chalk.red("✗"),
      r.score != null ? r.score.toFixed(2) : "—",
      truncate(r.input, 45),
      r.latency_ms != null ? `${r.latency_ms}ms` : "—",
    ]),
  );

  console.log();

  const scoreStr = summary.mean_score != null
    ? (summary.mean_score * 100).toFixed(1) + "%"
    : "n/a";
  const passStr = (summary.pass_rate * 100).toFixed(1) + "%";
  const color = summary.pass_rate >= 0.8
    ? chalk.green
    : summary.pass_rate >= 0.5
      ? chalk.yellow
      : chalk.red;

  console.log(
    `  Score: ${chalk.bold(scoreStr)}  Pass Rate: ${color(chalk.bold(passStr))}  (${summary.passed}/${summary.total} passed)`,
  );

  const failedResults = summary.results.filter((r) => !r.passed);
  if (failedResults.length) {
    console.log(chalk.dim(`\nFailed eval details:`));
    for (const r of failedResults) {
      console.log(chalk.dim(`  • "${truncate(r.input, 50)}":`));
      if (r.reasoning) console.log(chalk.dim(`    ${r.reasoning}`));
      for (const a of r.assertions.filter((a) => !a.passed)) {
        console.log(chalk.red(`    ✗ ${a.assertion}: ${a.message}`));
      }
    }
  }
}
