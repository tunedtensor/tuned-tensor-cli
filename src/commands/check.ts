import { Command } from "commander";
import chalk from "chalk";
import { printJson, isJsonMode } from "../output.js";
import { loadSpec, DEFAULT_SPEC_FILE } from "./init.js";
import { validateSpec } from "../eval/rules.js";
import type { ValidationCheck } from "../eval/types.js";

export function registerCheckCommand(parent: Command) {
  parent
    .command("check")
    .description("Validate a behaviour spec (free, no model needed)")
    .option("-f, --file <path>", "Spec file path", DEFAULT_SPEC_FILE)
    .action(async (cmdOpts) => {
      const spec = loadSpec(cmdOpts.file);
      const result = validateSpec(spec);

      if (isJsonMode()) return printJson(result);

      console.log(chalk.bold("Spec Validation"));
      for (const check of result.checks) {
        const icon = check.passed ? chalk.green("✓") : chalk.red("✗");
        const msg = check.message
          ? chalk.dim(` — ${check.message}`)
          : "";
        console.log(`  ${icon} ${check.name}${msg}`);
      }

      console.log();
      if (result.valid) {
        console.log(chalk.green("✓") + " Spec is valid. Run " + chalk.bold("tt eval --provider ollama") + " to evaluate model performance.");
      } else {
        console.log(chalk.red("✗") + " Spec has issues. Fix them before running evals.");
        process.exit(1);
      }
    });
}
