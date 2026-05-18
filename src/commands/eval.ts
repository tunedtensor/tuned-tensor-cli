import { Command } from "commander";
import chalk from "chalk";
import {
  printJson,
  isJsonMode,
} from "../output.js";
import { loadSpec, DEFAULT_SPEC_FILE } from "./init.js";
import { validateSpec } from "../eval/rules.js";
import type { ValidationCheck } from "../eval/types.js";

export function registerEvalCommand(parent: Command) {
  parent
    .command("eval")
    .description("Validate a local behaviour spec")
    .option("-m, --model <model>", "Deprecated; ignored because tt eval only validates the local spec")
    .option("-f, --file <path>", "Spec file path", DEFAULT_SPEC_FILE)
    .action(async (cmdOpts) => {
      const spec = loadSpec(cmdOpts.file);
      const validation = validateSpec(spec);

      if (!isJsonMode()) {
        console.log(chalk.dim(`\nSpec: ${cmdOpts.file}\n`));
      }

      if (isJsonMode()) {
        printJson(validation);
      } else {
        printValidation(validation.checks);
      }

      if (!validation.valid) {
        process.exitCode = 1;
      }
    });
}

function printValidation(checks: ValidationCheck[]) {
  console.log(chalk.bold("Spec Validation"));
  for (const check of checks) {
    const icon = check.passed
      ? chalk.green("✓")
      : check.severity === "warning"
        ? chalk.yellow("⚠")
        : chalk.red("✗");
    const msg = check.message
      ? chalk.dim(` — ${check.message}`)
      : "";
    console.log(`  ${icon} ${check.name}${msg}`);
  }
}
