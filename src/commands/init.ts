import { Command } from "commander";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { printSuccess, printError, printWarning } from "../output.js";
import type { LocalSpec } from "../eval/types.js";

const DEFAULT_SPEC_FILE = "tunedtensor.json";

const SCAFFOLD: LocalSpec = {
  name: "My Agent",
  description: "",
  base_model: "meta-llama/Llama-3.2-3B-Instruct",
  system_prompt: "You are a helpful assistant.",
  guidelines: [],
  constraints: [],
  examples: [
    { input: "Hello", output: "Hi! How can I help you today?" },
  ],
  eval_cases: [],
};

export function registerInitCommand(parent: Command) {
  parent
    .command("init")
    .description("Create a local behaviour spec file")
    .option("-n, --name <name>", "Spec name")
    .option("--model <model>", "Base model ID")
    .option("-f, --file <path>", "Output file path", DEFAULT_SPEC_FILE)
    .action(async (cmdOpts) => {
      const filePath = resolve(cmdOpts.file);

      if (existsSync(filePath)) {
        printWarning(`${cmdOpts.file} already exists. Use tt eval to run evals or edit it directly.`);
        return;
      }

      const spec: LocalSpec = { ...SCAFFOLD };
      if (cmdOpts.name) spec.name = cmdOpts.name;
      if (cmdOpts.model) spec.base_model = cmdOpts.model;

      writeFileSync(filePath, JSON.stringify(spec, null, 2) + "\n");
      printSuccess(`Created ${cmdOpts.file}`);
      console.log("\nNext steps:");
      console.log("  1. Edit the spec: system_prompt, guidelines, examples");
      console.log("  2. Run local evals:  tt eval");
      console.log("  3. Push to remote:   tt push");
    });
}

export function loadSpec(filePath?: string): LocalSpec {
  const resolved = resolve(filePath || DEFAULT_SPEC_FILE);
  if (!existsSync(resolved)) {
    printError(
      `Spec file not found: ${filePath || DEFAULT_SPEC_FILE}\nRun \`tt init\` to create one.`,
    );
    process.exit(1);
  }

  const raw = JSON.parse(readFileSync(resolved, "utf-8"));
  return raw as LocalSpec;
}

export { DEFAULT_SPEC_FILE };
