import { Command } from "commander";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  isJsonMode,
  printError,
  printJson,
  printSuccess,
  printWarning,
} from "../output.js";
import { canonicalizeBaseModel } from "../base-models.js";
import type { ProjectSpec } from "../project-spec.js";

const DEFAULT_SPEC_FILE = "tunedtensor.json";

const SCAFFOLD: ProjectSpec = {
  name: "My Agent",
  description: "",
  base_model: "Qwen/Qwen3.5-2B",
  system_prompt: "You are a helpful assistant.",
  guidelines: [],
  constraints: [],
  examples: [
    { input: "Hello", output: "Hi! How can I help you today?" },
    {
      input: "Summarize this update: The launch moved to Friday.",
      output: "The launch is now scheduled for Friday.",
    },
  ],
};

export function registerInitCommand(parent: Command) {
  parent
    .command("init")
    .description("Create a behaviour spec project file")
    .option("-n, --name <name>", "Spec name")
    .option("--model <model>", "Base model ID")
    .option("-f, --file <path>", "Output file path", DEFAULT_SPEC_FILE)
    .action(async (cmdOpts) => {
      const filePath = resolve(cmdOpts.file);

      if (existsSync(filePath)) {
        if (isJsonMode()) {
          printJson({ created: false, path: filePath });
          return;
        }
        printWarning(`${cmdOpts.file} already exists. Use tt eval to validate it or edit it directly.`);
        return;
      }

      const spec: ProjectSpec = {
        ...SCAFFOLD,
        examples: SCAFFOLD.examples.map((example) => ({ ...example })),
      };
      if (cmdOpts.name) spec.name = cmdOpts.name;
      if (cmdOpts.model) spec.base_model = canonicalizeBaseModel(cmdOpts.model);

      writeFileSync(filePath, JSON.stringify(spec, null, 2) + "\n");
      if (isJsonMode()) {
        printJson({ created: true, path: filePath, spec });
        return;
      }
      printSuccess(`Created ${cmdOpts.file}`);
      console.log("\nNext steps:");
      console.log("  1. Edit the spec: system_prompt, guidelines, examples");
      console.log("  2. Validate for cloud: tt eval");
      console.log("  3. Push to cloud:      tt push");
      console.log("     Or validate locally: tt local validate");
    });
}

export function loadSpec(filePath?: string): ProjectSpec {
  const resolved = resolve(filePath || DEFAULT_SPEC_FILE);
  if (!existsSync(resolved)) {
    printError(
      `Spec file not found: ${filePath || DEFAULT_SPEC_FILE}\nRun \`tt init\` to create one.`,
    );
    process.exit(1);
  }

  const raw = JSON.parse(readFileSync(resolved, "utf-8"));
  return raw as ProjectSpec;
}

export { DEFAULT_SPEC_FILE };
