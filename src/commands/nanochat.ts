import { accessSync, constants, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { Command } from "commander";
import {
  defaultNanochatLifecycleConfig,
  inspectNanochatSource,
  loadNanochatLifecycleConfig,
  runNanochatLifecycle,
} from "../local-runtime/nanochat-lifecycle.js";
import { isJsonMode, printJson, printSuccess } from "../output.js";

const DEFAULT_NANOCHAT_FILE = "nanochat.json";

export function registerNanochatCommands(parent: Command): void {
  const nanochat = parent.command("nanochat")
    .description("Run and audit a pinned nanochat training lifecycle");

  nanochat.command("init")
    .description("Create a bounded, GPU-smoke nanochat lifecycle configuration")
    .requiredOption("--checkout <path>", "Path to a nanochat Git checkout")
    .requiredOption("--python <path>", "Python executable with nanochat dependencies and CUDA PyTorch")
    .option("-f, --file <path>", "Output configuration file", DEFAULT_NANOCHAT_FILE)
    .action(async (options: { checkout: string; python: string; file: string }) => {
      const path = resolve(options.file);
      if (existsSync(path)) throw new Error(`Refusing to overwrite existing file: ${options.file}`);
      const source = await inspectNanochatSource(resolve(options.checkout));
      if (source.dirty) throw new Error("Refusing to initialize from a nanochat checkout with tracked modifications.");
      const config = defaultNanochatLifecycleConfig({
        checkout: source.checkout,
        revision: source.revision,
        python: resolve(options.python),
      });
      mkdirSync(dirname(path), { recursive: true });
      writeFileSync(path, `${JSON.stringify(config, null, 2)}\n`, { encoding: "utf8", mode: 0o600 });
      if (isJsonMode()) return printJson({ created: true, path, config });
      printSuccess(`Created ${options.file} pinned to nanochat ${source.revision.slice(0, 12)}.`);
    });

  nanochat.command("validate")
    .description("Validate configuration, source revision, cleanliness, and Python path without executing")
    .option("-f, --file <path>", "Lifecycle configuration file", DEFAULT_NANOCHAT_FILE)
    .action(async (options: { file: string }) => {
      const config = await loadNanochatLifecycleConfig(options.file);
      const source = await inspectNanochatSource(config.source.checkout);
      const errors: string[] = [];
      if (source.revision !== config.source.revision) {
        errors.push(`revision mismatch: expected ${config.source.revision}, found ${source.revision}`);
      }
      if (source.dirty && !config.source.allowDirty) errors.push("checkout has tracked modifications");
      try {
        accessSync(config.source.python, constants.X_OK);
      } catch {
        errors.push(`Python executable not found or not executable: ${config.source.python}`);
      }
      const result = { valid: errors.length === 0, errors, source, config };
      if (isJsonMode()) return printJson(result);
      if (errors.length) throw new Error(`Invalid nanochat lifecycle:\n- ${errors.join("\n- ")}`);
      printSuccess(`Nanochat lifecycle is valid at ${source.revision.slice(0, 12)}.`);
    });

  nanochat.command("run")
    .description("Execute every enabled stage and write a content-hashed lifecycle manifest")
    .option("-f, --file <path>", "Lifecycle configuration file", DEFAULT_NANOCHAT_FILE)
    .option("--run-id <uuid>", "Stable run identifier (defaults to a generated UUID)")
    .action(async (options: { file: string; runId?: string }) => {
      const config = await loadNanochatLifecycleConfig(options.file);
      const runId = options.runId;
      if (runId && !/^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(runId)) {
        throw new Error("--run-id must be a UUID");
      }
      const audit = await runNanochatLifecycle(config, { runId });
      if (isJsonMode()) return printJson(audit);
      printSuccess(`Nanochat lifecycle ${audit.run_id} completed. Audit: ${resolve(audit.artifact_root, "lifecycle.json")}`);
    });
}

export { DEFAULT_NANOCHAT_FILE };
