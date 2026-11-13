import type { Command } from "commander";
import { isJsonMode, printJson, printSuccess } from "../output.js";
import { createRunConsoleReporter } from "../local-runtime/console-reporter.js";
import { runImagePipeline } from "../local-runtime/image-runner.js";

export function registerImagePipelineCommand(pipeline: Command): void {
  pipeline.command("image")
    .description("Experimental H3-Image captioned-image LoRA pipeline (local CUDA)")
    .requiredOption("--spec <path>", "Image training spec (engine: image)")
    .option("--output <path>", "New run directory; existing directories are never overwritten")
    .option("--dry-run", "Validate inputs and display the plan without downloads or training")
    .option("--verbose", "Stream Python runtime logs")
    .action(async (options: { spec: string; output?: string; dryRun?: boolean; verbose?: boolean }) => {
      const result = await runImagePipeline({
        specPath: options.spec, outputDir: options.output, dryRun: options.dryRun,
        reporter: createRunConsoleReporter({ verbose: options.verbose }),
      });
      if (isJsonMode() || options.dryRun) return printJson(result);
      printSuccess(`Image pipeline completed. Report: ${"report_path" in result ? result.report_path : ""}`);
    });
}
