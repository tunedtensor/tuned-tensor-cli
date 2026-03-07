import { Command } from "commander";
import { setJsonMode, printError } from "./output.js";
import { registerAuthCommands } from "./commands/auth.js";
import { registerSpecsCommands } from "./commands/specs.js";
import { registerRunsCommands } from "./commands/runs.js";
import { registerDatasetsCommands } from "./commands/datasets.js";
import { registerModelsCommands } from "./commands/models.js";

import { registerUsageCommands } from "./commands/usage.js";
import { registerInitCommand } from "./commands/init.js";
import { registerEvalCommand } from "./commands/eval.js";
import { registerPushCommand } from "./commands/push.js";
import { ApiError } from "./client.js";

declare const __TT_VERSION__: string;

const program = new Command();

program
  .name("tt")
  .description("Tuned Tensor CLI — fine-tune and evaluate LLMs")
  .version(__TT_VERSION__)
  .option("-k, --api-key <key>", "API key (overrides stored key)")
  .option(
    "-u, --base-url <url>",
    "API base URL (default: https://www.tunedtensor.com)",
  )
  .option("--json", "Output raw JSON")
  .option("--no-color", "Disable colors")
  .hook("preAction", (_thisCommand, actionCommand) => {
    const rootOpts = program.opts();
    if (rootOpts.json) setJsonMode(true);
    if (rootOpts.color === false) {
      process.env.FORCE_COLOR = "0";
    }
  });

registerAuthCommands(program);
registerSpecsCommands(program);
registerRunsCommands(program);
registerDatasetsCommands(program);
registerModelsCommands(program);

registerUsageCommands(program);
registerInitCommand(program);
registerEvalCommand(program);
registerPushCommand(program);

program.parseAsync().catch((err) => {
  if (err instanceof ApiError) {
    printError(`[${err.status}] ${err.message}`);
    process.exit(1);
  }
  printError(err.message || String(err));
  process.exit(1);
});
