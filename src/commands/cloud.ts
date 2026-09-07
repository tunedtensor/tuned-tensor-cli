import { Command } from "commander";
import { get, type ClientOpts } from "../client.js";
import { isJsonMode, printDetail, printJson } from "../output.js";
import { registerSpecsCommands } from "./specs.js";
import { registerDatasetsCommands } from "./datasets.js";
import { registerLabelCommands } from "./label.js";
import { registerRunsCommands } from "./runs.js";
import { registerModelsCommands } from "./models.js";
import { registerPushCommand } from "./push.js";

/** Explicit cloud namespace keeps the existing local defaults unambiguous. */
export function registerCloudCommands(parent: Command): void {
  const cloud = parent.command("cloud").description("Operate cloud specs, datasets, runs, and models with a TT access token");
  registerPushCommand(cloud);
  registerSpecsCommands(cloud);
  registerDatasetsCommands(cloud);
  registerLabelCommands(cloud);
  registerRunsCommands(cloud);
  registerModelsCommands(cloud);
}

export interface ManagedAgentUsage {
  period_start: string;
  resets_at: string;
  requests: number;
  completed_requests: number;
  failed_requests: number;
  cancelled_requests: number;
  running_requests: number;
  usage_reported_requests: number;
  cost_reported_requests: number;
  prompt_tokens: number;
  completion_tokens: number;
  provider_cost_usd: number | null;
  daily_request_limit: number;
  remaining_requests: number;
  billing: "included_allowance";
}

export function registerUsageCommand(parent: Command): void {
  parent.command("usage")
    .description("Show managed agent request allowance and reported token usage")
    .action(async () => {
      const { data } = await get<ManagedAgentUsage>("/agent/usage", undefined, parent.optsWithGlobals() as ClientOpts);
      if (isJsonMode()) return printJson(data);
      printDetail([
        ["Managed requests", `${data.requests} / ${data.daily_request_limit}`],
        ["Remaining", String(data.remaining_requests)],
        ["Resets", data.resets_at],
        ["Completed", String(data.completed_requests)],
        ["Running", String(data.running_requests)],
        ["Failed / cancelled", `${data.failed_requests} / ${data.cancelled_requests}`],
        ["Reported input tokens", String(data.prompt_tokens)],
        ["Reported output tokens", String(data.completion_tokens)],
        ["Usage reported", `${data.usage_reported_requests} / ${data.requests} requests`],
        ["Provider cost reported", data.provider_cost_usd === null ? "Unknown" : `$${data.provider_cost_usd.toFixed(6)}`],
        ["Cost coverage", `${data.cost_reported_requests} / ${data.requests} requests`],
        ["Billing", "Included daily allowance"],
      ]);
      console.log("\nEach model request, including tool follow-ups, uses the allowance. Failed and cancelled requests also count. BYO provider usage is reported by that provider.\nUse `tt balance` for cloud training credits.");
    });
}
