import { Command } from "commander";
import type { AgentModelRuntime } from "../agent-model.js";
import {
  describeAgentModel,
  listAgentModels,
  setAgentModel,
} from "../agent-control.js";
import type { AgentThinkingLevel } from "../config.js";
import { isJsonMode } from "../output.js";

export interface AgentCommandsOptions {
  env: NodeJS.ProcessEnv;
  output: NodeJS.WritableStream;
  getRuntime(): Promise<AgentModelRuntime>;
}

export function registerAgentCommands(parent: Command, options: AgentCommandsOptions): void {
  const agent = parent.command("agent").description("Configure the laptop-local TT agent");

  agent.command("status").description("Show the selected local provider, model, thinking, and auth state")
    .action(async () => {
      const runtime = await options.getRuntime();
      const summary = describeAgentModel(runtime, options.env);
      if (!summary) {
        throw new Error(
          "The local agent is not configured. Run `tt agent configure --provider <provider> --model <model>`.",
        );
      }
      const value = {
        execution: "local",
        provider: summary.provider,
        provider_name: summary.providerName,
        model: summary.model,
        model_name: summary.modelName,
        thinking: summary.thinking,
        authenticated: summary.authenticated,
        supports_thinking: summary.supportsThinking,
      };
      const modelLabel = summary.modelName && summary.modelName !== summary.model
        ? `${summary.modelName} (${summary.provider}/${summary.model})`
        : `${summary.provider}/${summary.model}`;
      options.output.write(isJsonMode()
        ? `${JSON.stringify(value, null, 2)}\n`
        : `Local agent: ${modelLabel} (thinking: ${value.thinking}, auth: ${value.authenticated ? "configured" : "required"})\n`);
    });

  agent.command("models").description("List provider models available to the local agent")
    .option("--provider <provider>", "Filter by provider ID")
    .option("--all", "Include models whose provider auth is not configured")
    .action(async (commandOptions: { provider?: string; all?: boolean }) => {
      const runtime = await options.getRuntime();
      if (commandOptions.provider && !runtime.getProviders().some((provider) => provider.id === commandOptions.provider)) {
        throw new Error(`Unknown provider "${commandOptions.provider}".`);
      }
      const models = listAgentModels(runtime, {
        provider: commandOptions.provider,
        includeUnauthenticated: commandOptions.all,
      });
      if (isJsonMode()) {
        options.output.write(`${JSON.stringify({ data: models }, null, 2)}\n`);
        return;
      }
      if (models.length === 0) {
        options.output.write("No matching authenticated models. Use --all to inspect the full catalog.\n");
        return;
      }
      for (const model of models) {
        options.output.write(`${model.provider}/${model.id}${model.thinking ? "  thinking" : ""}${model.authenticated ? "" : "  auth required"}\n`);
      }
    });

  agent.command("configure").description("Select the local provider, model, and thinking level")
    .requiredOption("--provider <provider>", "Provider ID")
    .requiredOption("--model <model>", "Model ID")
    .option("--thinking <level>", "off, minimal, low, medium, high, xhigh, or max", "medium")
    .action(async (commandOptions: { provider: string; model: string; thinking: string }) => {
      const result = setAgentModel(
        await options.getRuntime(),
        options.env,
        commandOptions.provider,
        commandOptions.model,
        { thinking: commandOptions.thinking as AgentThinkingLevel, adjustThinking: false },
      );
      const value = { execution: "local", ...result.selection };
      options.output.write(isJsonMode()
        ? `${JSON.stringify(value, null, 2)}\n`
        : `Configured local agent: ${result.selection.provider}/${result.selection.model} (thinking: ${result.selection.thinking}).\n`);
    });
}
