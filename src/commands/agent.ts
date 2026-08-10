import { Command } from "commander";
import type { AgentModelRuntime } from "../agent-model.js";
import { resolveAgentModel, resolveAgentModelDefinition } from "../agent-model.js";
import {
  AGENT_THINKING_LEVELS,
  getAgentSelection,
  updateConfig,
  type AgentThinkingLevel,
} from "../config.js";
import { isJsonMode } from "../output.js";

export interface AgentCommandsOptions {
  env: NodeJS.ProcessEnv;
  output: NodeJS.WritableStream;
  getRuntime(): Promise<AgentModelRuntime>;
}

function selectionOrHint(env: NodeJS.ProcessEnv) {
  const selection = getAgentSelection(env);
  if (!selection) {
    throw new Error(
      "The local agent is not configured. Run `tt agent configure --provider <provider> --model <model>`.",
    );
  }
  return selection;
}

export function registerAgentCommands(parent: Command, options: AgentCommandsOptions): void {
  const agent = parent.command("agent").description("Configure the laptop-local Pi agent");

  agent.command("status").description("Show the selected local provider, model, thinking, and auth state")
    .action(async () => {
      const selection = selectionOrHint(options.env);
      const runtime = await options.getRuntime();
      const resolved = resolveAgentModelDefinition(runtime, selection);
      const authenticated = runtime.hasConfiguredAuth(selection.provider);
      const value = {
        execution: "local",
        provider: selection.provider,
        model: selection.model,
        thinking: selection.thinking,
        authenticated,
        supports_thinking: resolved.model.reasoning !== false,
      };
      options.output.write(isJsonMode()
        ? `${JSON.stringify(value, null, 2)}\n`
        : `Local agent: ${value.provider}/${value.model} (thinking: ${value.thinking}, auth: ${authenticated ? "configured" : "required"})\n`);
    });

  agent.command("models").description("List Pi provider models available to the local agent")
    .option("--provider <provider>", "Filter by Pi provider ID")
    .option("--all", "Include models whose provider auth is not configured")
    .action(async (commandOptions: { provider?: string; all?: boolean }) => {
      const runtime = await options.getRuntime();
      if (commandOptions.provider && !runtime.getProviders().some((provider) => provider.id === commandOptions.provider)) {
        throw new Error(`Unknown Pi provider "${commandOptions.provider}".`);
      }
      const models = runtime.getModels(commandOptions.provider).filter((model) =>
        commandOptions.all || runtime.hasConfiguredAuth(model.provider)
      ).map((model) => ({
        provider: model.provider,
        id: model.id,
        name: model.name ?? model.id,
        authenticated: runtime.hasConfiguredAuth(model.provider),
        thinking: model.reasoning !== false,
      }));
      if (isJsonMode()) {
        options.output.write(`${JSON.stringify({ data: models }, null, 2)}\n`);
        return;
      }
      if (models.length === 0) {
        options.output.write("No matching authenticated Pi models. Use --all to inspect the full catalog.\n");
        return;
      }
      for (const model of models) {
        options.output.write(`${model.provider}/${model.id}${model.thinking ? "  thinking" : ""}${model.authenticated ? "" : "  auth required"}\n`);
      }
    });

  agent.command("configure").description("Select the local Pi provider, model, and thinking level")
    .requiredOption("--provider <provider>", "Pi provider ID")
    .requiredOption("--model <model>", "Pi model ID")
    .option("--thinking <level>", "off, minimal, low, medium, high, xhigh, or max", "medium")
    .action(async (commandOptions: { provider: string; model: string; thinking: string }) => {
      if (!AGENT_THINKING_LEVELS.includes(commandOptions.thinking as AgentThinkingLevel)) {
        throw new Error(`--thinking must be one of: ${AGENT_THINKING_LEVELS.join(", ")}`);
      }
      const selection = {
        provider: commandOptions.provider,
        model: commandOptions.model,
        thinking: commandOptions.thinking as AgentThinkingLevel,
      };
      resolveAgentModel(await options.getRuntime(), selection);
      updateConfig({ agent: selection });
      const value = { execution: "local", ...selection };
      options.output.write(isJsonMode()
        ? `${JSON.stringify(value, null, 2)}\n`
        : `Configured local agent: ${selection.provider}/${selection.model} (thinking: ${selection.thinking}).\n`);
    });
}
