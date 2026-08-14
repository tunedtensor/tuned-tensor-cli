import {
  resolveAgentModel,
  resolveAgentModelDefinition,
  type AgentModelRuntime,
} from "./agent-model.js";
import {
  AGENT_THINKING_LEVELS,
  getAgentSelection,
  updateConfig,
  type AgentSelection,
  type AgentThinkingLevel,
} from "./config.js";

/**
 * Functions for reading and changing the laptop-local Pi agent model. Both
 * the `tt agent ...` commands and the `/model` shell command are thin wrappers
 * over these functions.
 */

export interface AgentModelChoice {
  provider: string;
  id: string;
  name?: string;
  authenticated: boolean;
  thinking: boolean;
}

export interface AgentModelSummary {
  provider: string;
  providerName?: string;
  model: string;
  modelName?: string;
  thinking: AgentThinkingLevel;
  authenticated: boolean;
  supportsThinking: boolean;
}

export interface SetAgentModelOptions {
  thinking?: AgentThinkingLevel;
  /**
   * When true (the default for quick switching), a model that cannot reason
   * falls back to thinking "off". When false, an incompatible thinking level
   * raises an error instead.
   */
  adjustThinking?: boolean;
}

export interface SetAgentModelResult {
  selection: AgentSelection;
  adjustedThinking: boolean;
}

export function listAgentModels(
  runtime: AgentModelRuntime,
  options: { provider?: string; includeUnauthenticated?: boolean } = {},
): AgentModelChoice[] {
  return runtime
    .getModels(options.provider)
    .filter((model) =>
      options.includeUnauthenticated || runtime.hasConfiguredAuth(model.provider)
    )
    .map((model) => ({
      provider: model.provider,
      id: model.id,
      name: model.name ?? model.id,
      authenticated: runtime.hasConfiguredAuth(model.provider),
      thinking: model.reasoning !== false,
    }))
    .sort((left, right) => {
      const leftKey = `${left.provider}/${left.id}`;
      const rightKey = `${right.provider}/${right.id}`;
      return leftKey.localeCompare(rightKey);
    });
}

export function describeAgentModel(
  runtime: AgentModelRuntime,
  env: Readonly<NodeJS.ProcessEnv>,
): AgentModelSummary | undefined {
  const selection = getAgentSelection(env);
  if (!selection) return undefined;
  const resolved = resolveAgentModelDefinition(runtime, selection);
  const providerName = runtime
    .getProviders()
    .find((provider) => provider.id === selection.provider)?.name;
  return {
    provider: selection.provider,
    providerName,
    model: selection.model,
    modelName: resolved.model.name,
    thinking: selection.thinking,
    authenticated: runtime.hasConfiguredAuth(selection.provider),
    supportsThinking: resolved.model.reasoning !== false,
  };
}

export function setAgentModel(
  runtime: AgentModelRuntime,
  env: Readonly<NodeJS.ProcessEnv>,
  provider: string,
  model: string,
  options: SetAgentModelOptions = {},
): SetAgentModelResult {
  const current = getAgentSelection(env);
  const requestedThinking = options.thinking ?? current?.thinking ?? "medium";
  if (!AGENT_THINKING_LEVELS.includes(requestedThinking)) {
    throw new Error(
      `Agent thinking must be one of: ${AGENT_THINKING_LEVELS.join(", ")}`,
    );
  }

  const adjust = options.adjustThinking ?? true;
  if (!adjust) {
    const selection: AgentSelection = {
      provider,
      model,
      thinking: requestedThinking,
    };
    resolveAgentModel(runtime, selection);
    updateConfig({ agent: selection });
    return { selection, adjustedThinking: false };
  }

  const definition = resolveAgentModelDefinition(runtime, {
    provider,
    model,
    thinking: "off",
  });
  const supportsThinking = definition.model.reasoning !== false;
  const effectiveThinking: AgentThinkingLevel = supportsThinking
    ? requestedThinking
    : "off";
  const selection: AgentSelection = {
    provider,
    model,
    thinking: effectiveThinking,
  };
  resolveAgentModel(runtime, selection);
  updateConfig({ agent: selection });
  return { selection, adjustedThinking: effectiveThinking !== requestedThinking };
}
