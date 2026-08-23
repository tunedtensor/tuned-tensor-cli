import { chmodSync, existsSync } from "node:fs";
import {
  getAgentAuthPath,
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
 * Functions for reading and changing the laptop-local TT agent model. Both
 * the `tt agent ...` commands and the `/model` shell command are thin wrappers
 * over these functions.
 */

/** Providers shown in `/login` and `/model` so onboarding stays short. */
export const FEATURED_AGENT_PROVIDERS = [
  "anthropic",
  "openai",
  "google",
  "openrouter",
] as const;

/** Explicit `/model` suggestions. First matching catalog id wins. */
export const RECOMMENDED_AGENT_MODELS = [
  { provider: "openai", ids: ["gpt-5.6-sol"] },
  {
    provider: "openrouter",
    ids: ["deepseek/deepseek-v4-flash-0731", "deepseek/deepseek-v4-flash"],
  },
] as const;

const FEATURED_PROVIDER_INDEX = new Map<string, number>(
  FEATURED_AGENT_PROVIDERS.map((id, index) => [id, index]),
);

function isFeaturedProvider(id: string): boolean {
  return FEATURED_PROVIDER_INDEX.has(id);
}

export interface ListAgentProvidersOptions {
  /** Limit the list to the onboarding providers. */
  featuredOnly?: boolean;
}

export interface AgentProviderChoice {
  id: string;
  name: string;
  authenticated: boolean;
}

export interface AgentModelChoice {
  provider: string;
  id: string;
  name: string;
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

export interface ListAgentModelsOptions {
  provider?: string;
  includeUnauthenticated?: boolean;
  /** Free-text filter matched against provider/model/name. */
  query?: string;
  /** Cap the returned list after relevance/alphabetic ordering. */
  limit?: number;
  /** Limit the list to the onboarding providers. */
  featuredOnly?: boolean;
}

function modelKey(model: { provider: string; id: string }): string {
  return `${model.provider}/${model.id}`;
}

function modelRelevance(
  model: AgentModelChoice,
  query: string,
): number | undefined {
  const q = query.trim().toLowerCase();
  if (!q) return undefined;
  const fullId = modelKey(model).toLowerCase();
  const bareId = model.id.toLowerCase();
  const name = model.name.toLowerCase();
  if (fullId === q || bareId === q) return 0;
  if (fullId.startsWith(q)) return 1;
  if (bareId.startsWith(q)) return 2;
  if (fullId.includes(q)) return 3;
  if (name.includes(q)) return 4;
  return undefined;
}

export function listAgentProviders(
  runtime: AgentModelRuntime,
  options: ListAgentProvidersOptions = {},
): AgentProviderChoice[] {
  return runtime
    .getProviders()
    .filter((provider) => !options.featuredOnly || isFeaturedProvider(provider.id))
    .map((provider): AgentProviderChoice => ({
      id: provider.id,
      name: provider.name ?? provider.id,
      authenticated: runtime.hasConfiguredAuth(provider.id),
    }))
    .sort((left, right) => {
      const leftRank = FEATURED_PROVIDER_INDEX.get(left.id);
      const rightRank = FEATURED_PROVIDER_INDEX.get(right.id);
      if (leftRank !== undefined || rightRank !== undefined) {
        return (leftRank ?? Number.MAX_SAFE_INTEGER) - (rightRank ?? Number.MAX_SAFE_INTEGER)
          || left.id.localeCompare(right.id);
      }
      return left.id.localeCompare(right.id);
    });
}

export function findAgentProvider(
  runtime: AgentModelRuntime,
  id: string,
): AgentProviderChoice | undefined {
  const needle = id.trim().toLowerCase();
  if (!needle) return undefined;
  return listAgentProviders(runtime).find((provider) => provider.id.toLowerCase() === needle);
}

export function listAgentModels(
  runtime: AgentModelRuntime,
  options: ListAgentModelsOptions = {},
): AgentModelChoice[] {
  let models: AgentModelChoice[] = runtime
    .getModels(options.provider)
    .filter((model) =>
      (options.includeUnauthenticated || runtime.hasConfiguredAuth(model.provider))
      && (!options.featuredOnly || isFeaturedProvider(model.provider))
    )
    .map((model): AgentModelChoice => ({
      provider: model.provider,
      id: model.id,
      name: model.name ?? model.id,
      authenticated: runtime.hasConfiguredAuth(model.provider),
      thinking: model.reasoning !== false,
    }));

  const query = options.query?.trim().toLowerCase();
  if (query) {
    const scored: Array<{ model: AgentModelChoice; score: number }> = [];
    for (const model of models) {
      const score = modelRelevance(model, query);
      if (score !== undefined) scored.push({ model, score });
    }
    scored.sort((left, right) =>
      left.score - right.score
      || modelKey(left.model).localeCompare(modelKey(right.model))
    );
    models = scored.map((entry) => entry.model);
  } else {
    models.sort((left, right) => modelKey(left).localeCompare(modelKey(right)));
  }

  if (options.limit !== undefined && options.limit > 0) {
    models = models.slice(0, options.limit);
  }
  return models;
}

/** The two onboarding model picks, if they exist in the catalog. */
export function recommendAgentModels(
  runtime: AgentModelRuntime,
  options: Pick<ListAgentModelsOptions, "includeUnauthenticated"> = {},
): AgentModelChoice[] {
  const models = listAgentModels(runtime, {
    includeUnauthenticated: options.includeUnauthenticated ?? true,
  });
  const picks: AgentModelChoice[] = [];
  for (const recommendation of RECOMMENDED_AGENT_MODELS) {
    const match = recommendation.ids
      .map((id) => models.find((model) =>
        model.provider === recommendation.provider && model.id === id
      ))
      .find((model) => model !== undefined);
    if (match) picks.push(match);
  }
  return picks;
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

export async function loginAgentProvider(
  runtime: AgentModelRuntime,
  provider: string,
  apiKey: string,
): Promise<{ provider: string }> {
  const choice = findAgentProvider(runtime, provider);
  if (!choice) {
    throw new Error(`Unknown provider "${provider}". Use /model to list providers.`);
  }
  const key = apiKey.trim();
  if (!key) {
    throw new Error("API key cannot be empty.");
  }
  if (typeof runtime.setRuntimeApiKey !== "function") {
    throw new Error("This session cannot store provider credentials.");
  }
  await runtime.setRuntimeApiKey(choice.id, key);
  const authPath = getAgentAuthPath();
  if (existsSync(authPath)) chmodSync(authPath, 0o600);
  return { provider: choice.id };
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
