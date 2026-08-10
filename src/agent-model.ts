import {
  getAgentDir,
  ModelRuntime,
} from "@earendil-works/pi-coding-agent";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import type { AgentSelection, AgentThinkingLevel } from "./config.js";

export interface AgentProviderInfo {
  id: string;
  name?: string;
}

export interface AgentModelInfo {
  id: string;
  provider: string;
  name?: string;
  reasoning?: boolean;
}

export interface AgentModelRuntime {
  getProviders(): readonly AgentProviderInfo[];
  getModels(provider?: string): readonly AgentModelInfo[];
  getModel(provider: string, model: string): AgentModelInfo | undefined;
  hasConfiguredAuth(provider: string): boolean;
}

export interface ResolvedAgentModel {
  model: AgentModelInfo;
  thinking: AgentThinkingLevel;
}

export function resolveAgentModel(
  runtime: AgentModelRuntime,
  selection: AgentSelection,
): ResolvedAgentModel {
  const resolved = resolveAgentModelDefinition(runtime, selection);
  if (!runtime.hasConfiguredAuth(selection.provider)) {
    throw new Error(
      `Authenticate Pi provider "${selection.provider}" first (for example with Pi's /login), then retry. Provider secrets are never accepted by tt flags.`,
    );
  }
  return resolved;
}

export function resolveAgentModelDefinition(
  runtime: AgentModelRuntime,
  selection: AgentSelection,
): ResolvedAgentModel {
  const provider = runtime.getProviders().find(
    (candidate) => candidate.id === selection.provider,
  );
  if (!provider) {
    throw new Error(
      `Unknown Pi provider "${selection.provider}". Run \`tt agent models --all\` to list providers and models.`,
    );
  }
  const model = runtime.getModel(selection.provider, selection.model);
  if (!model) {
    throw new Error(
      `Unknown model "${selection.provider}/${selection.model}". Run \`tt agent models --provider ${selection.provider} --all\`.`,
    );
  }
  if (selection.thinking !== "off" && model.reasoning === false) {
    throw new Error(
      `Model "${selection.provider}/${selection.model}" does not support thinking. Configure --thinking off.`,
    );
  }
  return { model, thinking: selection.thinking };
}

export async function createPiModelRuntime(): Promise<ModelRuntime> {
  const agentDir = getAgentDir();
  const modelsPath = join(agentDir, "models.json");
  ensureOpenRouterAppHeaders(modelsPath);
  return await ModelRuntime.create({
    authPath: join(agentDir, "auth.json"),
    modelsPath,
    allowModelNetwork: false,
  });
}

/**
 * OpenRouter attributes requests to an app via the HTTP-Referer/X-Title
 * headers. Without them the OpenRouter dashboard shows tt's agent traffic as
 * "Unknown" app. Pi composes provider-level `headers` from models.json over
 * its built-in openrouter provider, so we merge ours in (preserving any
 * existing user config) before the runtime loads.
 */
const OPENROUTER_APP_HEADERS: Record<string, string> = {
  "HTTP-Referer": "https://tunedtensor.com",
  "X-Title": "Tuned Tensor",
};

function ensureOpenRouterAppHeaders(modelsPath: string): void {
  let config: Record<string, unknown> = {};
  if (existsSync(modelsPath)) {
    try {
      const parsed = JSON.parse(readFileSync(modelsPath, "utf-8"));
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        config = parsed as Record<string, unknown>;
      }
    } catch {
      // Never clobber an unparseable user file.
      return;
    }
  }
  const before = JSON.stringify(config);

  const asRecord = (value: unknown): Record<string, unknown> | undefined =>
    value && typeof value === "object" && !Array.isArray(value)
      ? (value as Record<string, unknown>)
      : undefined;
  const empty = (): Record<string, unknown> => ({});
  const providers = asRecord(config.providers) ?? (config.providers = empty());
  const openrouter = asRecord(providers.openrouter) ?? (providers.openrouter = empty());
  const headers = asRecord(openrouter.headers) ?? (openrouter.headers = empty());
  Object.assign(headers, OPENROUTER_APP_HEADERS);

  if (JSON.stringify(config) === before) return;
  mkdirSync(dirname(modelsPath), { recursive: true });
  writeFileSync(modelsPath, JSON.stringify(config, null, 2) + "\n");
}
