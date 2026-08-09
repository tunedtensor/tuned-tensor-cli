import {
  getAgentDir,
  ModelRuntime,
} from "@earendil-works/pi-coding-agent";
import { join } from "node:path";
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
  return await ModelRuntime.create({
    authPath: join(agentDir, "auth.json"),
    modelsPath: join(agentDir, "models.json"),
    allowModelNetwork: false,
  });
}
