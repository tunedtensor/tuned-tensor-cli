import { ModelRuntime } from "@earendil-works/pi-coding-agent";
import { chmodSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { getAgentConfigDir, type AgentSelection, type AgentThinkingLevel } from "./config.js";

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
  setRuntimeApiKey?(provider: string, apiKey: string): Promise<void>;
}

export function missingProviderAuthMessage(provider: string): string {
  return `Provider "${provider}" is not authenticated. Open the tt shell and run /login ${provider} to save a key, then try again.`;
}

export function unknownAgentProviderMessage(provider: string): string {
  return `Unknown provider "${provider}". Use /login <id> or /model <id>.`;
}

const MIN_REDACTED_SECRET_LENGTH = 8;

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function pushSecret(secrets: string[], value: unknown): void {
  if (typeof value !== "string") return;
  const secret = value.trim();
  if (secret.length < MIN_REDACTED_SECRET_LENGTH) return;
  if (!secrets.includes(secret)) secrets.push(secret);
}

/** API keys and OAuth tokens from TT's agent auth file, for thread redaction. */
export function readStoredProviderSecrets(authPath = getAgentAuthPath()): string[] {
  if (!existsSync(authPath)) return [];
  let parsed: unknown;
  try {
    parsed = JSON.parse(readFileSync(authPath, "utf-8"));
  } catch {
    return [];
  }
  if (!isRecord(parsed)) return [];
  const secrets: string[] = [];
  for (const credential of Object.values(parsed)) {
    if (!isRecord(credential)) continue;
    if (credential.type === "api_key") {
      pushSecret(secrets, credential.key);
    } else if (credential.type === "oauth") {
      pushSecret(secrets, credential.access);
      pushSecret(secrets, credential.refresh);
    }
  }
  return secrets;
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
    throw new Error(missingProviderAuthMessage(selection.provider));
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
      `Unknown provider "${selection.provider}". Run \`tt agent models --all\` to list providers and models.`,
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

export function getAgentAuthPath(): string {
  return join(getAgentConfigDir(), "auth.json");
}

export function getAgentModelsPath(): string {
  return join(getAgentConfigDir(), "models.json");
}

export async function createPiModelRuntime(): Promise<ModelRuntime> {
  const agentDir = getAgentConfigDir();
  ensurePrivateDir(agentDir);
  const modelsPath = getAgentModelsPath();
  const authPath = getAgentAuthPath();
  ensureOpenRouterAppHeaders(modelsPath);
  const runtime = await ModelRuntime.create({
    authPath,
    modelsPath,
    allowModelNetwork: false,
  });
  if (existsSync(authPath)) chmodSync(authPath, 0o600);
  return runtime;
}

/**
 * OpenRouter attributes requests to an app via HTTP-Referer and
 * X-OpenRouter-Title. Keep legacy X-Title for backwards compatibility. tt
 * composes provider-level `headers` from models.json over Pi's built-in
 * OpenRouter attribution, so we merge ours in (preserving any existing user
 * config) before the runtime loads.
 */
const OPENROUTER_APP_HEADERS: Record<string, string> = {
  "HTTP-Referer": "https://tunedtensor.com",
  "X-OpenRouter-Title": "Tuned Tensor",
  "X-OpenRouter-Categories": "cli-agent",
  "X-Title": "Tuned Tensor",
};

function ensurePrivateDir(dir: string): void {
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true, mode: 0o700 });
  chmodSync(dir, 0o700);
}

function writePrivateJson(path: string, value: unknown): void {
  ensurePrivateDir(dirname(path));
  writeFileSync(path, JSON.stringify(value, null, 2) + "\n", { mode: 0o600 });
  chmodSync(path, 0o600);
}

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
  const attributionNames = new Set(
    Object.keys(OPENROUTER_APP_HEADERS).map((name) => name.toLowerCase()),
  );
  for (const name of Object.keys(headers)) {
    if (attributionNames.has(name.toLowerCase())) delete headers[name];
  }
  Object.assign(headers, OPENROUTER_APP_HEADERS);

  if (JSON.stringify(config) === before) return;
  writePrivateJson(modelsPath, config);
}
