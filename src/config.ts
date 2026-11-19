import {
  chmodSync,
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { join } from "node:path";
import { getConfigDir } from "./paths.js";

export { getAgentConfigDir, getConfigDir } from "./paths.js";

export interface Config {
  agent?: AgentSelection;
}

/** Keys written by earlier releases for the retired hosted service. */
const RETIRED_CONFIG_KEYS = ["api_key", "base_url"] as const;

export const AGENT_THINKING_LEVELS = [
  "off",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
] as const;

export type AgentThinkingLevel = typeof AGENT_THINKING_LEVELS[number];

export interface AgentSelection {
  provider: string;
  model: string;
  thinking: AgentThinkingLevel;
}

/**
 * Earlier releases offered hosted inference as `tunedtensor/managed`. That
 * service is retired, so a saved selection of it counts as unconfigured.
 */
export const RETIRED_MANAGED_PROVIDER = "tunedtensor";
export const RETIRED_MANAGED_MODEL = "managed";

export function isRetiredManagedSelection(provider: string | undefined, model?: string): boolean {
  return provider === RETIRED_MANAGED_PROVIDER && (model === undefined || model === RETIRED_MANAGED_MODEL);
}

const CONFIG_FILE_NAME = "config.json";

function getConfigPath(): string {
  return join(getConfigDir(), CONFIG_FILE_NAME);
}

export function readConfig(): Config {
  const path = getConfigPath();
  if (!existsSync(path)) return {};
  try {
    return JSON.parse(readFileSync(path, "utf-8"));
  } catch {
    return {};
  }
}

let configRevision = 0;

/**
 * Monotonic counter bumped on every local config write, so in-process caches
 * can detect selection changes without re-reading the file each time.
 */
export function getConfigRevision(): number {
  return configRevision;
}

export function writeConfig(config: Config): void {
  const dir = getConfigDir();
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true, mode: 0o700 });
  chmodSync(dir, 0o700);
  const path = getConfigPath();
  writeFileSync(path, JSON.stringify(config, null, 2) + "\n", {
    mode: 0o600,
  });
  chmodSync(path, 0o600);
  configRevision += 1;
}

export function updateConfig(partial: Partial<Config>): void {
  const current: Record<string, unknown> = { ...readConfig() };
  // Drop the retired hosted-service token rather than keep an unused credential on disk.
  for (const key of RETIRED_CONFIG_KEYS) delete current[key];
  writeConfig({ ...current, ...partial } as Config);
}

export function clearConfig(): void {
  const path = getConfigPath();
  if (existsSync(path)) writeConfig({});
}

export function getAgentSelection(
  env: Readonly<NodeJS.ProcessEnv> = process.env,
): AgentSelection | undefined {
  const saved = readConfig().agent;
  const stored = saved && isRetiredManagedSelection(saved.provider, saved.model) ? undefined : saved;
  const provider = env.TUNED_TENSOR_AGENT_PROVIDER || stored?.provider;
  const model = env.TUNED_TENSOR_AGENT_MODEL || stored?.model;
  if (isRetiredManagedSelection(provider, model)) {
    throw new Error(
      "Tuned Tensor managed inference has been retired. Choose your own provider with /login and /model, or `tt agent configure --provider <provider> --model <model>`.",
    );
  }
  const rawThinking = env.TUNED_TENSOR_AGENT_THINKING
    || (stored?.provider === provider ? stored?.thinking : undefined)
    || "medium";
  if (!AGENT_THINKING_LEVELS.includes(rawThinking as AgentThinkingLevel)) {
    throw new Error(
      "Agent thinking must be off, minimal, low, medium, high, xhigh, or max.",
    );
  }
  if (!provider && !model) return undefined;
  if (!provider || !model) {
    throw new Error(
      "Both an agent provider and model are required. Run `tt agent configure --provider <provider> --model <model>`.",
    );
  }
  return { provider, model, thinking: rawThinking as AgentThinkingLevel };
}
