import {
  chmodSync,
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

export interface Config {
  api_key?: string;
  base_url?: string;
  agent?: AgentSelection;
}

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

const CONFIG_DIR_NAME = "tuned-tensor";
const CONFIG_FILE_NAME = "config.json";

export function getConfigDir(): string {
  const xdg = process.env.XDG_CONFIG_HOME;
  const base = xdg || join(homedir(), ".config");
  return join(base, CONFIG_DIR_NAME);
}

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
  const current = readConfig();
  writeConfig({ ...current, ...partial });
}

export function clearConfig(): void {
  const path = getConfigPath();
  if (existsSync(path)) writeConfig({});
}

export const DEFAULT_BASE_URL = "https://tunedtensor.com";

export function getBaseUrl(opts?: { baseUrl?: string }): string {
  return opts?.baseUrl || process.env.TUNED_TENSOR_URL || readConfig().base_url || DEFAULT_BASE_URL;
}

export function getApiKey(opts?: { apiKey?: string }): string | undefined {
  return opts?.apiKey || process.env.TUNED_TENSOR_API_KEY || readConfig().api_key;
}

export function getAgentSelection(
  env: Readonly<NodeJS.ProcessEnv> = process.env,
): AgentSelection | undefined {
  const stored = readConfig().agent;
  const provider = env.TUNED_TENSOR_AGENT_PROVIDER || stored?.provider;
  const model = env.TUNED_TENSOR_AGENT_MODEL || stored?.model;
  const rawThinking = env.TUNED_TENSOR_AGENT_THINKING || stored?.thinking || "medium";
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
