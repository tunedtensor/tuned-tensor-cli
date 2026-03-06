import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

export interface Config {
  api_key?: string;
  base_url?: string;
}

const CONFIG_DIR_NAME = "tuned-tensor";
const CONFIG_FILE_NAME = "config.json";

function getConfigDir(): string {
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

export function writeConfig(config: Config): void {
  const dir = getConfigDir();
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  writeFileSync(getConfigPath(), JSON.stringify(config, null, 2) + "\n");
}

export function updateConfig(partial: Partial<Config>): void {
  const current = readConfig();
  writeConfig({ ...current, ...partial });
}

export function clearConfig(): void {
  const path = getConfigPath();
  if (existsSync(path)) writeFileSync(path, "{}\n");
}

export const DEFAULT_BASE_URL = "https://www.tunedtensor.com";

export function getBaseUrl(opts?: { baseUrl?: string }): string {
  return opts?.baseUrl || process.env.TUNED_TENSOR_URL || readConfig().base_url || DEFAULT_BASE_URL;
}

export function getApiKey(opts?: { apiKey?: string }): string | undefined {
  return opts?.apiKey || process.env.TUNED_TENSOR_API_KEY || readConfig().api_key;
}
