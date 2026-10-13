import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { loadLocalRunInput, type LocalRunInput } from "./local-project.js";
import { parseLocalRunnerConfig } from "./orchestrator.js";
import type { LocalBehaviorSpecFile, LocalRunnerConfig } from "./contracts.js";
import { DEFAULT_PROJECT_STORE_ROOT, expandUserPath } from "../paths.js";

export function configForSpec(spec: LocalBehaviorSpecFile, specPath: string): LocalRunnerConfig {
  return parseLocalRunnerConfig({
    storeRoot: process.env.TT_LOCAL_HOME?.trim() ? expandUserPath(process.env.TT_LOCAL_HOME.trim()) : DEFAULT_PROJECT_STORE_ROOT,
    ...spec.runtime,
    ...(spec.evaluation ? { evaluation: spec.evaluation } : {}),
  }, specPath);
}

/** Direct downloads can run before a valid adapter spec exists. */
export async function resolveDownloadConfig(specPath: string, configPath?: string, warn?: (message: string) => void): Promise<LocalRunnerConfig> {
  let input: LocalRunInput | undefined;
  try {
    if (existsSync(specPath)) input = await loadLocalRunInput(specPath);
  } catch (error) {
    // Invalid documents cannot contribute settings, but I/O failures should remain visible.
    if ((error as NodeJS.ErrnoException).code) throw error;
  }
  if (input?.kind === "spec") return (await resolveProjectConfig(specPath, configPath, warn, input)).config;
  return configPath
    ? parseLocalRunnerConfig(JSON.parse(await readFile(configPath, "utf8")), resolve(configPath))
    : parseLocalRunnerConfig({});
}

export async function resolveProjectConfig(specPath: string, configPath?: string, warn?: (message: string) => void, loaded?: LocalRunInput): Promise<{ config: LocalRunnerConfig; legacyConfigPath?: string }> {
  const input = loaded ?? (existsSync(specPath) ? await loadLocalRunInput(specPath) : undefined);
  const spec = input && input.kind !== "request" ? input.spec : undefined;
  // Approval snapshots already contain the reviewed, resolved configuration.
  // Do not reopen an adjacent mutable legacy config in the child process.
  if (!configPath && spec?.runtime && /^\..+\.tt-agent-[0-9a-f-]+\.json$/.test(basename(specPath))) {
    return { config: configForSpec(spec, specPath) };
  }
  const candidate = configPath ? resolve(configPath) : join(dirname(resolve(specPath)), "local-runner.json");
  if (configPath || existsSync(candidate)) {
    const legacy = parseLocalRunnerConfig(JSON.parse(await readFile(candidate, "utf8")), candidate);
    if (spec?.runtime || spec?.evaluation) {
      throw new Error("Conflicting configuration sources: tunedtensor.json contains runtime/evaluation settings and a legacy runner config is present. Migrate to tunedtensor.json; do not silently override the spec.");
    }
    warn?.("Legacy local-runner.json detected; migrate its settings into tunedtensor.json with tt pipeline migrate.");
    return { config: legacy, legacyConfigPath: candidate };
  }
  return { config: spec ? configForSpec(spec, specPath) : parseLocalRunnerConfig({}) };
}
