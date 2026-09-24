import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
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

export async function resolveProjectConfig(specPath: string, configPath?: string, warn?: (message: string) => void, loaded?: LocalRunInput): Promise<{ config: LocalRunnerConfig; legacyConfigPath?: string }> {
  const input = loaded ?? (existsSync(specPath) ? await loadLocalRunInput(specPath) : undefined);
  const spec = input && input.kind !== "request" ? input.spec : undefined;
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
