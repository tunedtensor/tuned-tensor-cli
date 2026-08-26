import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { isAbsolute, join, resolve } from "node:path";

/** Home and project state share this directory name. */
export const TUNED_TENSOR_DIR = ".tuned-tensor";

export const DEFAULT_ARTIFACT_ROOT = `${TUNED_TENSOR_DIR}/artifacts`;
export const DEFAULT_PROJECT_STORE_ROOT = `${TUNED_TENSOR_DIR}/store`;
export const DEFAULT_FOUNDATION_RUNS_DIR = `${TUNED_TENSOR_DIR}/foundation-runs`;

function envHome(env: NodeJS.ProcessEnv = process.env): string {
  const home = env.HOME?.trim();
  return home ? resolve(home) : homedir();
}

export function expandUserPath(
  value: string,
  env: NodeJS.ProcessEnv = process.env,
  baseDirectory = process.cwd(),
): string {
  const trimmed = value.trim();
  if (trimmed === "~") return envHome(env);
  if (trimmed.startsWith("~/")) return resolve(envHome(env), trimmed.slice(2));
  return isAbsolute(trimmed) ? resolve(trimmed) : resolve(baseDirectory, trimmed);
}

/** Laptop-local parent: config, agent, store, and uv caches. */
export function getTunedTensorHome(env: NodeJS.ProcessEnv = process.env): string {
  const override = env.TUNED_TENSOR_HOME?.trim();
  if (override) return expandUserPath(override, env);
  return join(envHome(env), TUNED_TENSOR_DIR);
}

export function getConfigDir(env: NodeJS.ProcessEnv = process.env): string {
  return getTunedTensorHome(env);
}

export function getAgentConfigDir(env: NodeJS.ProcessEnv = process.env): string {
  return join(getTunedTensorHome(env), "agent");
}

export function defaultStoreRoot(env: NodeJS.ProcessEnv = process.env): string {
  const override = env.TT_LOCAL_HOME?.trim();
  if (override) return expandUserPath(override, env);
  const next = join(getTunedTensorHome(env), "store");
  if (env.TUNED_TENSOR_HOME?.trim()) return next;
  const legacy = join(envHome(env), ".tuned-tensor-local");
  if (!existsSync(next) && existsSync(legacy)) return legacy;
  return next;
}

export function defaultCacheRoot(env: NodeJS.ProcessEnv = process.env): string {
  return join(getTunedTensorHome(env), "cache");
}

export function pythonEnvironmentPath(
  kind: "uv" | "uv-foundation",
  hash: string,
  env: NodeJS.ProcessEnv = process.env,
): string {
  const next = join(defaultCacheRoot(env), kind, hash);
  if (env.TUNED_TENSOR_HOME?.trim()) return next;
  const cacheHome = env.XDG_CACHE_HOME?.trim()
    ? resolve(env.XDG_CACHE_HOME)
    : join(envHome(env), ".cache");
  const legacy = join(cacheHome, "tuned-tensor-local", kind, hash);
  if (!existsSync(next) && existsSync(legacy)) return legacy;
  return next;
}
