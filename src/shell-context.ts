import { readFile, readdir, stat } from "node:fs/promises";
import { homedir } from "node:os";
import { basename, dirname, isAbsolute, join, resolve } from "node:path";
import type { WorkflowMode } from "./command-catalog.js";

export type TargetSource =
  | "environment"
  | "adjacent-config"
  | "default-cloud";

export interface ShellSpecContext {
  path: string;
  id?: string;
  name?: string;
  baseModel?: string;
  exampleCount?: number;
  parseError: boolean;
}

export interface ShellCloudContext {
  authenticated: boolean;
  keyPrefix?: string;
  baseUrl: string;
  configPath: string;
  configFound: boolean;
}

export interface ShellLatestRun {
  id: string;
  status?: string;
  specName?: string;
  updatedAt?: string;
}

export interface ShellLocalContext {
  configPath?: string;
  artifactRoot: string;
  storeRoot: string;
  activeModelId?: string;
  latestRun?: ShellLatestRun;
}

export interface ShellContext {
  cwd: string;
  projectName: string;
  inferredTarget: WorkflowMode;
  targetSource: TargetSource;
  spec?: ShellSpecContext;
  cloud: ShellCloudContext;
  local: ShellLocalContext;
  warnings: string[];
}

export interface DiscoverShellContextOptions {
  cwd?: string;
  env?: Readonly<NodeJS.ProcessEnv>;
}

interface JsonReadResult {
  found: boolean;
  value?: Record<string, unknown>;
  invalid: boolean;
}

async function readJsonObject(path: string): Promise<JsonReadResult> {
  try {
    const parsed = JSON.parse(await readFile(path, "utf8")) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return { found: true, invalid: true };
    }
    return {
      found: true,
      value: parsed as Record<string, unknown>,
      invalid: false,
    };
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return { found: false, invalid: false };
    }
    return { found: true, invalid: true };
  }
}

function stringField(
  value: Record<string, unknown> | undefined,
  key: string,
): string | undefined {
  const field = value?.[key];
  return typeof field === "string" && field.trim() ? field : undefined;
}

function expandPath(
  value: string,
  baseDirectory: string,
  homeDirectory: string,
): string {
  if (value === "~") return homeDirectory;
  if (value.startsWith("~/")) return resolve(homeDirectory, value.slice(2));
  return isAbsolute(value) ? resolve(value) : resolve(baseDirectory, value);
}

function environmentHome(env: Readonly<NodeJS.ProcessEnv>): string {
  return env.HOME ? resolve(env.HOME) : homedir();
}

function cloudConfigPath(
  env: Readonly<NodeJS.ProcessEnv>,
  homeDirectory: string,
): string {
  const configHome = env.XDG_CONFIG_HOME
    ? resolve(env.XDG_CONFIG_HOME)
    : join(homeDirectory, ".config");
  return join(configHome, "tuned-tensor", "config.json");
}

export function redactApiKey(key: string | undefined): string | undefined {
  if (!key) return undefined;
  if (key.length <= 4) return "…";
  const prefixLength = key.length > 8
    ? 8
    : Math.max(1, Math.floor(key.length / 2));
  return `${key.slice(0, prefixLength)}…`;
}

export function targetFromEnvironment(
  env: Readonly<NodeJS.ProcessEnv>,
): WorkflowMode | undefined {
  const value = env.TT_TARGET?.trim().toLowerCase();
  return value === "cloud" || value === "local" ? value : undefined;
}

async function adjacentFile(path: string): Promise<boolean> {
  try {
    return (await stat(path)).isFile();
  } catch {
    return false;
  }
}

async function readActiveModelId(storeRoot: string): Promise<string | undefined> {
  const result = await readJsonObject(join(storeRoot, "active-model.json"));
  if (!result.found || result.invalid) return undefined;
  const modelId = result.value?.model_id;
  if (modelId === null) return "base";
  return typeof modelId === "string" && modelId ? modelId : undefined;
}

async function readLatestRun(storeRoot: string): Promise<ShellLatestRun | undefined> {
  const runsRoot = join(storeRoot, "runs");
  let names: string[];
  try {
    names = (await readdir(runsRoot, { withFileTypes: true }))
      .filter((entry) => entry.isDirectory())
      .map((entry) => entry.name);
  } catch {
    return undefined;
  }

  const states = await Promise.all(names.map(async (name): Promise<ShellLatestRun | undefined> => {
    const result = await readJsonObject(join(runsRoot, name, "state.json"));
    if (!result.found || result.invalid || !result.value) return undefined;
    const id = stringField(result.value, "id") ?? name;
    const state: ShellLatestRun = {
      id,
      status: stringField(result.value, "status"),
      specName: stringField(result.value, "spec_name"),
      updatedAt: stringField(result.value, "updated_at"),
    };
    return state;
  }));

  return states
    .filter((state): state is ShellLatestRun => Boolean(state))
    .sort((left, right) => {
      const leftUpdated = left.updatedAt ?? "";
      const rightUpdated = right.updatedAt ?? "";
      if (leftUpdated !== rightUpdated) {
        return rightUpdated.localeCompare(leftUpdated);
      }
      return right.id.localeCompare(left.id);
    })[0];
}

/**
 * Discover lightweight shell context without contacting the network, probing
 * the host, or creating project/store directories.
 */
export async function discoverShellContext(
  options: DiscoverShellContextOptions = {},
): Promise<ShellContext> {
  const env = options.env ?? process.env;
  const cwd = resolve(options.cwd ?? process.cwd());
  const homeDirectory = environmentHome(env);
  const warnings: string[] = [];

  const specPath = join(cwd, "tunedtensor.json");
  const specJson = await readJsonObject(specPath);
  let spec: ShellSpecContext | undefined;
  if (specJson.found) {
    if (specJson.invalid || !specJson.value) {
      spec = { path: specPath, parseError: true };
      warnings.push("tunedtensor.json could not be parsed.");
    } else {
      const examples = specJson.value.examples;
      spec = {
        path: specPath,
        id: stringField(specJson.value, "id"),
        name: stringField(specJson.value, "name"),
        baseModel: stringField(specJson.value, "base_model"),
        exampleCount: Array.isArray(examples) ? examples.length : undefined,
        parseError: false,
      };
    }
  }

  const localConfigPath = join(cwd, "local-runner.json");
  const hasAdjacentLocalConfig = await adjacentFile(localConfigPath);
  const localConfigJson = hasAdjacentLocalConfig
    ? await readJsonObject(localConfigPath)
    : { found: false, invalid: false } satisfies JsonReadResult;
  if (localConfigJson.invalid) {
    warnings.push("local-runner.json could not be parsed.");
  }

  const localConfigDirectory = hasAdjacentLocalConfig
    ? dirname(localConfigPath)
    : cwd;
  const configuredArtifactRoot = stringField(localConfigJson.value, "artifactRoot");
  const configuredStoreRoot = stringField(localConfigJson.value, "storeRoot");
  const artifactRoot = configuredArtifactRoot
    ? expandPath(configuredArtifactRoot, localConfigDirectory, homeDirectory)
    : resolve(cwd, ".tt-local", "artifacts");
  const storeRoot = configuredStoreRoot
    ? expandPath(configuredStoreRoot, localConfigDirectory, homeDirectory)
    : env.TT_LOCAL_HOME
      ? expandPath(env.TT_LOCAL_HOME, cwd, homeDirectory)
      : join(homeDirectory, ".tuned-tensor-local");

  const cloudPath = cloudConfigPath(env, homeDirectory);
  const cloudConfigJson = await readJsonObject(cloudPath);
  if (cloudConfigJson.invalid) {
    warnings.push("The Tuned Tensor cloud config could not be parsed.");
  }
  const environmentApiKey = env.TUNED_TENSOR_API_KEY?.trim();
  const storedApiKey = stringField(cloudConfigJson.value, "api_key");
  const apiKey = environmentApiKey || storedApiKey;
  const baseUrl = env.TUNED_TENSOR_URL?.trim()
    || stringField(cloudConfigJson.value, "base_url")
    || "https://tunedtensor.com";

  const environmentTarget = targetFromEnvironment(env);
  if (env.TT_TARGET && !environmentTarget) {
    warnings.push(`Ignoring invalid TT_TARGET=${JSON.stringify(env.TT_TARGET)}; use cloud or local.`);
  }
  const inferredTarget = environmentTarget
    ?? (hasAdjacentLocalConfig ? "local" : "cloud");
  const targetSource: TargetSource = environmentTarget
    ? "environment"
    : hasAdjacentLocalConfig
      ? "adjacent-config"
      : "default-cloud";

  const [activeModelId, latestRun] = await Promise.all([
    readActiveModelId(storeRoot),
    readLatestRun(storeRoot),
  ]);

  return {
    cwd,
    projectName: basename(cwd) || cwd,
    inferredTarget,
    targetSource,
    spec,
    cloud: {
      authenticated: Boolean(apiKey),
      keyPrefix: redactApiKey(apiKey),
      baseUrl,
      configPath: cloudPath,
      configFound: cloudConfigJson.found,
    },
    local: {
      configPath: hasAdjacentLocalConfig ? localConfigPath : undefined,
      artifactRoot,
      storeRoot,
      activeModelId,
      latestRun,
    },
    warnings,
  };
}

function valueOrDash(value: string | number | undefined): string {
  return value === undefined || value === "" ? "—" : String(value);
}

function targetSourceLabel(source: TargetSource): string {
  switch (source) {
    case "environment":
      return "TT_TARGET";
    case "adjacent-config":
      return "adjacent local-runner.json";
    default:
      return "default";
  }
}

export function formatShellContext(
  context: ShellContext,
  selectedTarget: WorkflowMode,
  targetSource: TargetSource | "session" | "command-option",
): string[] {
  const specLabel = context.spec
    ? context.spec.parseError
      ? `${context.spec.path} (invalid JSON)`
      : context.spec.path
    : "not found";
  const sourceLabel = targetSource === "session"
    ? "session"
    : targetSource === "command-option"
      ? "--target"
      : targetSourceLabel(targetSource);
  const lines = [
    `Target         ${selectedTarget} (${sourceLabel})`,
    `Project        ${context.projectName}`,
    `Directory      ${context.cwd}`,
    `Spec           ${specLabel}`,
    `Spec ID        ${valueOrDash(context.spec?.id)}`,
    `Base model     ${valueOrDash(context.spec?.baseModel)}`,
    `Examples       ${valueOrDash(context.spec?.exampleCount)}`,
    `Cloud endpoint ${context.cloud.baseUrl}`,
    `Cloud auth     ${context.cloud.authenticated ? `yes (${context.cloud.keyPrefix})` : "no"}`,
    `Local config   ${context.local.configPath ?? "not found"}`,
    `Artifact root  ${context.local.artifactRoot}`,
    `Store root     ${context.local.storeRoot}`,
  ];
  for (const warning of context.warnings) lines.push(`Warning        ${warning}`);
  return lines;
}

export function formatShellStatus(
  context: ShellContext,
  selectedTarget: WorkflowMode,
): string[] {
  const lines = [
    `Workflow       ${selectedTarget}`,
    `Spec           ${context.spec?.name ?? (context.spec ? "unnamed" : "not found")}`,
  ];
  if (selectedTarget === "cloud") {
    lines.push(
      `Authentication ${context.cloud.authenticated ? `ready (${context.cloud.keyPrefix})` : "not configured"}`,
      `Endpoint       ${context.cloud.baseUrl}`,
      "Remote status  not queried",
    );
  } else {
    lines.push(
      `Local config   ${context.local.configPath ?? "not found"}`,
      `Active model   ${context.local.activeModelId ?? "base"}`,
      `Latest run     ${context.local.latestRun
        ? `${context.local.latestRun.id} (${context.local.latestRun.status ?? "unknown"})`
        : "none"}`,
      "Host checks    not run (use doctor)",
    );
  }
  return lines;
}
