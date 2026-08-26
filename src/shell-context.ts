import { readFile, readdir, stat } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import {
  DEFAULT_ARTIFACT_ROOT,
  defaultStoreRoot,
  expandUserPath,
  getConfigDir,
} from "./paths.js";
import { hardenExistingLocalStore } from "./local-runtime/store.js";

export type TargetSource = "default-local";

export interface ShellSpecContext {
  path: string;
  id?: string;
  name?: string;
  baseModel?: string;
  exampleCount?: number;
  parseError: boolean;
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

export interface ShellAgentContext {
  provider?: string;
  model?: string;
  thinking?: string;
}

export interface ShellContext {
  cwd: string;
  projectName: string;
  spec?: ShellSpecContext;
  local: ShellLocalContext;
  agent?: ShellAgentContext;
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
  env: Readonly<NodeJS.ProcessEnv>,
): string {
  return expandUserPath(value, env, baseDirectory);
}

function configPath(env: Readonly<NodeJS.ProcessEnv>): string {
  return join(getConfigDir(env), "config.json");
}

function agentSelectionFrom(
  env: Readonly<NodeJS.ProcessEnv>,
  configAgent: unknown,
): ShellAgentContext | undefined {
  const stored = configAgent && typeof configAgent === "object" && !Array.isArray(configAgent)
    ? configAgent as Record<string, unknown>
    : {};
  const provider = env.TUNED_TENSOR_AGENT_PROVIDER?.trim() || stringField(stored, "provider");
  const model = env.TUNED_TENSOR_AGENT_MODEL?.trim() || stringField(stored, "model");
  const thinking = env.TUNED_TENSOR_AGENT_THINKING?.trim() || stringField(stored, "thinking");
  if (!provider && !model && !thinking) return undefined;
  return { provider, model, thinking };
}

export function redactApiKey(key: string | undefined): string | undefined {
  if (!key) return undefined;
  if (key.length <= 4) return "…";
  const prefixLength = key.length > 8
    ? 8
    : Math.max(1, Math.floor(key.length / 2));
  return `${key.slice(0, prefixLength)}…`;
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
 * the host, or creating project/store directories. Existing local-store
 * permissions are repaired before its state is read.
 */
export async function discoverShellContext(
  options: DiscoverShellContextOptions = {},
): Promise<ShellContext> {
  const env = options.env ?? process.env;
  const cwd = resolve(options.cwd ?? process.cwd());
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
    ? expandPath(configuredArtifactRoot, localConfigDirectory, env)
    : resolve(cwd, DEFAULT_ARTIFACT_ROOT);
  const storeRoot = configuredStoreRoot
    ? expandPath(configuredStoreRoot, localConfigDirectory, env)
    : defaultStoreRoot(env);

  const storedConfigPath = configPath(env);
  const storedConfigJson = await readJsonObject(storedConfigPath);
  if (storedConfigJson.invalid) {
    warnings.push("The Tuned Tensor config could not be parsed.");
  }
  const agent = agentSelectionFrom(env, storedConfigJson.value?.agent);

  await hardenExistingLocalStore(storeRoot);
  const [activeModelId, latestRun] = await Promise.all([
    readActiveModelId(storeRoot),
    readLatestRun(storeRoot),
  ]);

  return {
    cwd,
    projectName: basename(cwd) || cwd,
    spec,
    local: {
      configPath: hasAdjacentLocalConfig ? localConfigPath : undefined,
      artifactRoot,
      storeRoot,
      activeModelId,
      latestRun,
    },
    agent,
    warnings,
  };
}

function valueOrDash(value: string | number | undefined): string {
  return value === undefined || value === "" ? "—" : String(value);
}

function agentLabel(context: ShellContext): string {
  const agent = context.agent;
  if (!agent?.provider && !agent?.model) return "not configured";
  const id = [agent.provider, agent.model].filter(Boolean).join("/");
  return agent.thinking ? `${id} (thinking ${agent.thinking})` : id;
}

export function formatShellContext(context: ShellContext): string[] {
  const specLabel = context.spec
    ? context.spec.parseError
      ? `${context.spec.path} (invalid JSON)`
      : context.spec.path
    : "not found";
  const lines = [
    `Project        ${context.projectName}`,
    `Directory      ${context.cwd}`,
    `Spec           ${specLabel}`,
    `Spec ID        ${valueOrDash(context.spec?.id)}`,
    `Base model     ${valueOrDash(context.spec?.baseModel)}`,
    `Examples       ${valueOrDash(context.spec?.exampleCount)}`,
    `Agent model    ${agentLabel(context)}`,
    `Local config   ${context.local.configPath ?? "not found"}`,
    `Artifact root  ${context.local.artifactRoot}`,
    `Store root     ${context.local.storeRoot}`,
  ];
  for (const warning of context.warnings) lines.push(`Warning        ${warning}`);
  return lines;
}

export function formatShellStatus(context: ShellContext): string[] {
  return [
    `Spec           ${context.spec?.name ?? (context.spec ? "unnamed" : "not found")}`,
    `Agent model    ${agentLabel(context)}`,
    `Local config   ${context.local.configPath ?? "not found"}`,
    `Active model   ${context.local.activeModelId ?? "base"}`,
    `Latest run     ${context.local.latestRun
      ? `${context.local.latestRun.id} (${context.local.latestRun.status ?? "unknown"})`
      : "none"}`,
    "Host checks    not run (use doctor)",
  ];
}
