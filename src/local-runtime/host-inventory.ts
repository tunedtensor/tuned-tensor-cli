import { constants } from "node:fs";
import { access, statfs } from "node:fs/promises";
import { cpus, freemem, homedir, totalmem, type } from "node:os";
import { dirname, join, resolve } from "node:path";
import { spawn } from "node:child_process";
import { DEFAULT_ARTIFACT_ROOT, defaultStoreRoot } from "../paths.js";
import type { LocalRunnerConfig } from "./contracts.js";
import {
  minimalMachineLearningEnvironment,
  withHuggingFaceCacheEnvironment,
} from "./huggingface-cache.js";
import {
  buildBundledPythonCommand,
  withBundledPythonEnvironment,
} from "./process-runner.js";

export interface CommandResult {
  code: number | null;
  stdout: string;
  stderr: string;
  error?: string;
}

interface CommandOptions {
  cwd?: string;
  env?: NodeJS.ProcessEnv;
  timeoutMs?: number;
}

async function runCommand(command: string, args: string[], options: CommandOptions = {}): Promise<CommandResult> {
  const timeoutMs = options.timeoutMs ?? 10_000;
  let interrupted: string | undefined;
  const result = await new Promise<CommandResult>((resolveResult) => {
    let settled = false;
    let stdout = "";
    let stderr = "";
    let stopError: string | undefined;
    let forceKillTimer: NodeJS.Timeout | null = null;
    const child = spawn(command, args, {
      cwd: options.cwd,
      env: options.env ?? process.env,
      stdio: ["ignore", "pipe", "pipe"],
      detached: process.platform !== "win32",
    });
    const killProcessGroup = (signal: NodeJS.Signals) => {
      if (child.pid && process.platform !== "win32") {
        try {
          process.kill(-child.pid, signal);
          return;
        } catch {
          // The child may have exited between the check and signal.
        }
      }
      child.kill(signal);
    };
    const requestStop = (signal: NodeJS.Signals = "SIGTERM") => {
      killProcessGroup(signal);
      if (!forceKillTimer) {
        forceKillTimer = setTimeout(() => killProcessGroup("SIGKILL"), 5_000);
        forceKillTimer.unref();
      }
    };
    const onSigint = () => {
      interrupted = "interrupted by SIGINT";
      stopError = interrupted;
      requestStop("SIGINT");
    };
    const onSigterm = () => {
      interrupted = "interrupted by SIGTERM";
      stopError = interrupted;
      requestStop("SIGTERM");
    };
    const finish = (result: CommandResult) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (forceKillTimer) clearTimeout(forceKillTimer);
      process.off("SIGINT", onSigint);
      process.off("SIGTERM", onSigterm);
      resolveResult(result);
    };
    const timer = setTimeout(() => {
      stopError = `timed out after ${timeoutMs}ms`;
      requestStop();
    }, timeoutMs);
    timer.unref();
    process.once("SIGINT", onSigint);
    process.once("SIGTERM", onSigterm);
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", (error) => {
      finish({ code: null, stdout, stderr, error: error.message });
    });
    child.on("close", (code) => {
      finish({ code: stopError ? null : code, stdout, stderr, error: stopError });
    });
  });
  if (interrupted) throw new Error(interrupted);
  return result;
}

function firstLine(value: string): string {
  return value.trim().split(/\r?\n/)[0] ?? "";
}

export interface HostGpu {
  index: number;
  name: string;
  driver_version?: string;
  memory_total_bytes?: number;
  memory_free_bytes?: number;
  compute_cap?: string;
  unified_memory: boolean;
}

export interface HostDisk {
  name: string;
  path: string;
  free_bytes?: number;
  ok: boolean;
  message: string;
}

export interface HostPythonRuntime {
  ok: boolean;
  message: string;
  torch?: string;
  transformers?: string;
  peft?: string;
  cuda_available?: boolean;
  cuda_device?: string;
  compute_capability?: number[];
  total_memory_bytes?: number;
  bf16_supported?: boolean;
}

export interface HostInventory {
  collected_at: string;
  quick: boolean;
  node: { version: string; major: number; ok: boolean };
  os: {
    platform: string;
    arch: string;
    type: string;
    cpu_count: number;
    total_memory_bytes: number;
    free_memory_bytes: number;
  };
  uv?: { ok: boolean; version?: string; message: string };
  gpus: HostGpu[];
  nvidia_smi: { ok: boolean; message: string };
  python?: HostPythonRuntime;
  disks: HostDisk[];
}

export interface CollectHostInventoryOptions {
  config?: LocalRunnerConfig;
  quick?: boolean;
  env?: NodeJS.ProcessEnv;
  cwd?: string;
}

function parseMiB(value: string | undefined): number | undefined {
  if (!value) return undefined;
  const parsed = Number(value.replace(/[^\d.]/g, ""));
  if (!Number.isFinite(parsed) || parsed <= 0) return undefined;
  return Math.round(parsed * 1024 * 1024);
}

function parseNvidiaGpus(stdout: string): HostGpu[] {
  const gpus: HostGpu[] = [];
  for (const line of stdout.split(/\r?\n/).map((item) => item.trim()).filter(Boolean)) {
    if (/^(?:mon|tue|wed|thu|fri|sat|sun)\b/i.test(line) || line.startsWith("+---") || line.startsWith("|")) {
      continue;
    }
    const parts = line.split(",").map((part) => part.trim());
    if (parts.length < 2) continue;
    const index = Number(parts[0]);
    const name = parts[1] ?? "NVIDIA GPU";
    gpus.push({
      index: Number.isFinite(index) ? index : gpus.length,
      name,
      driver_version: parts[2] || undefined,
      memory_total_bytes: parseMiB(parts[3]),
      memory_free_bytes: parseMiB(parts[4]),
      compute_cap: parts[5] || undefined,
      unified_memory: /spark|gb10/i.test(name),
    });
  }
  return gpus;
}

async function diskCheck(name: string, path: string): Promise<HostDisk> {
  const resolvedPath = resolve(path);
  try {
    await access(resolvedPath, constants.R_OK);
    const fs = await statfs(resolvedPath);
    const freeBytes = Number(fs.bavail) * Number(fs.bsize);
    return {
      name,
      path: resolvedPath,
      free_bytes: freeBytes,
      ok: true,
      message: `${resolvedPath} (${Math.round(freeBytes / (1024 ** 3) * 10) / 10} GiB available)`,
    };
  } catch (error) {
    let probe = resolvedPath;
    for (let i = 0; i < 6; i += 1) {
      const parent = dirname(probe);
      if (parent === probe) break;
      probe = parent;
      try {
        const fs = await statfs(probe);
        const freeBytes = Number(fs.bavail) * Number(fs.bsize);
        return {
          name,
          path: resolvedPath,
          free_bytes: freeBytes,
          ok: true,
          message: `${resolvedPath} not created yet; ${probe} has ${Math.round(freeBytes / (1024 ** 3) * 10) / 10} GiB available`,
        };
      } catch {
        continue;
      }
    }
    return {
      name,
      path: resolvedPath,
      ok: false,
      message: `${resolvedPath}: ${error instanceof Error ? error.message : String(error)}`,
    };
  }
}

function pythonProbeSource(): string {
  return [
    "import json",
    "import torch, transformers, peft, huggingface_hub",
    "payload = {'python_ok': True, 'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': getattr(peft, '__version__', None), 'cuda_available': bool(torch.cuda.is_available())}",
    "cuda = torch.cuda.is_available()",
    "payload.update({'cuda_device': torch.cuda.get_device_name(0), 'compute_capability': list(torch.cuda.get_device_capability(0)), 'total_memory_bytes': torch.cuda.get_device_properties(0).total_memory, 'bf16_supported': bool(torch.cuda.is_bf16_supported())} if cuda else {})",
    "print(json.dumps(payload))",
  ].join("; ");
}

export function parsePythonProbeJson(text: string): HostPythonRuntime | undefined {
  const line = [...text.split(/\r?\n/)].reverse().find((item) => item.trim().startsWith("{"));
  if (!line) return undefined;
  try {
    const parsed = JSON.parse(line) as Record<string, unknown>;
    if (parsed.python_ok !== true) return undefined;
    return {
      ok: true,
      message: line.trim(),
      torch: typeof parsed.torch === "string" ? parsed.torch : undefined,
      transformers: typeof parsed.transformers === "string" ? parsed.transformers : undefined,
      peft: typeof parsed.peft === "string" ? parsed.peft : undefined,
      cuda_available: parsed.cuda_available === true,
      cuda_device: typeof parsed.cuda_device === "string" ? parsed.cuda_device : undefined,
      compute_capability: Array.isArray(parsed.compute_capability)
        ? parsed.compute_capability.filter((value): value is number => typeof value === "number")
        : undefined,
      total_memory_bytes: typeof parsed.total_memory_bytes === "number"
        ? parsed.total_memory_bytes
        : undefined,
      bf16_supported: parsed.bf16_supported === true,
    };
  } catch {
    return undefined;
  }
}

async function collectPythonRuntime(
  config: LocalRunnerConfig | undefined,
  env: NodeJS.ProcessEnv,
): Promise<HostPythonRuntime> {
  const entrypoint = buildBundledPythonCommand("-c", [pythonProbeSource()]);
  const childEnv = withBundledPythonEnvironment(
    withHuggingFaceCacheEnvironment(
      minimalMachineLearningEnvironment(env),
      config?.paths.modelCache,
    ),
  );
  const result = await runCommand(entrypoint.command, entrypoint.commandArgs, {
    env: childEnv,
    timeoutMs: 1_800_000,
  });
  if (result.code === 0) {
    return parsePythonProbeJson(result.stdout) ?? {
      ok: true,
      message: firstLine(result.stdout) || "Python runtime probe succeeded",
    };
  }
  return {
    ok: false,
    message: result.error ?? (firstLine(result.stderr) || `uv exited ${result.code}`),
  };
}

export async function collectHostInventory(
  options: CollectHostInventoryOptions = {},
): Promise<HostInventory> {
  const env = options.env ?? process.env;
  const config = options.config;
  const cwd = options.cwd ?? process.cwd();
  const nodeVersion = process.versions.node;
  const nodeMajor = Number(nodeVersion.split(".")[0]);
  const artifactRoot = resolve(cwd, config?.artifactRoot ?? join(cwd, DEFAULT_ARTIFACT_ROOT));
  const storeRoot = resolve(config?.storeRoot ?? defaultStoreRoot(env));
  const modelCache = resolve(
    config?.paths.modelCache ?? env.HF_HOME ?? join(homedir(), ".cache", "huggingface"),
  );

  const [artifactDisk, storeDisk, cacheDisk, uvVersion, nvidiaQueried] = await Promise.all([
    diskCheck("artifact-root", artifactRoot),
    diskCheck("store-root", storeRoot),
    diskCheck("model-cache", modelCache),
    runCommand("uv", ["--version"], { env, timeoutMs: 10_000 }),
    runCommand("nvidia-smi", [
      "--query-gpu=index,name,driver_version,memory.total,memory.free,compute_cap",
      "--format=csv,noheader,nounits",
    ], { env, timeoutMs: 10_000 }),
  ]);

  let nvidia = nvidiaQueried;
  let gpus = nvidiaQueried.code === 0 ? parseNvidiaGpus(nvidiaQueried.stdout) : [];
  if (nvidiaQueried.code !== 0 || gpus.length === 0) {
    const fallback = await runCommand("nvidia-smi", [], { env, timeoutMs: 10_000 });
    if (fallback.code === 0) {
      nvidia = fallback;
      if (gpus.length === 0) {
        const nameLine = firstLine(fallback.stdout);
        if (nameLine) {
          gpus = [{
            index: 0,
            name: nameLine,
            unified_memory: /spark|gb10/i.test(nameLine),
          }];
        }
      }
    } else if (nvidiaQueried.code !== 0) {
      nvidia = fallback;
    }
  }

  const inventory: HostInventory = {
    collected_at: new Date().toISOString(),
    quick: options.quick === true,
    node: {
      version: nodeVersion,
      major: nodeMajor,
      ok: nodeMajor >= 22,
    },
    os: {
      platform: process.platform,
      arch: process.arch,
      type: type(),
      cpu_count: cpus().length,
      total_memory_bytes: totalmem(),
      free_memory_bytes: freemem(),
    },
    uv: {
      ok: uvVersion.code === 0,
      version: uvVersion.code === 0 ? firstLine(uvVersion.stdout) : undefined,
      message: uvVersion.code === 0
        ? firstLine(uvVersion.stdout)
        : uvVersion.error ?? (firstLine(uvVersion.stderr) || "uv is not available"),
    },
    gpus,
    nvidia_smi: {
      ok: nvidia.code === 0,
      message: nvidia.code === 0
        ? (gpus[0] ? `${gpus[0].name}${gpus[0].memory_total_bytes ? ` (${Math.round((gpus[0].memory_total_bytes) / (1024 ** 3))} GiB)` : ""}` : firstLine(nvidia.stdout) || "nvidia-smi reported a GPU")
        : nvidia.error ?? (firstLine(nvidia.stderr) || "nvidia-smi not available"),
    },
    disks: [artifactDisk, storeDisk, cacheDisk],
  };

  if (options.quick !== true) {
    inventory.python = uvVersion.code === 0
      ? await collectPythonRuntime(config, env)
      : { ok: false, message: "Python probe skipped because uv is not available" };
  }

  return inventory;
}
