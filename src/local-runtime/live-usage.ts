import { execFile } from "node:child_process";
import { readFileSync } from "node:fs";
import { statfs } from "node:fs/promises";
import { cpus, freemem, homedir, loadavg, totalmem } from "node:os";
import { dirname, join, resolve } from "node:path";

/**
 * A fast, read-only sample of what this laptop is doing right now: CPU and
 * memory pressure plus per-GPU utilization. Unlike `tt hardware`, it never
 * runs the bundled Python probe and never writes the hardware snapshot, so the
 * shell can call it at startup and on `/system` without noticeable delay.
 */
export interface LiveGpuUsage {
  index: number;
  name: string;
  /** Percent of time the GPU was busy over the driver's sample window. */
  utilization_percent?: number;
  memory_used_bytes?: number;
  memory_total_bytes?: number;
  temperature_c?: number;
  /** Apple Silicon and Spark-class GPUs share system memory. */
  unified_memory: boolean;
  /** "nvidia" GPUs can train locally; "apple" GPUs are listed for context. */
  vendor: "nvidia" | "apple";
}

export interface LiveUsage {
  sampled_at: string;
  cpu: {
    model: string;
    cores: number;
    /** Busy share across all cores over the sample window, 0–100. */
    utilization_percent?: number;
    load_average: [number, number, number];
  };
  memory: { used_bytes: number; total_bytes: number };
  /** Space where base models download (the Hugging Face cache). */
  disk?: { path: string; free_bytes: number; total_bytes: number };
  gpus: LiveGpuUsage[];
  /** Why no NVIDIA GPU was listed, when none was. */
  gpu_note?: string;
}

export interface SampleLiveUsageOptions {
  env?: NodeJS.ProcessEnv;
  /** How long to watch CPU counters. */
  cpuSampleMs?: number;
  /** Upper bound for the nvidia-smi query. */
  gpuTimeoutMs?: number;
  platform?: NodeJS.Platform;
  arch?: string;
}

interface CpuTimes {
  idle: number;
  total: number;
}

function cpuTimes(): CpuTimes {
  let idle = 0;
  let total = 0;
  for (const cpu of cpus()) {
    const times = cpu.times;
    idle += times.idle;
    total += times.user + times.nice + times.sys + times.irq + times.idle;
  }
  return { idle, total };
}

export function cpuUtilization(before: CpuTimes, after: CpuTimes): number | undefined {
  const total = after.total - before.total;
  if (!(total > 0)) return undefined;
  const busy = total - (after.idle - before.idle);
  return Math.min(100, Math.max(0, Math.round((busy / total) * 100)));
}

function optionalNumber(value: string | undefined): number | undefined {
  if (value === undefined) return undefined;
  const trimmed = value.trim();
  if (!trimmed || /^\[?n\/a\]?$/i.test(trimmed) || /not supported/i.test(trimmed)) return undefined;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : undefined;
}

const MIB = 1024 * 1024;

export const NVIDIA_USAGE_QUERY = [
  "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
  "--format=csv,noheader,nounits",
];

/** Parse rows produced by {@link NVIDIA_USAGE_QUERY}. */
export function parseNvidiaUsage(stdout: string): LiveGpuUsage[] {
  const gpus: LiveGpuUsage[] = [];
  for (const line of stdout.split(/\r?\n/).map((item) => item.trim()).filter(Boolean)) {
    const parts = line.split(",").map((part) => part.trim());
    if (parts.length < 2 || !/^\d+$/.test(parts[0] ?? "")) continue;
    const name = parts[1] || "NVIDIA GPU";
    const used = optionalNumber(parts[3]);
    const total = optionalNumber(parts[4]);
    gpus.push({
      index: Number(parts[0]),
      name,
      utilization_percent: optionalNumber(parts[2]),
      memory_used_bytes: used === undefined ? undefined : Math.round(used * MIB),
      memory_total_bytes: total === undefined || total <= 0 ? undefined : Math.round(total * MIB),
      temperature_c: optionalNumber(parts[5]),
      unified_memory: /spark|gb10/i.test(name),
      vendor: "nvidia",
    });
  }
  return gpus;
}

function queryNvidia(env: NodeJS.ProcessEnv, timeoutMs: number): Promise<{ gpus: LiveGpuUsage[]; note?: string }> {
  return new Promise((resolve) => {
    execFile("nvidia-smi", NVIDIA_USAGE_QUERY, { env, timeout: timeoutMs, windowsHide: true }, (error, stdout) => {
      if (error) {
        const code = (error as NodeJS.ErrnoException).code;
        resolve({
          gpus: [],
          note: code === "ENOENT"
            ? "nvidia-smi not found"
            : error.killed ? "nvidia-smi timed out" : "nvidia-smi failed",
        });
        return;
      }
      const gpus = parseNvidiaUsage(String(stdout));
      resolve(gpus.length > 0 ? { gpus } : { gpus, note: "nvidia-smi listed no GPUs" });
    });
  });
}

/** Drop marketing noise such as "(R)", "(TM)" and clock suffixes. */
export function tidyCpuModel(model: string): string {
  return model
    .replace(/\((?:R|TM|tm|r)\)/g, "")
    .replace(/\s+(?:CPU|Processor)\b/gi, "")
    .replace(/\s*@\s*[\d.]+\s*[GM]Hz/i, "")
    .replace(/\s+\d+-Core$/i, "")
    .replace(/\s+/g, " ")
    .trim() || "CPU";
}

/** Arm Ltd. (implementer 0x41) core names by MIDR part number. */
const ARM_CORES: Record<string, string> = {
  "0xd03": "Cortex-A53", "0xd05": "Cortex-A55", "0xd08": "Cortex-A72", "0xd0b": "Cortex-A76",
  "0xd0c": "Neoverse-N1", "0xd40": "Neoverse-V1", "0xd41": "Cortex-A78", "0xd44": "Cortex-X1",
  "0xd47": "Cortex-A710", "0xd48": "Cortex-X2", "0xd49": "Neoverse-N2", "0xd4d": "Cortex-A715",
  "0xd4e": "Cortex-X3", "0xd4f": "Neoverse-V2", "0xd80": "Cortex-A520", "0xd81": "Cortex-A720",
  "0xd82": "Cortex-X4", "0xd84": "Neoverse-V3", "0xd85": "Cortex-X925", "0xd87": "Cortex-A725",
  "0xd8e": "Neoverse-N3",
};

/**
 * Name the cores in an Arm Linux `/proc/cpuinfo`, which lists implementer and
 * part IDs instead of a "model name" (so Node reports "unknown"). Big.LITTLE
 * parts are joined, e.g. "Cortex-X925 + Cortex-A725" on GB10.
 */
export function armCpuModel(cpuinfo: string): string | undefined {
  const names: string[] = [];
  let implementer: string | undefined;
  for (const line of cpuinfo.split(/\r?\n/)) {
    const [key, value] = line.split(":").map((item) => item.trim().toLowerCase());
    if (key === "cpu implementer") implementer = value;
    if (key !== "cpu part" || !value) continue;
    const name = implementer === "0x41" ? ARM_CORES[value] : undefined;
    if (name && !names.includes(name)) names.push(name);
  }
  return names.length > 0 ? names.join(" + ") : undefined;
}

function cpuModel(reported: string | undefined, platform: NodeJS.Platform, arch: string): string {
  const model = tidyCpuModel(reported ?? "");
  if (platform !== "linux" || !/^(?:unknown|CPU)$/i.test(model)) return model;
  try {
    return armCpuModel(readFileSync("/proc/cpuinfo", "utf8")) ?? (arch === "arm64" ? "Arm CPU" : model);
  } catch {
    return arch === "arm64" ? "Arm CPU" : model;
  }
}

async function modelCacheDisk(env: NodeJS.ProcessEnv): Promise<LiveUsage["disk"]> {
  const path = resolve(env.HF_HOME?.trim() || join(homedir(), ".cache", "huggingface"));
  // The cache may not exist yet; report the filesystem it would land on.
  for (let probe = path, depth = 0; depth < 8; depth += 1) {
    try {
      const fs = await statfs(probe);
      return {
        path,
        free_bytes: Number(fs.bavail) * Number(fs.bsize),
        total_bytes: Number(fs.blocks) * Number(fs.bsize),
      };
    } catch {
      const parent = dirname(probe);
      if (parent === probe) break;
      probe = parent;
    }
  }
  return undefined;
}

function appleGpu(cpuModel: string, platform: NodeJS.Platform, arch: string): LiveGpuUsage | undefined {
  if (platform !== "darwin" || arch !== "arm64") return undefined;
  const chip = /apple/i.test(cpuModel) ? cpuModel.trim() : "Apple Silicon";
  return {
    index: 0,
    name: `${chip} GPU`,
    memory_total_bytes: totalmem(),
    unified_memory: true,
    vendor: "apple",
  };
}

export async function sampleLiveUsage(options: SampleLiveUsageOptions = {}): Promise<LiveUsage> {
  const env = options.env ?? process.env;
  const platform = options.platform ?? process.platform;
  const arch = options.arch ?? process.arch;
  const before = cpuTimes();
  const [nvidia, disk] = await Promise.all([
    queryNvidia(env, options.gpuTimeoutMs ?? 2_000),
    modelCacheDisk(env).catch(() => undefined),
    new Promise((resolve) => setTimeout(resolve, options.cpuSampleMs ?? 200)),
  ]);
  const after = cpuTimes();
  const list = cpus();
  const model = cpuModel(list[0]?.model, platform, arch);
  const total = totalmem();
  const apple = nvidia.gpus.length === 0 ? appleGpu(model, platform, arch) : undefined;
  const [one, five, fifteen] = loadavg();
  return {
    sampled_at: new Date().toISOString(),
    cpu: {
      model,
      cores: list.length,
      utilization_percent: cpuUtilization(before, after),
      load_average: [one ?? 0, five ?? 0, fifteen ?? 0],
    },
    memory: { used_bytes: Math.max(0, total - freemem()), total_bytes: total },
    disk,
    gpus: apple ? [apple] : nvidia.gpus,
    gpu_note: apple ? undefined : nvidia.note,
  };
}
