import { chmod, mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { getTunedTensorHome } from "../paths.js";
import type { CapabilityReport } from "./capability.js";
import { formatCapabilitySummary } from "./capability.js";
import type { HostInventory } from "./host-inventory.js";

export const HARDWARE_SNAPSHOT_VERSION = 1;
export const HARDWARE_SNAPSHOT_TTL_MS = 24 * 60 * 60 * 1000;

export interface HardwareSnapshot {
  version: typeof HARDWARE_SNAPSHOT_VERSION;
  collected_at: string;
  quick: boolean;
  gpu_fingerprint?: { name: string; total_memory_bytes?: number };
  inventory: HostInventory;
  capabilities: CapabilityReport;
  summary: string;
}

export interface LoadedHardwareSnapshot extends HardwareSnapshot {
  stale: boolean;
  path: string;
}

export function hardwareSnapshotPath(env: NodeJS.ProcessEnv = process.env): string {
  return join(getTunedTensorHome(env), "hardware.json");
}

export async function readHardwareSnapshot(
  env: NodeJS.ProcessEnv = process.env,
): Promise<LoadedHardwareSnapshot | undefined> {
  const path = hardwareSnapshotPath(env);
  try {
    const parsed = JSON.parse(await readFile(path, "utf8")) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return undefined;
    const snapshot = parsed as HardwareSnapshot;
    if (snapshot.version !== HARDWARE_SNAPSHOT_VERSION) return undefined;
    if (typeof snapshot.collected_at !== "string" || !snapshot.capabilities || !snapshot.inventory) {
      return undefined;
    }
    const age = Date.now() - Date.parse(snapshot.collected_at);
    const stale = !Number.isFinite(age) || age > HARDWARE_SNAPSHOT_TTL_MS || age < 0;
    return { ...snapshot, stale, path };
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
    return undefined;
  }
}

export async function writeHardwareSnapshot(
  snapshot: Omit<HardwareSnapshot, "version" | "summary"> & { summary?: string },
  env: NodeJS.ProcessEnv = process.env,
): Promise<HardwareSnapshot> {
  const path = hardwareSnapshotPath(env);
  await mkdir(dirname(path), { recursive: true, mode: 0o700 });
  const gpu = snapshot.capabilities.gpu;
  const record: HardwareSnapshot = {
    version: HARDWARE_SNAPSHOT_VERSION,
    collected_at: snapshot.collected_at,
    quick: snapshot.quick,
    gpu_fingerprint: gpu
      ? { name: gpu.name, total_memory_bytes: gpu.memory_total_bytes }
      : undefined,
    inventory: snapshot.inventory,
    capabilities: snapshot.capabilities,
    summary: snapshot.summary ?? formatCapabilitySummary(snapshot.capabilities),
  };
  await writeFile(path, `${JSON.stringify(record, null, 2)}\n`, { mode: 0o600 });
  await chmod(path, 0o600);
  return record;
}

export function formatHostStatusLine(snapshot: LoadedHardwareSnapshot | undefined): string {
  if (!snapshot) return "not inventoried (run tt hardware)";
  const stale = snapshot.stale ? `; stale — run tt hardware` : "";
  return `${snapshot.summary}${stale}`;
}

export function formatAgentHostBlock(snapshot: LoadedHardwareSnapshot | undefined): string {
  if (!snapshot) {
    return [
      "Host capability is not inventoried yet.",
      "When the user asks to examine this machine, GPU, VRAM, CUDA, or what can train, fine-tune, or infer here, call examine_hardware before recommending a base model, engine, or pipeline.",
    ].join(" ");
  }
  const stale = snapshot.stale ? " (stale — call examine_hardware to refresh)" : "";
  return [
    `Host capability (cached ${snapshot.collected_at}${stale}):`,
    snapshot.summary,
    "When the user wants to examine this host, GPU, or decide what can train/fine-tune/infer here, call examine_hardware before recommending a base model, engine, or pipeline.",
    "Recommend only workloads marked ready; mention tight as a caution. Never invent generic 7B/70B sizing.",
  ].join("\n");
}
