import {
  ADAPTER_MEMORY_PROFILES,
  FOUNDATION_DEFAULT_DEPTH,
  FOUNDATION_MAX_DEPTH,
  adapterMemoryProfile,
  foundationInferenceBytes,
  foundationTrainBytes,
  isUnifiedGpuName,
  type AdapterMemoryProfile,
} from "./capability-profiles.js";
import type { HostGpu, HostInventory } from "./host-inventory.js";

export type CapabilityVerdictStatus = "ready" | "tight" | "not_possible";
export type CapabilityWorkload = "train" | "finetune" | "inference";

export interface CapabilityVerdict {
  status: CapabilityVerdictStatus;
  reason: string;
  required_bytes?: number;
  available_bytes?: number;
}

export interface AdapterCapability {
  id: string;
  spark_class: boolean;
  train: CapabilityVerdict;
  finetune: CapabilityVerdict;
  inference: CapabilityVerdict;
}

export interface FoundationCapability {
  default_depth: number;
  suggested_max_depth: number;
  train: CapabilityVerdict;
  finetune: CapabilityVerdict;
  inference: CapabilityVerdict;
  serve: CapabilityVerdict;
}

export interface CapabilityReport {
  cuda_available: boolean;
  gpu?: HostGpu;
  adapters: AdapterCapability[];
  foundation: FoundationCapability;
  notes: string[];
}

const HEADROOM = 1.2;
const FIT_RATIO = 0.7;

function formatGiB(bytes: number): string {
  const amount = bytes / (1024 ** 3);
  const precision = amount >= 10 ? 0 : 1;
  return `${amount.toFixed(precision)} GiB`;
}

function compareFit(available: number | undefined, required: number, missing: string): CapabilityVerdict {
  if (available === undefined) {
    return { status: "not_possible", reason: missing, required_bytes: required };
  }
  if (available < required) {
    return {
      status: "not_possible",
      reason: `needs ~${formatGiB(required)}; ${formatGiB(available)} available`,
      required_bytes: required,
      available_bytes: available,
    };
  }
  if (available < required * HEADROOM) {
    return {
      status: "tight",
      reason: `fits in ${formatGiB(available)} with little headroom (needs ~${formatGiB(required)})`,
      required_bytes: required,
      available_bytes: available,
    };
  }
  return {
    status: "ready",
    reason: `${formatGiB(available)} available (needs ~${formatGiB(required)})`,
    required_bytes: required,
    available_bytes: available,
  };
}

function trainingBlockers(inventory: HostInventory): string[] {
  const blockers: string[] = [];
  if (!inventory.node.ok) blockers.push(`Node ${inventory.node.version} is below 22`);
  if (inventory.uv && !inventory.uv.ok) blockers.push(inventory.uv.message);
  if (inventory.python && !inventory.python.ok) blockers.push(inventory.python.message);
  return blockers;
}

function cudaAvailable(inventory: HostInventory): boolean {
  if (inventory.python?.ok && inventory.python.cuda_available !== undefined) {
    return inventory.python.cuda_available;
  }
  return inventory.nvidia_smi.ok && inventory.gpus.length > 0;
}

function trainingGpu(inventory: HostInventory): HostGpu | undefined {
  const listed = inventory.gpus.find((gpu) => gpu.index === 0) ?? inventory.gpus[0];
  if (listed) {
    if (!listed.memory_total_bytes && inventory.python?.total_memory_bytes) {
      return { ...listed, memory_total_bytes: inventory.python.total_memory_bytes };
    }
    return listed;
  }
  if (inventory.python?.cuda_device) {
    return {
      index: 0,
      name: inventory.python.cuda_device,
      memory_total_bytes: inventory.python.total_memory_bytes,
      unified_memory: isUnifiedGpuName(inventory.python.cuda_device),
    };
  }
  return undefined;
}

function gpuMemory(gpu: HostGpu | undefined): number | undefined {
  return gpu?.memory_total_bytes;
}

function diskForCache(inventory: HostInventory): number | undefined {
  return inventory.disks.find((disk) => disk.name === "model-cache")?.free_bytes;
}

function worse(left: CapabilityVerdict, right: CapabilityVerdict): CapabilityVerdict {
  const rank = { not_possible: 2, tight: 1, ready: 0 };
  return rank[left.status] >= rank[right.status] ? left : right;
}

function withBlockers(verdict: CapabilityVerdict, blockers: string[]): CapabilityVerdict {
  if (blockers.length === 0) return verdict;
  return {
    status: "not_possible",
    reason: blockers.join("; "),
    required_bytes: verdict.required_bytes,
    available_bytes: verdict.available_bytes,
  };
}

function adapterVerdicts(
  profile: AdapterMemoryProfile,
  inventory: HostInventory,
  gpu: HostGpu | undefined,
  hasCuda: boolean,
): AdapterCapability {
  const blockers = trainingBlockers(inventory);
  const gpuBytes = gpuMemory(gpu);
  const ram = inventory.os.total_memory_bytes;
  const disk = diskForCache(inventory);

  let train: CapabilityVerdict;
  if (!hasCuda) {
    train = {
      status: "not_possible",
      reason: "LoRA training requires CUDA",
      required_bytes: profile.trainBytes,
    };
  } else {
    train = compareFit(gpuBytes, profile.trainBytes, "GPU memory is unknown");
  }
  train = withBlockers(train, blockers);
  if (disk !== undefined && disk < profile.diskBytes && train.status !== "not_possible") {
    train = worse(train, compareFit(disk, profile.diskBytes, "model cache disk is unknown"));
  }

  const finetune = train;

  let inference: CapabilityVerdict;
  if (hasCuda) {
    inference = compareFit(gpuBytes, profile.inferenceBytes, "GPU memory is unknown");
  } else {
    inference = compareFit(ram, profile.inferenceBytes, "system RAM is unknown");
    if (inference.status !== "not_possible") {
      inference = {
        ...inference,
        reason: `CPU inference: ${inference.reason}`,
      };
    }
  }

  return {
    id: profile.id,
    spark_class: profile.sparkClass,
    train,
    finetune,
    inference,
  };
}

function suggestedFoundationDepth(availableBytes: number | undefined): number {
  if (availableBytes === undefined) return FOUNDATION_DEFAULT_DEPTH;
  const budget = availableBytes * FIT_RATIO;
  let suggested = FOUNDATION_DEFAULT_DEPTH;
  for (let depth = FOUNDATION_DEFAULT_DEPTH; depth <= FOUNDATION_MAX_DEPTH; depth += 1) {
    if (foundationTrainBytes(depth) <= budget) suggested = depth;
    else break;
  }
  return suggested;
}

export function evaluateCapabilities(inventory: HostInventory): CapabilityReport {
  const hasCuda = cudaAvailable(inventory);
  const gpu = trainingGpu(inventory);
  const notes: string[] = [];
  if (gpu?.unified_memory || isUnifiedGpuName(gpu?.name)) {
    notes.push("GPU memory is a unified pool shared with system RAM (Spark-class).");
  }
  if (inventory.quick) {
    notes.push("Quick probe skipped the bundled torch/uv runtime.");
  }
  if (!inventory.node.ok) {
    notes.push(`Node ${inventory.node.version} is below the 22.x required by the local runner.`);
  }

  const adapters = ADAPTER_MEMORY_PROFILES.map((profile) =>
    adapterVerdicts(profile, inventory, gpu, hasCuda)
  );

  const blockers = trainingBlockers(inventory);
  const gpuBytes = gpuMemory(gpu);
  const ram = inventory.os.total_memory_bytes;
  const trainBudget = hasCuda ? gpuBytes : undefined;
  const suggested = suggestedFoundationDepth(trainBudget);
  const defaultTrain = foundationTrainBytes(FOUNDATION_DEFAULT_DEPTH);
  const inferBudget = hasCuda ? gpuBytes : ram;

  let train: CapabilityVerdict;
  if (!hasCuda) {
    train = {
      status: "not_possible",
      reason: "Foundation pretrain, SFT, and RL require CUDA",
      required_bytes: defaultTrain,
    };
  } else {
    train = compareFit(gpuBytes, defaultTrain, "GPU memory is unknown");
  }
  train = withBlockers(train, blockers);

  const inference = compareFit(
    inferBudget,
    foundationInferenceBytes(FOUNDATION_DEFAULT_DEPTH),
    hasCuda ? "GPU memory is unknown" : "system RAM is unknown",
  );

  const foundation: FoundationCapability = {
    default_depth: FOUNDATION_DEFAULT_DEPTH,
    suggested_max_depth: hasCuda && blockers.length === 0 ? suggested : FOUNDATION_DEFAULT_DEPTH,
    train,
    finetune: train,
    inference: hasCuda
      ? inference
      : { ...inference, reason: `CPU eval: ${inference.reason}` },
    serve: {
      status: "not_possible",
      reason: "tt serve cannot host foundation checkpoints yet",
    },
  };

  return { cuda_available: hasCuda, gpu, adapters, foundation, notes };
}

export function formatCapabilitySummary(report: CapabilityReport): string {
  const gpu = report.gpu;
  const gpuLabel = gpu
    ? `${gpu.name}${gpu.memory_total_bytes ? `, ${formatGiB(gpu.memory_total_bytes)}` : ""}${gpu.unified_memory ? " unified" : ""}`
    : "no NVIDIA GPU";
  const adapterBits = report.adapters.map((adapter) => {
    const short = adapter.id.split("/").at(-1) ?? adapter.id;
    return `${short} train ${adapter.train.status}`;
  });
  return [
    `GPU ${gpuLabel}, CUDA ${report.cuda_available ? "yes" : "no"}`,
    `Adapter: ${adapterBits.join("; ")}`,
    `Foundation: train ${report.foundation.train.status}, suggested max depth ${report.foundation.suggested_max_depth}; serve not supported`,
  ].join(". ");
}

export function warningsFromSnapshot(
  report: CapabilityReport,
  target: { engine: "adapter" | "foundation"; baseModel?: string },
): string[] {
  const warnings: string[] = [];
  if (target.engine === "foundation") {
    for (const kind of ["train", "finetune"] as const) {
      const verdict = report.foundation[kind];
      if (verdict.status === "not_possible" || verdict.status === "tight") {
        warnings.push(`Foundation ${kind} is ${verdict.status}: ${verdict.reason}`);
      }
    }
    return warnings;
  }
  const modelId = target.baseModel;
  const adapter = modelId
    ? report.adapters.find((item) => item.id.toLowerCase() === modelId.trim().toLowerCase())
    : undefined;
  if (!adapter) {
    if (modelId && !adapterMemoryProfile(modelId)) return warnings;
    const blocked = report.adapters.filter((item) => item.train.status === "not_possible");
    if (blocked.length === report.adapters.length) {
      warnings.push(`Adapter train is not_possible: ${report.adapters[0]?.train.reason ?? "CUDA required"}`);
    }
    return warnings;
  }
  for (const kind of ["train", "finetune", "inference"] as const) {
    const verdict = adapter[kind];
    if (verdict.status === "not_possible" || verdict.status === "tight") {
      warnings.push(`${adapter.id} ${kind} is ${verdict.status}: ${verdict.reason}`);
    }
  }
  return warnings;
}

export { formatGiB };
