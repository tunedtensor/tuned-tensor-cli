import type { CapabilityReport, CapabilityVerdict } from "./capability.js";
import { supportsPackagedTrainingRuntime, type HostInventory } from "./host-inventory.js";

export type WorkflowStatus = "ready" | "needs_setup" | "limited" | "remote" | "unverified" | "unsupported";
export interface WorkflowGuidance {
  id: "prepare" | "decision" | "adapter" | "foundation" | "serve";
  label: string;
  status: WorkflowStatus;
  reason: string;
  next: string;
}
export const WORKFLOW_STATUS_LABEL: Record<WorkflowStatus, string> = {
  ready: "Ready",
  needs_setup: "Needs setup",
  limited: "Limited resources",
  remote: "Remote GPU",
  unverified: "Needs verification",
  unsupported: "Unsupported runtime",
};
const GiB = 1024 ** 3;

/** Guidance, not admission: actual model, batch size and sequence length still matter. */
export function workflowGuidance(host: HostInventory, report: CapabilityReport): WorkflowGuidance[] {
  const platform = host.os.platform;
  const supported = supportsPackagedTrainingRuntime(host.os);
  const setup = !host.node.ok ? "Install Node 22 or newer."
    : host.uv?.ok === false ? "Install uv to set up the training runtime." : undefined;
  const probe = host.decision_python;
  const device = probe?.cuda_available ? "CUDA" : probe?.mps_available ? "Apple MPS" : "CPU";
  const decision: WorkflowGuidance = {
    id: "decision", label: "Decision training", status: "unverified",
    reason: platform === "darwin" && host.os.arch === "arm64"
      ? "CPU supported; Apple MPS availability needs a runtime check."
      : "CPU supported; no NVIDIA GPU required.",
    next: "Run tt hardware, then try tt init --engine decision.",
  };
  if (!supported) {
    Object.assign(decision, {
      status: "unsupported",
      reason: "No packaged decision runtime for this OS/architecture.",
      next: "Use a supported Linux or Apple Silicon machine; specs and data preparation remain available.",
    });
  } else if (setup || probe?.ok === false) {
    Object.assign(decision, {
      status: "needs_setup",
      reason: setup ?? `Decision runtime check failed: ${probe!.message}`,
      next: "Resolve the setup issue and rerun tt hardware.",
    });
  } else if (probe?.ok) {
    Object.assign(decision, {
      status: platform === "win32" ? "unverified" : "ready",
      reason: `${device} runtime available; decision model fit is estimated.`,
      next: "Try tt init --engine decision; use a small dataset first.",
    });
  }
  if (supported && decision.status !== "needs_setup") {
    const memory = device === "CUDA" && report.gpu && !report.gpu.unified_memory
      ? report.gpu.memory_free_bytes ?? report.gpu.memory_total_bytes : host.os.free_memory_bytes;
    const lowDisk = host.disks.find(d => d.name === "model-cache" && d.free_bytes !== undefined && d.free_bytes < 6 * GiB);
    if ((memory !== undefined && memory < 8 * GiB) || lowDisk || host.os.cpu_count < 4) {
      decision.status = "limited";
      decision.reason += lowDisk ? " Less than 6 GiB cache space free."
        : memory !== undefined && memory < 8 * GiB ? " Less than 8 GiB working memory free."
          : " Fewer than 4 CPU cores; training may be slow.";
      decision.next = "Start with a small batch; free memory/disk or try CPU with freeze_encoder for a smaller training workload.";
    }
    if (platform === "win32") decision.reason += " Native Windows training is not yet end-to-end certified.";
  }
  const remote = "Configure runtime.gpu for your AWS GPU; tt doctor checks it.";
  const training = (id: "adapter" | "foundation", label: string, verdict: CapabilityVerdict): WorkflowGuidance => {
    if (!supported) return {
      id, label, status: "remote",
      reason: "This packaged local training runtime is unavailable.",
      next: remote,
    };
    if (setup || host.python?.ok === false) return {
      id, label, status: "needs_setup",
      reason: setup ?? host.python!.message,
      next: "Fix the local runtime, or configure runtime.gpu for AWS.",
    };
    if (!report.cuda_available) return {
      id, label, status: "remote",
      reason: "Current TT training requires NVIDIA CUDA.",
      next: remote,
    };
    if (platform !== "linux") return {
      id, label, status: "unverified",
      reason: "Native training on this OS is not certified; Windows users can use a Linux/WSL2 environment.",
      next: "Run tt doctor in the training environment, or use your AWS GPU.",
    };
    if (verdict.status !== "ready") return {
      id, label, status: "limited",
      reason: verdict.reason,
      next: "Choose a smaller supported model/configuration or use your AWS GPU.",
    };
    const free = report.gpu?.unified_memory ? host.os.free_memory_bytes : report.gpu?.memory_free_bytes;
    if (free !== undefined && verdict.required_bytes !== undefined && free < verdict.required_bytes) {
      return {
        id, label, status: "limited",
        reason: `Available GPU memory is below the estimated workload requirement (${(free / GiB).toFixed(1)} GiB free).`,
        next: "Free GPU/shared memory, choose a smaller configuration, or use your AWS GPU.",
      };
    }
    const verified = !host.quick && host.python?.ok;
    return {
      id, label, status: verified ? "ready" : "unverified",
      reason: verdict.reason,
      next: verified
        ? "tt doctor --spec tunedtensor.json checks the selected configuration."
        : "Run tt hardware to verify the training runtime.",
    };
  };
  const best = report.adapters.find(a => a.finetune.status === "ready")
    ?? report.adapters.find(a => a.finetune.status === "tight")
    ?? report.adapters[0]!;
  return [
    {
      id: "prepare", label: "Specs and datasets", status: host.node.ok ? "ready" : "needs_setup",
      reason: "Create specs, inspect and split local data; no GPU required.",
      next: host.node.ok
        ? "Ask TT to prepare a dataset or create a spec. Agent chat requires managed access or your own model provider."
        : "Install Node 22 or newer.",
    },
    decision,
    training("adapter", "LLM fine-tuning", { ...best.finetune, reason: `${best.id}: ${best.finetune.reason}` }),
    training("foundation", "Train from scratch", report.foundation.train),
    {
      id: "serve", label: "LLM serving",
      status: platform === "linux" && report.cuda_available ? "unverified" : "remote",
      reason: "tt serve uses a separate Linux/NVIDIA CUDA runtime; CPU evaluation does not imply CPU serving.",
      next: "On the GPU machine, use tt serve <model-id> --print-command to check the launch plan; actual startup checks the serving runtime.",
    },
  ];
}
