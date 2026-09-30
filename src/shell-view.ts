import chalk from "chalk";
import { evaluateCapabilities, type CapabilityReport, type CapabilityVerdict } from "./local-runtime/capability.js";
import type { HostInventory } from "./local-runtime/host-inventory.js";
import type { LiveGpuUsage, LiveUsage } from "./local-runtime/live-usage.js";
import { renderMascot, mascotSays, MASCOT_WIDTH } from "./mascot.js";
import type { ShellContext } from "./shell-context.js";
import { sanitizeTerminalText, terminalWidth, wrapTerminalLine } from "./terminal-markdown.js";
import { hintLines, truncateText } from "./spec-view.js";

const accent = chalk.hex("#8B5CF6");
const GIB = 1024 ** 3;
const BAR_WIDTH = 10;
/** The panels stay readable on wide terminals instead of stretching. */
const MAX_PANEL_WIDTH = 88;

export function panelWidth(columns?: number): number {
  return Math.max(40, Math.min(columns ?? 80, MAX_PANEL_WIDTH));
}

export function formatGiB(bytes: number): string {
  const amount = bytes / GIB;
  return amount >= 100 ? amount.toFixed(0) : amount.toFixed(1);
}

function loadColor(percent: number): (text: string) => string {
  if (percent >= 85) return chalk.red;
  if (percent >= 60) return chalk.yellow;
  return chalk.green;
}

/** A fixed-width meter such as `██████░░░░`, colored by load. */
export function usageBar(percent: number | undefined, width = BAR_WIDTH): string {
  if (percent === undefined) return chalk.dim("·".repeat(width));
  const clamped = Math.max(0, Math.min(100, percent));
  const filled = Math.round((clamped / 100) * width);
  return `${loadColor(clamped)("█".repeat(filled))}${chalk.dim("░".repeat(width - filled))}`;
}

function percentLabel(percent: number | undefined): string {
  if (percent === undefined) return chalk.dim("  n/a");
  return loadColor(percent)(`${String(Math.round(percent)).padStart(3)}%`).padStart(4);
}

/** Section title followed by a quiet rule, e.g. `MACHINE ─────`. */
export function sectionRule(title: string, width: number, detail?: string): string {
  const label = `${accent.bold(title)}${detail ? ` ${chalk.dim(detail)}` : ""}`;
  const rule = Math.max(3, width - terminalWidth(label) - 1);
  return `${label} ${chalk.dim("─".repeat(rule))}`;
}

// ── Fine-tune fit ─────────────────────────────────────────────────────────

export type FitStatus = CapabilityVerdict["status"] | "remote" | "unknown";

export interface FineTuneFit {
  status: FitStatus;
  text: string;
  /** Where the verdict came from, shown dimmed. */
  source?: string;
}

/** A capability inventory from the live sample, without the Python probe. */
export function inventoryFromUsage(usage: LiveUsage): HostInventory {
  const nvidia = usage.gpus.filter((gpu) => gpu.vendor === "nvidia");
  return {
    collected_at: usage.sampled_at,
    quick: true,
    node: { version: process.versions.node, major: Number(process.versions.node.split(".")[0]), ok: true },
    os: {
      platform: process.platform,
      arch: process.arch,
      type: process.platform,
      cpu_count: usage.cpu.cores,
      total_memory_bytes: usage.memory.total_bytes,
      free_memory_bytes: usage.memory.total_bytes - usage.memory.used_bytes,
    },
    gpus: nvidia.map((gpu) => ({
      index: gpu.index,
      name: gpu.name,
      memory_total_bytes: gpu.memory_total_bytes,
      memory_free_bytes: gpu.memory_total_bytes !== undefined && gpu.memory_used_bytes !== undefined
        ? gpu.memory_total_bytes - gpu.memory_used_bytes
        : undefined,
      unified_memory: gpu.unified_memory,
    })),
    nvidia_smi: { ok: nvidia.length > 0, message: usage.gpu_note ?? "" },
    disks: [],
  };
}

function shortModel(id: string, width = 24): string {
  return truncateText(id.split("/").at(-1) ?? id, width);
}

const STATUS_WORD: Record<CapabilityVerdict["status"], string> = {
  ready: "ready",
  tight: "tight",
  not_possible: "not possible",
};

/**
 * Summarize whether this machine can fine-tune the current spec. A fresh full
 * `tt hardware` snapshot wins; otherwise the live sample gives a quick verdict
 * from the same certified memory profiles (never generic size guesses).
 */
export function describeFineTuneFit(usage: LiveUsage | undefined, context: ShellContext): FineTuneFit {
  const spec = context.spec;
  if (spec?.remoteGpu) {
    return {
      status: "remote",
      text: `training runs on AWS ${spec.remoteGpu} (runtime.gpu)`,
      source: "tt doctor checks it",
    };
  }
  const snapshot = context.host && !context.host.stale ? context.host.capabilities : undefined;
  let report: CapabilityReport | undefined = snapshot;
  let source = snapshot ? "from tt hardware" : "quick check · tt hardware for a full probe";
  if (!report && usage) report = evaluateCapabilities(inventoryFromUsage(usage));
  if (!report) return { status: "unknown", text: "not checked yet", source: "run tt hardware" };

  const noCuda = !report.cuda_available;
  if (spec?.engine === "foundation") {
    const verdict = report.foundation.train;
    return {
      status: verdict.status,
      text: verdict.status === "not_possible"
        ? `foundation training ${STATUS_WORD[verdict.status]} — ${verdict.reason}`
        : `foundation training ${STATUS_WORD[verdict.status]} · suggested max depth ${report.foundation.suggested_max_depth}`,
      source: noCuda ? "set runtime.gpu to train on your AWS GPU" : source,
    };
  }

  if (spec?.baseModel) {
    const adapter = report.adapters.find((item) => item.id.toLowerCase() === spec.baseModel!.toLowerCase());
    if (!adapter) {
      return {
        status: "unknown",
        text: `no certified memory profile for ${shortModel(spec.baseModel, 32)}`,
        source: "tt hardware lists supported base models",
      };
    }
    const verdict = adapter.finetune;
    return {
      status: verdict.status,
      text: `LoRA ${shortModel(adapter.id)} ${STATUS_WORD[verdict.status]} — ${verdict.reason}`,
      source: noCuda ? "set runtime.gpu to train on your AWS GPU" : source,
    };
  }

  const ready = report.adapters.filter((item) => item.finetune.status !== "not_possible");
  if (ready.length === 0) {
    return {
      status: "not_possible",
      text: noCuda ? "no local CUDA GPU for LoRA training" : "no certified base model fits this GPU",
      source: noCuda ? "runtime.gpu can use your AWS GPU" : source,
    };
  }
  const best = ready.some((item) => item.finetune.status === "ready") ? "ready" : "tight";
  return {
    status: best,
    text: `LoRA ${best} for ${ready.map((item) => shortModel(item.id, 20)).join(", ")}`,
    source,
  };
}

function fitMark(status: FitStatus): string {
  switch (status) {
    case "ready": return chalk.green("✓");
    case "tight": return chalk.yellow("!");
    case "not_possible": return chalk.red("✗");
    case "remote": return accent("↗");
    default: return chalk.dim("?");
  }
}

// ── Machine panel ─────────────────────────────────────────────────────────

function gpuDetail(gpu: LiveGpuUsage): string {
  const parts: string[] = [];
  if (gpu.vendor === "apple") {
    parts.push(`${formatGiB(gpu.memory_total_bytes ?? 0)} GiB unified · Metal`);
  } else if (gpu.memory_total_bytes !== undefined) {
    const used = gpu.memory_used_bytes;
    parts.push(used === undefined
      ? `${formatGiB(gpu.memory_total_bytes)} GiB VRAM`
      : `VRAM ${formatGiB(used)}/${formatGiB(gpu.memory_total_bytes)} GiB`);
  }
  if (gpu.temperature_c !== undefined) parts.push(`${Math.round(gpu.temperature_c)}°C`);
  return parts.join("  ");
}

interface MeterRow {
  label: string;
  name: string;
  percent: number | undefined;
  detail: string;
}

/** Lay meters out in shared columns; the name column absorbs spare width. */
function renderMeterRows(rows: MeterRow[], width: number): string[] {
  const detailWidth = Math.max(0, ...rows.map((row) => terminalWidth(row.detail)));
  // label 6 + gaps + bar + percent; the name column takes what is left.
  const fixed = 6 + 1 + 1 + BAR_WIDTH + 1 + 4 + 2;
  const nameWidth = Math.max(6, Math.min(28, width - fixed - detailWidth));
  const detailRoom = Math.max(8, width - fixed - nameWidth);
  return rows.map((row) => {
    const nameCell = truncateText(row.name, nameWidth).padEnd(nameWidth);
    return `${chalk.bold(row.label.padEnd(6))} ${nameCell} ${usageBar(row.percent)} ${percentLabel(row.percent)}  ${chalk.dim(truncateText(row.detail, detailRoom))}`;
  });
}

export interface MachinePanelOptions {
  columns?: number;
  /** Include per-profile verdicts and disk space (the `/system` view). */
  detailed?: boolean;
}

export function renderMachinePanel(
  usage: LiveUsage,
  context: ShellContext,
  options: MachinePanelOptions = {},
): string[] {
  const width = panelWidth(options.columns);
  const lines: string[] = [sectionRule("MACHINE", width)];

  if (usage.gpus.length === 0) {
    const reason = usage.gpu_note ? ` (${usage.gpu_note})` : "";
    lines.push(`${chalk.bold("GPU".padEnd(6))} ${chalk.dim(truncateText(`no NVIDIA GPU detected${reason}`, width - 7))}`);
  }
  const meters: MeterRow[] = usage.gpus.map((gpu) => ({
    label: usage.gpus.length > 1 ? `GPU ${gpu.index}` : "GPU",
    // "NVIDIA GeForce RTX 4090" reads better as "GeForce RTX 4090" in a narrow column.
    name: sanitizeTerminalText(gpu.name).replace(/^NVIDIA\s+(?=\S)/, ""),
    percent: gpu.utilization_percent,
    detail: gpuDetail(gpu),
  }));
  meters.push({
    label: "CPU",
    name: `${usage.cpu.cores}× ${sanitizeTerminalText(usage.cpu.model)}`,
    percent: usage.cpu.utilization_percent,
    detail: `load ${usage.cpu.load_average[0].toFixed(2)}`,
  });
  meters.push({
    label: "RAM",
    name: "system memory",
    percent: usage.memory.total_bytes > 0 ? (usage.memory.used_bytes / usage.memory.total_bytes) * 100 : undefined,
    detail: `${formatGiB(usage.memory.used_bytes)}/${formatGiB(usage.memory.total_bytes)} GiB`,
  });
  if (usage.disk && usage.disk.total_bytes > 0) {
    meters.push({
      label: "Disk",
      name: "model cache",
      percent: ((usage.disk.total_bytes - usage.disk.free_bytes) / usage.disk.total_bytes) * 100,
      detail: `${formatGiB(usage.disk.free_bytes)} GiB free`,
    });
  }
  lines.push(...renderMeterRows(meters, width));

  const fit = describeFineTuneFit(usage, context);
  lines.push(`${chalk.bold("Tune".padEnd(6))} ${fitMark(fit.status)} ${truncateText(fit.text, width - 9)}`);
  if (fit.source) lines.push(`${" ".repeat(9)}${chalk.dim(truncateText(fit.source, width - 9))}`);

  if (options.detailed) {
    const report = context.host && !context.host.stale
      ? context.host.capabilities
      : evaluateCapabilities(inventoryFromUsage(usage));
    if (report) {
      lines.push("", sectionRule("CERTIFIED BASE MODELS", width, "LoRA fine-tune"));
      const idWidth = Math.min(44, Math.floor(width * 0.55));
      const reasonWidth = Math.max(10, width - idWidth - 3);
      const row = (status: FitStatus, label: string, reason: string) =>
        `${fitMark(status)} ${truncateText(label, idWidth).padEnd(idWidth)} ${chalk.dim(truncateText(reason, reasonWidth))}`;
      for (const adapter of report.adapters) {
        lines.push(row(adapter.finetune.status, adapter.id, adapter.finetune.reason));
      }
      const foundation = report.foundation.train;
      lines.push(row(
        foundation.status,
        "foundation engine (from scratch)",
        foundation.status === "not_possible" ? foundation.reason : `max depth ${report.foundation.suggested_max_depth}`,
      ));
    }
    lines.push("");
    const note = context.host
      ? `Full probe ${context.host.stale ? "is stale" : "cached"} from ${context.host.collectedAt.slice(0, 16).replace("T", " ")} · tt hardware refreshes it`
      : "No full probe yet · tt hardware checks CUDA, torch and disk space";
    lines.push(...wrapTerminalLine(note, width).map((line) => chalk.dim(line)));
  }
  return lines;
}

// ── Banner ────────────────────────────────────────────────────────────────

export interface BannerInput {
  version?: string;
  context: ShellContext;
  activeModel: string;
  usage?: LiveUsage;
  columns?: number;
}

function agentLabel(context: ShellContext): string | undefined {
  const agent = context.agent;
  if (!agent?.provider || !agent.model) return undefined;
  return `${agent.provider}/${agent.model}`;
}

/** The banner's opening line: the single most useful next step. */
export function bannerTip(input: BannerInput): string {
  const { context, usage } = input;
  const agent = agentLabel(context);
  if (!agent) {
    return "Use /login tunedtensor for managed inference, or /model for your own provider. Workflow commands work now.";
  }
  if (!context.spec) {
    return "No tunedtensor.json here yet. Describe the model you want and I'll draft a spec, or run init.";
  }
  if (context.spec.parseError) {
    return "tunedtensor.json doesn't parse. /spec show points at the problem.";
  }
  const busy = usage?.gpus.find((gpu) => (gpu.utilization_percent ?? 0) >= 85);
  if (busy) {
    return `${busy.name} is ${busy.utilization_percent}% busy; training now may be slow.`;
  }
  const examples = context.spec.exampleCount ?? 0;
  if (examples < 20) {
    return `${examples} example${examples === 1 ? "" : "s"} is a small set. Ask me to draft more, or /spec examples to review them.`;
  }
  return "Ask TT anything. Known commands run directly.";
}

function specLine(context: ShellContext): string {
  const spec = context.spec;
  if (!spec) return `${chalk.dim("spec")} ${chalk.yellow("none here")} ${chalk.dim("· init creates one")}`;
  if (spec.parseError) return `${chalk.dim("spec")} ${chalk.red("tunedtensor.json is invalid JSON")}`;
  const name = sanitizeTerminalText(spec.name ?? "unnamed");
  const facts = [
    spec.engine === "foundation" ? "foundation" : spec.baseModel ? sanitizeTerminalText(spec.baseModel) : undefined,
    spec.exampleCount === undefined ? undefined : `${spec.exampleCount} examples`,
  ].filter(Boolean).join(" · ");
  return `${chalk.dim("spec")} ${chalk.bold(name)}${facts ? chalk.dim(` · ${facts}`) : ""}`;
}

export function renderBanner(input: BannerInput): string {
  const width = panelWidth(input.columns);
  const { context } = input;
  const tip = bannerTip(input);
  const heading = [
    `${accent.bold("tt")}${input.version ? ` ${chalk.dim(`v${input.version}`)}` : ""}  ${chalk.dim("Tuned Tensor")}`,
    specLine(context),
    chalk.dim(`agent ${agentLabel(context) ?? "not configured"} · workflow model ${input.activeModel}`),
  ];
  const mascot = renderMascot();
  const textWidth = width - MASCOT_WIDTH - 1;
  // Bottom-align the heading so it sits beside the cube's face.
  const offset = mascot.length - heading.length;
  const lines = mascot.map((row, index) => {
    const text = heading[index - offset] ?? "";
    return `${row} ${terminalWidth(text) > textWidth ? truncateText(text, textWidth) : text}`;
  });

  if (input.usage) {
    lines.push("", ...renderMachinePanel(input.usage, context, { columns: input.columns }));
  }
  // Wrap the tip so continuation rows hang under the message text.
  const prefix = mascotSays("");
  const hang = terminalWidth(prefix);
  const wrapped = wrapTerminalLine(tip, Math.max(20, width - hang));
  lines.push("", `${prefix}${wrapped[0]}`, ...wrapped.slice(1).map((line) => `${" ".repeat(hang)}${line}`));
  lines.push(...hintLines([
    "/spec review",
    "/system machine",
    "/help commands",
    "tab complete",
    "ctrl+c stop/clear",
    "ctrl+d exit",
  ], width));
  return `${lines.join("\n")}\n\n`;
}
