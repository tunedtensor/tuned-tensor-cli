import { describe, expect, it } from "vitest";
import { armCpuModel, cpuUtilization, parseNvidiaUsage, sampleLiveUsage, tidyCpuModel, type LiveUsage } from "../local-runtime/live-usage.js";
import { describeFineTuneFit, inventoryFromUsage, renderMachinePanel, usageBar } from "../shell-view.js";
import type { ShellContext } from "../shell-context.js";
import { stripVTControlCharacters } from "node:util";

const GIB = 1024 ** 3;

function usage(gpus: LiveUsage["gpus"]): LiveUsage {
  return {
    sampled_at: "2026-01-01T00:00:00.000Z",
    cpu: { model: "CPU", cores: 8, utilization_percent: 10, load_average: [1, 1, 1] },
    memory: { used_bytes: 8 * GIB, total_bytes: 32 * GIB },
    gpus,
  };
}

const gpu24: LiveUsage["gpus"][number] = {
  index: 0,
  name: "NVIDIA GeForce RTX 4090",
  utilization_percent: 5,
  memory_used_bytes: GIB,
  memory_total_bytes: 24 * GIB,
  unified_memory: false,
  vendor: "nvidia",
};

function context(spec?: Partial<NonNullable<ShellContext["spec"]>>): ShellContext {
  return {
    cwd: "/p",
    projectName: "p",
    spec: spec ? { path: "/p/tunedtensor.json", parseError: false, ...spec } : undefined,
    local: { artifactRoot: "/p/a", storeRoot: "/p/s" },
    warnings: [],
  };
}

describe("parseNvidiaUsage", () => {
  it("reads utilization, memory and temperature per GPU", () => {
    const gpus = parseNvidiaUsage("0, NVIDIA GeForce RTX 4090, 67, 18432, 24564, 61\n1, NVIDIA GB10, [N/A], [N/A], [N/A], 40\n");
    expect(gpus).toEqual([
      expect.objectContaining({ index: 0, utilization_percent: 67, memory_used_bytes: 18432 * 1024 ** 2, memory_total_bytes: 24564 * 1024 ** 2, temperature_c: 61, unified_memory: false }),
      expect.objectContaining({ index: 1, name: "NVIDIA GB10", utilization_percent: undefined, memory_total_bytes: undefined, unified_memory: true }),
    ]);
  });

  it("ignores banners and malformed rows", () => {
    expect(parseNvidiaUsage("NVIDIA-SMI has failed\n\n")).toEqual([]);
  });
});

describe("live CPU sampling", () => {
  it("computes busy share between samples", () => {
    expect(cpuUtilization({ idle: 100, total: 200 }, { idle: 150, total: 400 })).toBe(75);
    expect(cpuUtilization({ idle: 1, total: 1 }, { idle: 1, total: 1 })).toBeUndefined();
  });

  it("tidies CPU marketing names", () => {
    expect(tidyCpuModel("Intel(R) Xeon(R) Processor @ 2.10GHz")).toBe("Intel Xeon");
    expect(tidyCpuModel("AMD Ryzen 9 7950X 16-Core Processor")).toBe("AMD Ryzen 9 7950X");
    expect(tidyCpuModel("Apple M3 Max")).toBe("Apple M3 Max");
  });

  it("names Arm cores from /proc/cpuinfo", () => {
    const core = (part: string) => `processor\t: 0\nCPU implementer\t: 0x41\nCPU architecture: 8\nCPU part\t: ${part}\n`;
    expect(armCpuModel([core("0xd85"), core("0xd85"), core("0xd87")].join("\n"))).toBe("Cortex-X925 + Cortex-A725");
    expect(armCpuModel(core("0xfff"))).toBeUndefined();
    expect(armCpuModel("processor\t: 0\nmodel name\t: Intel Xeon\n")).toBeUndefined();
  });

  it("samples this machine without a GPU tool", async () => {
    const sample = await sampleLiveUsage({ env: { PATH: "/nonexistent" }, cpuSampleMs: 10, platform: "linux" });
    expect(sample.cpu.cores).toBeGreaterThan(0);
    expect(sample.memory.total_bytes).toBeGreaterThan(0);
    expect(sample.gpus).toEqual([]);
    expect(sample.gpu_note).toBe("nvidia-smi not found");
  });

  it("lists Apple Silicon GPUs as unified memory", async () => {
    const sample = await sampleLiveUsage({ env: { PATH: "/nonexistent" }, cpuSampleMs: 10, platform: "darwin", arch: "arm64" });
    expect(sample.gpus).toEqual([expect.objectContaining({ vendor: "apple", unified_memory: true })]);
  });
});

describe("describeFineTuneFit", () => {
  it("uses certified profiles for the spec's base model", () => {
    expect(describeFineTuneFit(usage([gpu24]), context({ baseModel: "Qwen/Qwen3.5-2B" })))
      .toMatchObject({ status: "ready", text: expect.stringMatching(/LoRA Qwen3\.5-2B ready/) });
    expect(describeFineTuneFit(usage([gpu24]), context({ baseModel: "meta-models/Muse-Glimmer-30B" })))
      .toMatchObject({ status: "not_possible" });
  });

  it("never guesses for uncertified models", () => {
    expect(describeFineTuneFit(usage([gpu24]), context({ baseModel: "acme/Custom-7B" })))
      .toMatchObject({ status: "unknown", text: expect.stringContaining("no certified memory profile") });
  });

  it("points at runtime.gpu when there is no local CUDA GPU", () => {
    expect(describeFineTuneFit(usage([]), context({ baseModel: "Qwen/Qwen3.5-2B" })))
      .toMatchObject({ status: "not_possible", source: expect.stringContaining("runtime.gpu") });
    expect(describeFineTuneFit(usage([]), context({ remoteGpu: "i-0123456789abcdef0" })))
      .toMatchObject({ status: "remote", text: expect.stringContaining("i-0123456789abcdef0") });
  });

  it("covers the foundation engine and the no-spec summary", () => {
    expect(describeFineTuneFit(usage([gpu24]), context({ engine: "foundation" })))
      .toMatchObject({ status: "ready", text: expect.stringMatching(/suggested max depth \d+/) });
    expect(describeFineTuneFit(usage([gpu24]), context()).text).toContain("LoRA ready for Qwen3.5-2B");
    expect(describeFineTuneFit(undefined, context())).toMatchObject({ status: "unknown" });
  });

  it("uses system RAM when nvidia-smi reports no memory on a unified GB10", () => {
    const [gb10] = parseNvidiaUsage("0, NVIDIA GB10, 0, [N/A], [N/A], 40\n");
    const spark = { ...usage([gb10!]), memory: { used_bytes: 18 * GIB, total_bytes: 119 * GIB } };
    const fit = describeFineTuneFit(spark, context());
    expect(fit.status).toBe("ready");
    expect(fit.text).toContain("LoRA ready for");    const panel = stripVTControlCharacters(renderMachinePanel(spark, context(), { columns: 120 }).join("\n"));
    expect(panel).toContain("unified with system RAM");
  });

  it("builds a quick inventory with free VRAM", () => {
    expect(inventoryFromUsage(usage([gpu24])).gpus[0]).toMatchObject({ memory_free_bytes: 23 * GIB });
  });
});

describe("usageBar", () => {
  it("fills proportionally", () => {
    expect(stripVTControlCharacters(usageBar(50))).toBe("█████░░░░░");
    expect(stripVTControlCharacters(usageBar(undefined))).toBe("··········");
  });
});
