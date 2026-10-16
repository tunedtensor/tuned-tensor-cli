import { describe, expect, it } from "vitest";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { stripVTControlCharacters } from "node:util";
import { evaluateCapabilities, formatCapabilitySummary } from "../local-runtime/capability.js";
import { assessHardware } from "../local-runtime/hardware.js";
import { hardwareSnapshotPath, readHardwareSnapshot } from "../local-runtime/hardware-snapshot.js";
import { parsePythonProbeJson, type HostInventory } from "../local-runtime/host-inventory.js";
import { sampleLiveUsage } from "../local-runtime/live-usage.js";
import { describeFineTuneFit, inventoryFromUsage, machineCapabilities, renderMachinePanel } from "../shell-view.js";
import { terminalWidth } from "../terminal-markdown.js";

const GiB = 1024 ** 3;
function host(platform = "linux", arch = "x64"): HostInventory {
  return { collected_at: new Date().toISOString(), quick: false,
    node: { version: "22.23.1", major: 22, ok: true },
    os: { platform, arch, type: platform, cpu_count: 8, total_memory_bytes: 32 * GiB, free_memory_bytes: 24 * GiB },
    uv: { ok: true, message: "ok" }, gpus: [], nvidia_smi: { ok: false, message: "not installed" },
    python: { ok: true, message: "ok", cuda_available: false },
    decision_python: { ok: true, message: "ok", cuda_available: false, mps_available: false },
    disks: [{ name: "model-cache", path: tmpdir(), free_bytes: 200 * GiB, ok: true, message: "ok" }],
  };
}
function gpu(h: HostInventory, total = 24, free = 20) {
  h.nvidia_smi.ok = true;
  h.gpus = [{ index: 0, name: "NVIDIA RTX 4090", memory_total_bytes: total * GiB, memory_free_bytes: free * GiB, unified_memory: false }];
  h.python!.cuda_available = true;
  h.decision_python!.cuda_available = true;
  return h;
}
const workflows = (h: HostInventory) => Object.fromEntries(evaluateCapabilities(h).workflows!.map(w => [w.id, w]));

describe("workflow guidance across OS and resource profiles", () => {
  it("offers CPU decisions and preparation without requiring a GPU", () => {
    const w = workflows(host());
    expect(w.prepare.status).toBe("ready");
    expect(w.decision).toMatchObject({ status: "ready", reason: expect.stringContaining("CPU") });
    expect(w.adapter.status).toBe("remote");
    expect(w.foundation.next).toContain("runtime.gpu");
    expect(w.serve.reason).toContain("CPU evaluation does not imply CPU serving");
  });
  it("keeps decision readiness independent from a broken adapter runtime", () => {
    const h = host(); h.python = { ok: false, message: "adapter dependencies missing" };
    expect(workflows(h).decision.status).toBe("ready");
    expect(workflows(h).adapter.status).toBe("needs_setup");
  });
  it("reports MPS only when actually probed", () => {
    const h = host("darwin", "arm64"); h.decision_python!.mps_available = true;
    expect(workflows(h).decision).toMatchObject({ status: "ready", reason: expect.stringContaining("Apple MPS") });
    h.decision_python!.mps_available = false;
    expect(workflows(h).decision.reason).toContain("CPU");
    delete h.decision_python; h.quick = true;
    expect(workflows(h).decision).toMatchObject({ status: "unverified", reason: expect.stringContaining("needs a runtime check") });
    expect(workflows(h).serve.status).toBe("remote");
  });
  it.each([["darwin", "x64"], ["win32", "arm64"], ["freebsd", "x64"]])("does not promise unavailable wheels on %s/%s", (platform, arch) => {
    const w = workflows(host(platform, arch));
    expect(w.decision.status).toBe("unsupported");
    expect(w.prepare.status).toBe("ready");
  });
  it("keeps native Windows unverified even with a detected NVIDIA GPU", () => {
    const w = workflows(gpu(host("win32")));
    expect(w.decision.status).toBe("unverified");
    expect(w.decision.reason).toContain("Native Windows");
    expect(w.adapter).toMatchObject({ status: "unverified", reason: expect.stringContaining("WSL2") });
    expect(w.serve.status).toBe("remote");
  });
  it("can offer local training on Linux/WSL when CUDA and runtime checks pass", () => {
    const w = workflows(gpu(host()));
    expect(w.decision.reason).toContain("CUDA");
    expect(w.adapter.status).toBe("ready");
    expect(w.foundation.status).toBe("ready");
    expect(w.serve.status).toBe("unverified");
  });
  it("does not treat nvidia-smi as proof that PyTorch can use CUDA", () => {
    const h = gpu(host()); h.python!.cuda_available = false; h.decision_python!.cuda_available = false;
    expect(workflows(h).adapter.status).toBe("remote");
    expect(workflows(h).decision.reason).toContain("CPU");
    h.quick = true; delete h.python; delete h.decision_python;
    expect(workflows(h).adapter.status).toBe("unverified");
  });
  it.each(["memory", "disk", "cores", "vram"])("warns about constrained %s without declaring CPU training impossible", resource => {
    const h = resource === "vram" ? gpu(host(), 24, 3) : host();
    if (resource === "memory") h.os.free_memory_bytes = 3 * GiB;
    if (resource === "disk") h.disks[0]!.free_bytes = GiB;
    if (resource === "cores") h.os.cpu_count = 2;
    expect(workflows(h).decision.status).toBe("limited");
    expect(workflows(h).prepare.status).toBe("ready");
  });
  it("does not advertise a busy GPU as ready based only on total VRAM", () => {
    const w = workflows(gpu(host(), 24, 0.001));
    expect(w.adapter.status).toBe("limited");
    expect(w.foundation.status).toBe("limited");
    expect(w.adapter.reason).toContain("0.0 GiB free");
  });
  it("uses shared free RAM for Spark, not a second independent GPU memory pool", () => {
    const h = gpu(host(), 128, 128); h.gpus[0]!.unified_memory = true; h.os.free_memory_bytes = 3 * GiB;
    expect(workflows(h).decision.status).toBe("limited");
  });
  it.each(["uv", "node", "decision"])("offers setup instructions for missing %s", missing => {
    const h = host();
    if (missing === "uv") h.uv!.ok = false;
    if (missing === "node") h.node.ok = false;
    if (missing === "decision") h.decision_python = { ok: false, message: "dependency import failed" };
    expect(workflows(h).decision.status).toBe("needs_setup");
  });
  it("parses MPS availability separately from CUDA", () => {
    expect(parsePythonProbeJson('{"python_ok":true,"cuda_available":false,"mps_available":true}'))
      .toMatchObject({ cuda_available: false, mps_available: true });
    expect(parsePythonProbeJson('{"python_ok":true}')).not.toHaveProperty("mps_available", true);
  });
  it("refreshes resource pressure while retaining fresh checks, and drops stale checks", () => {
    const saved = host();
    const live = { sampled_at: saved.collected_at, platform: "linux" as const, arch: "x64", cpu: { model: "CPU", cores: 8, load_average: [0, 0, 0] as [number, number, number] }, memory: { total_bytes: 32 * GiB, used_bytes: 30 * GiB }, gpus: [] };
    const context = { cwd: "/test", projectName: "test", local: { artifactRoot: "/test", storeRoot: "/test" }, warnings: [], host: { collectedAt: saved.collected_at, stale: false, summary: "", line: "", inventory: saved } };
    const decision = () => machineCapabilities(live, context).workflows!.find(w => w.id === "decision")!;
    expect(decision().status).toBe("limited");
    live.memory.used_bytes = 4 * GiB;
    expect(decision().status).toBe("ready");
    context.host.stale = true;
    expect(decision().status).toBe("unverified");
  });
  it("keeps per-model fit separate from the smallest supported model", () => {
    const h = gpu(host(), 128, 24);
    const live = { sampled_at: h.collected_at, platform: "linux" as const, arch: "x64", cpu: { model: "CPU", cores: 8, load_average: [0, 0, 0] as [number, number, number] }, memory: { total_bytes: 256 * GiB, used_bytes: 4 * GiB }, gpus: [{ index: 0, name: h.gpus[0]!.name, vendor: "nvidia" as const, unified_memory: false, memory_total_bytes: 128 * GiB, memory_used_bytes: 104 * GiB }] };
    const context = { cwd: "/test", projectName: "test", local: { artifactRoot: "/test", storeRoot: "/test" }, warnings: [], host: { collectedAt: h.collected_at, stale: false, summary: "", line: "", inventory: h }, spec: { path: "/test/tunedtensor.json", parseError: false, baseModel: "meta-models/Muse-Glimmer-30B" } };
    expect(machineCapabilities(live, context).workflows!.find(w => w.id === "adapter")!.status).toBe("ready");
    expect(describeFineTuneFit(live, context)).toMatchObject({ status: "tight", text: expect.stringContaining("insufficient free memory") });
    live.gpus[0]!.name = "different GPU";
    expect(machineCapabilities(live, context).workflows!.find(w => w.id === "adapter")!.status).toBe("unverified");
  });
  it("upgrades old cached reports without falsely verifying their decision runtime", async () => {
    const root = await mkdtemp(join(tmpdir(), "tt-hardware-legacy-"));
    try {
      const h = host(); delete h.decision_python;
      const caps = evaluateCapabilities(h); delete caps.workflows;
      const env = { TUNED_TENSOR_HOME: root };
      await writeFile(hardwareSnapshotPath(env), JSON.stringify({ version: 1, collected_at: h.collected_at, quick: false, inventory: h, capabilities: caps, summary: "old CUDA-only summary" }));
      const restored = await readHardwareSnapshot(env);
      expect(restored?.summary).toContain("Decision training: Needs verification");
      expect(restored?.stale).toBe(false);
    } finally { await rm(root, { recursive: true, force: true }); }
  });
});

describe("native host and terminal smoke (runs on Linux, macOS and Windows CI)", () => {
  it("samples the real CPU/RAM and persists guidance with optional programs absent", async () => {
    const root = await mkdtemp(join(tmpdir(), "tt-hardware-native-"));
    try {
      const env = { ...process.env, PATH: join(root, "absent-bin"), Path: join(root, "absent-bin"), TUNED_TENSOR_HOME: root };
      const live = await sampleLiveUsage({ env, cpuSampleMs: 10 });
      expect(live.platform).toBe(process.platform);
      expect(live.cpu.cores).toBeGreaterThan(0);
      expect(live.memory.total_bytes).toBeGreaterThan(0);
      expect(inventoryFromUsage(live).os.platform).toBe(process.platform);
      const report = await assessHardware({ quick: true, env, cwd: root });
      expect(report.inventory.os.arch).toBe(process.arch);
      expect(report.capabilities.workflows?.map(w => w.id)).toEqual(["prepare", "decision", "adapter", "foundation", "serve"]);
      expect((await readHardwareSnapshot(env))?.summary).toBe(formatCapabilitySummary(report.capabilities));
      const context = { cwd: root, projectName: "test", local: { artifactRoot: root, storeRoot: root }, warnings: [] };
      for (const columns of [40, 80, 120]) {
        const lines = renderMachinePanel(live, context, { columns });
        const text = stripVTControlCharacters(lines.join("\n"));
        expect(text).toContain("WHAT YOU CAN DO");
        expect(text).toContain("Decision training");
        expect(text).toContain("Specs and datasets");
        expect(lines.filter(line => terminalWidth(line) > Math.min(columns, 88)).map(stripVTControlCharacters)).toEqual([]);
      }
    } finally { await rm(root, { recursive: true, force: true }); }
  });
});
