import { describe, expect, it } from "vitest";
import { evaluateCapabilities, warningsFromSnapshot } from "../local-runtime/capability.js";
import { gib } from "../local-runtime/capability-profiles.js";
import type { HostInventory } from "../local-runtime/host-inventory.js";
import { formatHostStatusLine } from "../local-runtime/hardware-snapshot.js";

function inventory(overrides: Partial<HostInventory> = {}): HostInventory {
  return {
    collected_at: "2026-08-29T12:00:00.000Z",
    quick: true,
    node: { version: "22.19.0", major: 22, ok: true },
    os: {
      platform: "linux",
      arch: "arm64",
      type: "Linux",
      cpu_count: 20,
      total_memory_bytes: gib(128),
      free_memory_bytes: gib(80),
    },
    uv: { ok: true, version: "uv 0.8.0", message: "uv 0.8.0" },
    gpus: [],
    nvidia_smi: { ok: false, message: "nvidia-smi not available" },
    disks: [{
      name: "model-cache",
      path: "/tmp/hf",
      free_bytes: gib(200),
      ok: true,
      message: "ok",
    }],
    ...overrides,
  };
}

describe("hardware capability verdicts", () => {
  it("marks adapter train impossible without CUDA and allows CPU inference for Qwen", () => {
    const report = evaluateCapabilities(inventory({
      os: {
        platform: "linux",
        arch: "x64",
        type: "Linux",
        cpu_count: 8,
        total_memory_bytes: gib(32),
        free_memory_bytes: gib(16),
      },
    }));
    const qwen = report.adapters.find((item) => item.id === "Qwen/Qwen3.5-2B")!;
    const nemotron = report.adapters.find((item) => item.id.includes("Nemotron"))!;
    expect(report.cuda_available).toBe(false);
    expect(qwen.train.status).toBe("not_possible");
    expect(qwen.inference.status).toBe("ready");
    expect(nemotron.inference.status).toBe("not_possible");
    expect(report.foundation.train.status).toBe("not_possible");
    expect(report.foundation.serve.status).toBe("not_possible");
  });

  it("fits Qwen LoRA on a 24 GiB GPU and rejects Spark-class 30B models", () => {
    const report = evaluateCapabilities(inventory({
      nvidia_smi: { ok: true, message: "RTX 4090" },
      gpus: [{
        index: 0,
        name: "NVIDIA GeForce RTX 4090",
        memory_total_bytes: gib(24),
        unified_memory: false,
      }],
    }));
    const qwen = report.adapters.find((item) => item.id === "Qwen/Qwen3.5-2B")!;
    const muse = report.adapters.find((item) => item.id.includes("Muse"))!;
    expect(qwen.train.status).toBe("ready");
    expect(muse.train.status).toBe("not_possible");
    expect(report.foundation.train.status).toBe("ready");
    expect(report.foundation.suggested_max_depth).toBeGreaterThan(2);
  });

  it("treats Spark unified memory as enough for Nemotron train", () => {
    const report = evaluateCapabilities(inventory({
      nvidia_smi: { ok: true, message: "NVIDIA GB10" },
      gpus: [{
        index: 0,
        name: "NVIDIA GB10",
        memory_total_bytes: gib(128),
        unified_memory: true,
      }],
    }));
    const nemotron = report.adapters.find((item) => item.id.includes("Nemotron"))!;
    expect(report.notes.some((note) => /unified/i.test(note))).toBe(true);
    expect(nemotron.train.status).toBe("ready");
    expect(report.foundation.suggested_max_depth).toBeGreaterThanOrEqual(32);
    expect(warningsFromSnapshot(report, {
      engine: "adapter",
      baseModel: "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    })).toEqual([]);
  });

  it("formats a missing snapshot as a status hint", () => {
    expect(formatHostStatusLine(undefined)).toContain("tt hardware");
  });
});
