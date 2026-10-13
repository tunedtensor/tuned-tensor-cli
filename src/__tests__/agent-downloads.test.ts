import { describe, expect, it, vi } from "vitest";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { approvePreparedAction } from "../agent-approval.js";
import type { AgentAction } from "../agent-client.js";
import { createTunedTensorTools, type AgentToolApi } from "../agent-tools.js";

const spec = {
  name: "Chat",
  base_model: "Qwen/Qwen3.5-2B",
  system_prompt: "Be helpful.",
  examples: [],
  dataset_prebuilt: {
    huggingface: { repo: "HuggingFaceH4/ultrachat_200k", columns: { messages: "messages" } },
    training: "data/train_sft-00000-of-00003.parquet",
    test: "data/test_sft-00000-of-00001.parquet",
  },
};

function workspace(document: unknown = spec): string {
  const root = mkdtempSync(join(tmpdir(), "tt-agent-download-"));
  writeFileSync(join(root, "tunedtensor.json"), JSON.stringify(document));
  return root;
}

async function prepare(root: string, kind: "dataset" | "base_model"): Promise<AgentAction> {
  const api: AgentToolApi = {
    get: vi.fn(),
    postRead: vi.fn(),
    propose: vi.fn(async (action: AgentAction) => action),
  };
  const tool = createTunedTensorTools(api, { workspaceRoot: root, localOnly: true })
    .find((candidate) => candidate.name === "prepare_download")!;
  await tool.execute("download", { kind });
  return vi.mocked(api.propose).mock.calls[0]![0];
}

describe("agent downloads", () => {
  it("prepares a dataset download and runs tt datasets prefetch only on approval", async () => {
    const root = workspace();
    try {
      const action = await prepare(root, "dataset");
      expect(action).toMatchObject({
        operation: "prefetch_local",
        status: "proposed",
        arguments: { kind: "dataset", spec_path: "./tunedtensor.json", spec_sha256: expect.stringMatching(/^[a-f0-9]{64}$/) },
        preview: {
          repo: "HuggingFaceH4/ultrachat_200k",
          files: { training: spec.dataset_prebuilt.training, test: spec.dataset_prebuilt.test },
          command: ["tt", "datasets", "prefetch", "./tunedtensor.json"],
        },
      });
      const runPipelineCommand = vi.fn(async () => ({ exitCode: 0, signal: null }));
      const output = await approvePreparedAction(action, { get: vi.fn(), post: vi.fn(), put: vi.fn() }, async () => {}, {
        workspaceRoot: root,
        runPipelineCommand,
      });
      expect(runPipelineCommand).toHaveBeenCalledWith(["datasets", "prefetch", "tunedtensor.json"], expect.objectContaining({ cwd: expect.any(String) }));
      expect(output).toMatchObject({ completed: true, kind: "dataset" });
      expect(action.status).toBe("completed");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("downloads the base model and records a failed download as retryable", async () => {
    const root = workspace();
    try {
      const action = await prepare(root, "base_model");
      expect(action.preview).toMatchObject({ base_model: "Qwen/Qwen3.5-2B", command: ["tt", "models", "prefetch", "./tunedtensor.json"] });
      const runPipelineCommand = vi.fn(async () => ({ exitCode: 1, signal: null }));
      await expect(approvePreparedAction(action, { get: vi.fn(), post: vi.fn(), put: vi.fn() }, async () => {}, {
        workspaceRoot: root,
        runPipelineCommand,
      })).rejects.toThrow(/models prefetch tunedtensor\.json exited with code 1/);
      expect(runPipelineCommand).toHaveBeenCalledWith(["models", "prefetch", "tunedtensor.json"], expect.anything());
      expect(action.status).toBe("failed");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("refuses a download after the reviewed spec changed", async () => {
    const root = workspace();
    try {
      const action = await prepare(root, "dataset");
      writeFileSync(join(root, "tunedtensor.json"), JSON.stringify({ ...spec, name: "Changed" }));
      const runPipelineCommand = vi.fn();
      await expect(approvePreparedAction(action, { get: vi.fn(), post: vi.fn(), put: vi.fn() }, async () => {}, {
        workspaceRoot: root,
        runPipelineCommand,
      })).rejects.toThrow(/spec changed/);
      expect(runPipelineCommand).not.toHaveBeenCalled();
      expect(action.status).toBe("failed");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("explains when there is nothing to download", async () => {
    const { dataset_prebuilt: _dataset, ...local } = spec;
    const root = workspace({ ...local, examples: [{ input: "a", output: "b" }, { input: "c", output: "d" }] });
    try {
      await expect(prepare(root, "dataset")).rejects.toThrow(/no dataset_prebuilt\.huggingface source/);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});
