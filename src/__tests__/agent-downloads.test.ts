import { describe, expect, it, vi } from "vitest";
import { existsSync, mkdtempSync, realpathSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { approvePreparedAction } from "../agent-approval.js";
import type { AgentAction } from "../agent-client.js";
import { createTunedTensorTools, type AgentToolApi } from "../agent-tools.js";
import { resolveProjectConfig } from "../local-runtime/project-workflow.js";
import { LocalAgentStore } from "../agent-store.js";
import { createLocalAgentClient } from "../local-agent-client.js";

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
  const root = realpathSync(mkdtempSync(join(tmpdir(), "tt-agent-download-")));
  writeFileSync(join(root, "tunedtensor.json"), JSON.stringify(document));
  return root;
}

async function prepare(root: string, kind: "dataset" | "base_model", source?: Record<string, unknown>): Promise<AgentAction> {
  const api: AgentToolApi = {
    get: vi.fn(),
    postRead: vi.fn(),
    propose: vi.fn(async (action: AgentAction) => action),
  };
  const tool = createTunedTensorTools(api, { workspaceRoot: root, localOnly: true })
    .find((candidate) => candidate.name === "prepare_download")!;
  await tool.execute("download", { kind, ...(source ? { source } : {}) });
  return vi.mocked(api.propose).mock.calls[0]![0];
}

const approve = (action: AgentAction, root: string, runPipelineCommand: (...args: never[]) => unknown) =>
  approvePreparedAction(action, { get: vi.fn(), post: vi.fn(), put: vi.fn() }, async () => {}, {
    workspaceRoot: root,
    runPipelineCommand: runPipelineCommand as never,
  });

describe("agent downloads", () => {
  it.each(["base_model", "dataset"] as const)("carries a completed %s download into the next conversation turn", async (kind) => {
    const root = workspace();
    try {
      const prompts: string[] = [];
      const store = new LocalAgentStore(join(root, ".agent"));
      const repo = kind === "base_model" ? "org/model" : "org/data";
      const revision = "1".repeat(40);
      const client = createLocalAgentClient({
        store, workspaceRoot: root,
        selection: { provider: "test", model: "test", thinking: "off" },
        modelRuntime: {
          getProviders: () => [{ id: "test" }], getModels: () => [{ id: "test", provider: "test" }],
          getModel: () => ({ id: "test", provider: "test" }), hasConfiguredAuth: () => true,
        },
        toolApi: { get: vi.fn(), postRead: vi.fn(), propose: async (action) => action },
        mutationApi: { get: vi.fn(), post: vi.fn(), put: vi.fn() },
        runPipelineCommand: async () => ({ exitCode: 0, stdout: JSON.stringify({
          repo, revision, snapshot_path: "/cache/snapshot", manifest_path: "/assets/manifest.json",
          file_count: 13, size_bytes: 59581829216, verified_blob_count: 13,
          files: { training: { path: "train.jsonl" } },
        }) }),
        createAgent: (options) => ({
          state: { messages: [] }, subscribe: () => () => {}, abort: () => {},
          prompt: async (text) => {
            prompts.push(text);
            if (prompts.length === 1) await options.tools.find((tool) => tool.name === "prepare_download")!
              .execute("download", { kind, source: { repo, ...(kind === "dataset" ? { training: "train.jsonl" } : {}) } });
          },
        }),
      });
      const thread = await client.createThread();
      const turn = await client.runTurn(thread.id, "Download it", () => {});
      await client.approveAction(turn.actions[0]!.id, () => {});
      expect((await store.load(thread.id)).actions[0]?.result).toMatchObject({ repo, revision, snapshot_path: "/cache/snapshot" });
      await client.runTurn(thread.id, "Do we have it already?", () => {});
      expect(prompts[1]).toContain('"status":"completed"');
      expect(prompts[1]).toContain(repo);
      expect(prompts[1]).toContain(revision);
      expect(prompts[1]).toContain("/cache/snapshot");
      if (kind === "dataset") expect(prompts[1]).toContain("train.jsonl");
      else expect(prompts[1]).toContain('"checksums_verified":true');
      await client.runTurn(thread.id, "hello", () => {});
      expect(prompts[2]).toBe("hello");
    } finally { rmSync(root, { recursive: true, force: true }); }
  });

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
      expect(runPipelineCommand).toHaveBeenCalledWith(["datasets", "prefetch", expect.stringContaining(".tt-agent-")], expect.objectContaining({ cwd: expect.any(String) }));
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
      expect(runPipelineCommand).toHaveBeenCalledWith(["models", "prefetch", expect.stringContaining(".tt-agent-")], expect.anything());
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

  it("downloads a dataset the spec does not name without editing the spec", async () => {
    const root = workspace();
    try {
      const before = readFileSync(join(root, "tunedtensor.json"), "utf8");
      const action = await prepare(root, "dataset", { repo: "org/new-data", training: "train.jsonl", test: "test.jsonl" });
      expect(action.preview).toMatchObject({
        repo: "org/new-data",
        revision: "latest (resolved by the download)",
        files: { training: "train.jsonl", test: "test.jsonl" },
        command: ["tt", "datasets", "prefetch", "./tunedtensor.json", "--repo", "org/new-data", "--training", "train.jsonl", "--test", "test.jsonl"],
      });
      const runner = vi.fn(async (args: string[]) => {
        // The spec only supplies the cache location; the source comes from the flags.
        expect(args.slice(3)).toEqual(["--repo", "org/new-data", "--training", "train.jsonl", "--test", "test.jsonl"]);
        return { exitCode: 0, signal: null };
      });
      expect(await approve(action, root, runner)).toMatchObject({ completed: true, repo: "org/new-data" });
      expect(runner).toHaveBeenCalledOnce();
      expect(readFileSync(join(root, "tunedtensor.json"), "utf8")).toBe(before);
    } finally { rmSync(root, { recursive: true, force: true }); }
  });

  it("downloads a model by repo even when the spec is a decision spec", async () => {
    const root = workspace({ name: "Router", engine: "decision", base_model: "convaiinnovations/laya", decision: { type: "noul" }, examples: [] });
    try {
      const action = await prepare(root, "base_model", { repo: "org/other-model" });
      expect(action.preview).toMatchObject({ base_model: "org/other-model", certified: false });
      const runner = vi.fn(async (args: string[]) => {
        // No adapter spec: a snapshot path that does not exist selects the default cache.
        expect(existsSync(args[2]!)).toBe(false);
        expect(args.slice(3)).toEqual(["--repo", "org/other-model"]);
        return { exitCode: 0, signal: null };
      });
      await approve(action, root, runner);
      expect(runner).toHaveBeenCalledOnce();
    } finally { rmSync(root, { recursive: true, force: true }); }
  });

  it("returns the revision from this download's report for a follow-up spec edit", async () => {
    const root = workspace();
    try {
      const revision = "1".repeat(40);
      const action = await prepare(root, "dataset", { repo: "org/data", training: "train.jsonl", revision });
      const runner = vi.fn(async (_args: string[], options: { captureJson?: boolean }) => {
        expect(options.captureJson).toBe(true);
        return { exitCode: 0, stdout: JSON.stringify({ repo: "org/data", revision, snapshot_path: "/cache/snapshot", files: { training: { path: "train.jsonl" } } }) };
      });
      expect(await approve(action, root, runner)).toMatchObject({ revision, files: ["train.jsonl"], snapshot_path: "/cache/snapshot" });
      const wrong = await prepare(root, "base_model", { repo: "org/model", revision });
      await expect(approve(wrong, root, vi.fn(async () => ({ exitCode: 0, stdout: JSON.stringify({ base_model: "org/model", snapshot_revision: "2".repeat(40) }) })))).rejects.toThrow(/different revision/);
    } finally { rmSync(root, { recursive: true, force: true }); }
  });

  it("downloads a source before a spec exists, and refuses a spec added after review", async () => {
    const root = workspace();
    try {
      rmSync(join(root, "tunedtensor.json"));
      const source = { repo: "org/data", training: "train.jsonl" };
      const action = await prepare(root, "dataset", source);
      expect(await approve(action, root, vi.fn(async () => ({ exitCode: 0 })))).toMatchObject({ completed: true });
      expect(existsSync(join(root, "tunedtensor.json"))).toBe(false);
      const stale = await prepare(root, "dataset", source);
      writeFileSync(join(root, "tunedtensor.json"), JSON.stringify(spec));
      const runner = vi.fn();
      await expect(approve(stale, root, runner)).rejects.toThrow(/spec changed/);
      expect(runner).not.toHaveBeenCalled();
    } finally { rmSync(root, { recursive: true, force: true }); }
  });

  it("uses the sealed cache configuration despite a legacy config beside the snapshot", async () => {
    const root = workspace();
    try {
      writeFileSync(join(root, "local-runner.json"), JSON.stringify({ paths: { modelCache: "./reviewed-cache" } }));
      const action = await prepare(root, "dataset");
      const runner = vi.fn(async (args: string[]) => {
        writeFileSync(join(root, "local-runner.json"), JSON.stringify({ paths: { modelCache: "./changed-cache" } }));
        expect((await resolveProjectConfig(args[2]!)).config.paths.modelCache).toBe(join(root, "reviewed-cache"));
        return { exitCode: 0 };
      });
      await approve(action, root, runner);
    } finally { rmSync(root, { recursive: true, force: true }); }
  });

  it("rejects a dataset source without a training file", async () => {
    const root = workspace();
    try {
      await expect(prepare(root, "dataset", { repo: "org/new-data" })).rejects.toThrow(/source\.training/);
      await expect(prepare(root, "dataset", { repo: "org/new-data", training: "../escape.jsonl" })).rejects.toThrow(/inside the repo/);
    } finally { rmSync(root, { recursive: true, force: true }); }
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

  it("executes the reviewed snapshot even if the source changes during dispatch", async () => {
    const root = workspace({ ...spec, runtime: { paths: { modelCache: "./cache" } } });
    let executedPath = "";
    try {
      const action = await prepare(root, "dataset");
      const runner = vi.fn(async (args: string[]) => {
        writeFileSync(join(root, "tunedtensor.json"), JSON.stringify({ ...spec, base_model: "changed/model" }));
        executedPath = args[2]!;
        const executed = JSON.parse(readFileSync(resolve(root, executedPath), "utf8"));
        expect(executed.base_model).toBe(spec.base_model);
        expect(executed.runtime.paths.modelCache).toBe(join(root, "cache"));
        return { exitCode: 0, signal: null };
      });
      await approvePreparedAction(action, { get: vi.fn(), post: vi.fn(), put: vi.fn() }, async () => {}, { workspaceRoot: root, runPipelineCommand: runner });
      expect(existsSync(executedPath)).toBe(false);
    } finally { rmSync(root, { recursive: true, force: true }); }
  });

  it("refuses to redirect approval to another workspace with an identical spec", async () => {
    const root = workspace();
    const other = workspace();
    try {
      const action = await prepare(root, "dataset");
      const runner = vi.fn();
      await expect(approvePreparedAction(action, { get: vi.fn(), post: vi.fn(), put: vi.fn() }, async () => {}, {
        workspaceRoot: other, runPipelineCommand: runner,
      })).rejects.toThrow(/workspace changed/i);
      expect(runner).not.toHaveBeenCalled();
    } finally { rmSync(root, { recursive: true, force: true }); rmSync(other, { recursive: true, force: true }); }
  });

  it("refuses changes to the adjacent legacy cache configuration", async () => {
    const root = workspace();
    try {
      writeFileSync(join(root, "local-runner.json"), JSON.stringify({ paths: { modelCache: "./first" } }));
      const action = await prepare(root, "dataset");
      writeFileSync(join(root, "local-runner.json"), JSON.stringify({ paths: { modelCache: "./second" } }));
      const runner = vi.fn();
      await expect(approvePreparedAction(action, { get: vi.fn(), post: vi.fn(), put: vi.fn() }, async () => {}, {
        workspaceRoot: root, runPipelineCommand: runner,
      })).rejects.toThrow(/local-runner.json changed/);
      expect(runner).not.toHaveBeenCalled();
    } finally { rmSync(root, { recursive: true, force: true }); }
  });

  it("records a thrown command failure as failed instead of outcome unknown", async () => {
    const root = workspace();
    try {
      const action = await prepare(root, "dataset");
      await expect(approvePreparedAction(action, { get: vi.fn(), post: vi.fn(), put: vi.fn() }, async () => {}, {
        workspaceRoot: root, runPipelineCommand: async () => { throw new Error("spawn failed"); },
      })).rejects.toThrow(/spawn failed/);
      expect(action.status).toBe("failed");
    } finally { rmSync(root, { recursive: true, force: true }); }
  });
});
