import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { parquetWriteBuffer } from "hyparquet-writer";
import { listLocalDatasets, inspectLocalDataset } from "../agent-local-datasets.js";
import { recordAssetDownload } from "../local-runtime/assets.js";
import { executeDatasetSplit, prepareDatasetSplit } from "../local-runtime/dataset-split.js";
import { createTunedTensorTools } from "../agent-tools.js";

const repo = "org/coding";
const revision = "a".repeat(40);
let root: string;
let snapshot: string;

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), "tt-local-dataset-"));
  vi.stubEnv("TUNED_TENSOR_HOME", join(root, "tt"));
  snapshot = join(root, "hub", "datasets--org--coding", "snapshots", revision);
  await mkdir(join(snapshot, "data"), { recursive: true });
});
afterEach(async () => { vi.unstubAllEnvs(); await rm(root, { recursive: true, force: true }); });

async function cache(files: Record<string, string | Uint8Array>) {
  for (const [file, bytes] of Object.entries(files)) await writeFile(join(snapshot, file), bytes);
  return await recordAssetDownload({ kind: "dataset", repo, revision, snapshotPath: snapshot, paths: Object.keys(files) });
}

describe("agent access to downloaded datasets", () => {
  it("splits a recorded cached chat dataset into the workspace and rejects changed cache bytes", async () => {
    const data = Array.from({ length: 100 }, (_, i) => JSON.stringify({ messages: [
      { role: "user", content: `Request ${i}` }, { role: "assistant", content: `Answer ${i}` },
    ] })).join("\n") + "\n";
    await cache({ "data/train.jsonl": data });
    const workspace = join(root, "workspace");
    await mkdir(workspace);
    const source = `hf://datasets/${repo}@${revision}/data/train.jsonl`;
    const inspected = await inspectLocalDataset(repo, revision, "data/train.jsonl");
    expect(inspected).toMatchObject({ split_source: source });
    const prepared = await prepareDatasetSplit(workspace, { file: source, output: "split" });
    const result = await executeDatasetSplit(workspace, prepared);
    expect(result).toMatchObject({ source, counts: { training: 80, validation: 10, test: 10 } });
    expect(await readFile(join(snapshot, "data/train.jsonl"), "utf8")).toBe(data);
    await writeFile(join(snapshot, "data/train.jsonl"), data.replace("Answer 0", "Tampered"));
    await expect(prepareDatasetSplit(workspace, { file: source, output: "changed" })).rejects.toThrow(/checksum/);
    await expect(prepareDatasetSplit(workspace, { file: source.replace("train.jsonl", "unknown.jsonl"), output: "unknown" })).rejects.toThrow(/recorded file/);
  });

  it("splits verified git-blob symlinks and refuses a link escaping the cache", async () => {
    const data = Array.from({ length: 10 }, (_, i) => JSON.stringify({ messages: [
      { role: "user", content: `Request ${i}` }, { role: "assistant", content: "label" },
    ] })).join("\n") + "\n";
    const blobs = join(root, "hub", "datasets--org--coding", "blobs");
    await mkdir(blobs);
    const digest = createHash("sha1").update(`blob ${Buffer.byteLength(data)}\0`).update(data).digest("hex");
    await writeFile(join(blobs, digest), data);
    const file = join(snapshot, "data/train.jsonl");
    await symlink(join(blobs, digest), file);
    await recordAssetDownload({ kind: "dataset", repo, revision, snapshotPath: snapshot, paths: ["data/train.jsonl"] });
    const workspace = join(root, "workspace");
    await mkdir(workspace);
    const source = `hf://datasets/${repo}@${revision}/data/train.jsonl`;
    const proposal = await prepareDatasetSplit(workspace, { file: source, output: "split" });
    expect(await executeDatasetSplit(workspace, proposal)).toMatchObject({ counts: { training: 8, validation: 1, test: 1 } });
    await rm(file);
    await writeFile(join(root, "outside.jsonl"), data);
    await symlink(join(root, "outside.jsonl"), file);
    await expect(prepareDatasetSplit(workspace, { file: source, output: "escaped" })).rejects.toThrow(/outside its cache/);
  });

  it("lists actual local downloads and reads the Parquet columns and bounded sample without network", async () => {
    const parquet = new Uint8Array(parquetWriteBuffer({ columnData: [
      { name: "prompt", data: ["Write Python", "Fix Python", "Third row"], type: "STRING" },
      { name: "completion", data: ["x".repeat(2000), "print(1)", "not sampled"], type: "STRING" },
    ] }));
    await cache({ "data/train.parquet": parquet, "data/test.parquet": parquet });
    const network = vi.fn(() => { throw new Error("Unexpected network"); });
    vi.stubGlobal("fetch", network);
    try {
      const result = await listLocalDatasets();
      expect(result.datasets).toMatchObject([{ repo, revision, file_count: 2, files: [{ available: true }, { available: true }] }]);
      const inspected = await inspectLocalDataset(repo, revision, "data/train.parquet");
      expect(inspected).toMatchObject({ columns: ["prompt", "completion"], row_count: "3", schema_source: "parquet_metadata" });
      expect(inspected.sample).toHaveLength(2);
      expect(JSON.stringify(inspected.sample)).toContain("[truncated]");
      expect(JSON.stringify(inspected.sample)).not.toContain("not sampled");
      expect(network).not.toHaveBeenCalled();
    } finally { vi.unstubAllGlobals(); }
  });

  it("inspects chat JSONL, preserves cached bytes, and distinguishes sampled fields from a full schema", async () => {
    const data = JSON.stringify({ messages: [{ role: "user", content: "Hello" }, { role: "assistant", content: "Hi" }] }) + "\n";
    await cache({ "data/train.jsonl": data });
    const result = await inspectLocalDataset(repo, revision, "data/train.jsonl");
    expect(result).toMatchObject({ columns: ["messages"], schema_source: "sampled_jsonl_records", row_count: null });
    expect(result.sample).toEqual([JSON.parse(data)]);
    expect(await readFile(join(snapshot, "data/train.jsonl"), "utf8")).toBe(data);
  });

  it("reports missing files and rejects unrecorded paths, traversal, and escaping cache symlinks", async () => {
    await cache({ "data/train.jsonl": '{"input":"a","output":"b"}\n' });
    await expect(inspectLocalDataset(repo, revision, "../train.jsonl")).rejects.toThrow(/exact recorded file/);
    await expect(inspectLocalDataset(repo, revision, "data/unknown.jsonl")).rejects.toThrow(/exact recorded file/);
    await expect(inspectLocalDataset(repo, "b".repeat(40), "data/train.jsonl")).rejects.toThrow(/No TT-downloaded/);
    await rm(join(snapshot, "data/train.jsonl"));
    expect((await listLocalDatasets()).datasets[0]!.files[0]!.available).toBe(false);
    await writeFile(join(root, "private.jsonl"), '{"secret":"do not read"}\n');
    await symlink(join(root, "private.jsonl"), join(snapshot, "data/train.jsonl"));
    await expect(inspectLocalDataset(repo, revision, "data/train.jsonl")).rejects.toThrow(/outside its cache/);
  });

  it("supports normal Hugging Face blob symlinks and refuses oversized JSONL records", async () => {
    await cache({ "data/train.jsonl": '{"input":"a","output":"b"}\n' });
    const blobs = join(root, "hub", "datasets--org--coding", "blobs");
    await mkdir(blobs);
    await writeFile(join(blobs, "blob"), '{"input":"a","output":"b"}\n');
    await rm(join(snapshot, "data/train.jsonl"));
    await symlink(join(blobs, "blob"), join(snapshot, "data/train.jsonl"));
    expect((await inspectLocalDataset(repo, revision, "data/train.jsonl")).columns).toEqual(["input", "output"]);
    await writeFile(join(blobs, "blob"), JSON.stringify({ input: "x".repeat(300000), output: "b" }) + "\n");
    await expect(inspectLocalDataset(repo, revision, "data/train.jsonl")).rejects.toThrow(/first 256 KiB/);
  });

  it("exposes local inspection without TT account access and paginates downloads", async () => {
    await cache({ "data/train.jsonl": '{"input":"a","output":"b"}\n' });
    const forbidden = vi.fn(async (): Promise<never> => { throw new Error("Unexpected hosted API call"); });
    const tools = createTunedTensorTools({ get: forbidden, postRead: forbidden, propose: forbidden }, { localOnly: true });
    expect(tools.some((tool) => tool.name === "list_datasets")).toBe(false);
    const list = await tools.find((tool) => tool.name === "list_local_datasets")!.execute("list", {});
    expect(list.details).toMatchObject({ total: 1 });
    const inspect = await tools.find((tool) => tool.name === "inspect_local_dataset")!.execute("inspect", { repo, revision, file: "data/train.jsonl" });
    expect(inspect.details).toMatchObject({ columns: ["input", "output"] });
    expect((await listLocalDatasets(2, 1)).datasets).toEqual([]);
    expect(forbidden).not.toHaveBeenCalled();
  });
});
