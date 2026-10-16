import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { mkdtemp, readFile, readdir, rename, rm, symlink, writeFile } from "node:fs/promises";
import { readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { executeDatasetSplit, inspectDatasetFile, prepareDatasetSplit } from "../local-runtime/dataset-split.js";
import { examplesFromChatJsonl } from "../local-runtime/dataset.js";

vi.mock("node:fs/promises", async (importOriginal) => {
  const fs = await importOriginal<typeof import("node:fs/promises")>();
  return { ...fs, rename: vi.fn(fs.rename) };
});
let root: string;
const row = (n: number) => ({ id: n, group_id: String(Math.floor(n / 2)), metadata: { category: n < 50 ? "a" : "b" },
  messages: [{ role: "user", content: `Request ${n}` }, { role: "assistant", content: `Answer ${n}` }] });
async function save(rows = Array.from({ length: 100 }, (_, i) => row(i))) {
  await writeFile(join(root, "data.jsonl"), rows.map(r => JSON.stringify(r)).join("\n") + "\n");
}
async function split(output: string, extra = {}) {
  return executeDatasetSplit(root, await prepareDatasetSplit(root, { file: "data.jsonl", output, ...extra }));
}
async function rows(output: string, part: string) {
  return (await readFile(join(root, output, `${part}.jsonl`), "utf8")).trim().split("\n").map(s => JSON.parse(s) as ReturnType<typeof row>);
}
beforeEach(async () => { root = await mkdtemp(join(tmpdir(), "tt-split-")); await save(); });
afterEach(async () => { await rm(root, { recursive: true, force: true }); });

it("validates the entire source, reports usable metadata fields, and keeps preparation read-only", async () => {
  expect(await inspectDatasetFile(root, "data.jsonl")).toMatchObject({ rows: 100, fields: expect.arrayContaining(["group_id", "metadata.category"]) });
  const p = await prepareDatasetSplit(root, { file: "data.jsonl", output: "split" });
  expect(p.preview.counts).toEqual({ training: 80, validation: 10, test: 10 });
  expect(await readdir(root)).toEqual(["data.jsonl"]);
});
it("reproduces seeded memberships, changes them with a new seed and preserves every record", async () => {
  const original = await readFile(join(root, "data.jsonl"), "utf8");
  await split("a"); await split("b"); await split("c", { seed: "123" });
  const all = [];
  for (const part of ["training", "validation", "test"]) {
    expect(await rows("a", part)).toEqual(await rows("b", part));
    all.push(...await rows("a", part));
    expect(await examplesFromChatJsonl(join(root, "a", `${part}.jsonl`))).toHaveLength((await rows("a", part)).length);
  }
  expect(all.sort((a,b) => a.id-b.id)).toEqual(Array.from({ length: 100 }, (_,i) => row(i)));
  expect(await rows("a", "training")).not.toEqual(await rows("c", "training"));
  expect(await readFile(join(root, "data.jsonl"), "utf8")).toBe(original);
});
it("keeps custom groups together while stratifying and supports ordered two-way splitting", async () => {
  await split("grouped", { group_by: "group_id", stratify_by: "metadata.category" });
  const seen = new Set<string>();
  for (const part of ["training", "validation", "test"]) {
    const data = await rows("grouped", part);
    const groups = new Set(data.map(r => r.group_id));
    for (const g of groups) { expect(seen.has(g)).toBe(false); seen.add(g); }
    expect(data.filter(r => r.metadata.category === "a")).toHaveLength(data.length / 2);
  }
  const result = await split("ordered", { method: "ordered", train: .9, validation: .1, test: 0 });
  expect(result.dataset_prebuilt.test).toBeUndefined();
  expect((await rows("ordered", "training")).map(r => r.id)).toEqual(Array.from({ length: 90 }, (_,i) => i));
});
it("keeps normalized duplicate prompts together and refuses conflicting explicit groups", async () => {
  const data = Array.from({ length: 100 }, (_,i) => row(i));
  data[99]!.messages[0]!.content = "  REQUEST   0  ";
  await save(data); await split("dup");
  for (const part of ["training", "validation", "test"]) {
    const ids = (await rows("dup", part)).map(r => r.id);
    expect(ids.includes(0)).toBe(ids.includes(99));
  }
  await expect(split("bad", { group_by: "group_id" })).rejects.toThrow(/Duplicate prompts/);
});
it.each(["ordered", "random"])("keeps repeated prompts in training so held-out rows remain usable (%s)", async (method) => {
  const data = Array.from({ length: 100 }, (_, i) => row(i));
  data[99]!.messages[0]!.content = "  REQUEST   98  ";
  await save(data);
  for (const group_by of [undefined, "group_id"]) {
    const output = group_by ?? "prompt";
    const result = await split(output, { method, group_by, stratify_by: "metadata.category" });
    expect(result.training_only_rows).toBe(2);
    expect(result.counts).toEqual(group_by
      ? { training: 80, validation: 8, test: 12 }
      : { training: 80, validation: 10, test: 10 });
    expect((await rows(output, "training")).map(r => r.id)).toEqual(expect.arrayContaining([98, 99]));
    for (const part of ["validation", "test"]) {
      expect((await rows(output, part)).some(r => r.id >= 98)).toBe(false);
    }
  }
});
it("preserves every row when repeated prompts exceed the training quota", async () => {
  const data = Array.from({ length: 100 }, (_, i) => row(i));
  for (const item of data.slice(0, 90)) item.messages[0]!.content = "Repeated prompt";
  await save(data);
  const prepared = await prepareDatasetSplit(root, { file: "data.jsonl", output: "duplicates" });
  expect(prepared.preview).toMatchObject({ training_only_rows: 90, counts: { training: 90, validation: 5, test: 5 } });
  await executeDatasetSplit(root, prepared);
  const all = (await Promise.all(["training", "validation", "test"].map(part => rows("duplicates", part)))).flat();
  expect(all.sort((a, b) => a.id - b.id)).toEqual(data);
  await save(data.slice(0, 90));
  await expect(split("no-holdout")).rejects.toThrow(/repeated prompts must stay in training/);
});
it("removes an empty reservation after publication fails", async () => {
  vi.mocked(rename).mockRejectedValueOnce(new Error("Rename failed"));
  await expect(split("failed")).rejects.toThrow("Rename failed");
  expect(await readdir(root)).toEqual(["data.jsonl"]);
});
it("preserves another writer's file when publishing into the reserved destination fails", async () => {
  vi.mocked(rename).mockImplementationOnce(async (_source, destination) => {
    await writeFile(join(String(destination), "other-writer.txt"), "keep me");
    throw Object.assign(new Error("Destination is no longer empty"), { code: "ENOTEMPTY" });
  });
  await expect(split("contended")).rejects.toThrow(/no longer empty/);
  expect(await readFile(join(root, "contended", "other-writer.txt"), "utf8")).toBe("keep me");
  expect((await readdir(root)).filter(name => name.startsWith(".tt-split-"))).toEqual([]);
});
it("refuses changed sources and cancellation without publishing partial outputs", async () => {
  const p = await prepareDatasetSplit(root, { file: "data.jsonl", output: "stale" });
  await save(Array.from({ length: 101 }, (_,i) => row(i)));
  await expect(executeDatasetSplit(root, p)).rejects.toThrow(/changed after review/);
  const fresh = await prepareDatasetSplit(root, { file: "data.jsonl", output: "cancel" });
  await expect(executeDatasetSplit(root, fresh, AbortSignal.abort())).rejects.toThrow();
  expect(await readdir(root)).toEqual(["data.jsonl"]);
});
it("rejects invalid ratios, missing strata, malformed rows, traversal, symlinks and existing destinations", async () => {
  await expect(split("bad", { train: .8, validation: .3 })).rejects.toThrow(/sum to 1/);
  await expect(split("bad", { stratify_by: "missing" })).rejects.toThrow(/scalar/);
  await expect(split("../escape")).rejects.toThrow(/inside/);
  await expect(split("data.jsonl")).rejects.toThrow(/already exists/);
  await symlink(join(root,"data.jsonl"),join(root,"link.jsonl"));
  await expect(inspectDatasetFile(root,"link.jsonl")).rejects.toThrow(/symlinks/);
  await writeFile(join(root,"data.jsonl"), JSON.stringify(row(0))+"\n{\n");
  await expect(split("bad")).rejects.toThrow(/Row 2: malformed/);
  await save([row(0)]);
  await expect(split("small")).rejects.toThrow(/split is empty/);
});

it("cleans private staging if cancelled after execution has opened its outputs", async () => {
  const p = await prepareDatasetSplit(root, { file: "data.jsonl", output: "cancel" });
  const controller = new AbortController();
  const original = controller.signal.throwIfAborted.bind(controller.signal);
  const check = vi.spyOn(controller.signal, "throwIfAborted").mockImplementation(() => {
    if (readdirSync(root).some(name => name.startsWith(".tt-split-"))) controller.abort();
    original();
  });
  try {
    await expect(executeDatasetSplit(root, p, controller.signal)).rejects.toThrow();
    expect(controller.signal.aborted).toBe(true);
    expect(await readdir(root)).toEqual(["data.jsonl"]);
  } finally { check.mockRestore(); }
});
