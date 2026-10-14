import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { parquetWriteBuffer } from "hyparquet-writer";
import type { AssetManifest } from "../../src/local-runtime/assets.js";
import { previewDataset } from "../../src/local-runtime/dataset-preview.js";

const revision = "0123456789abcdef0123456789abcdef01234567";

async function download(root: string, files: Record<string, string | Uint8Array>, kind: AssetManifest["kind"] = "dataset"): Promise<AssetManifest> {
  const snapshot = join(root, "snapshot");
  for (const [name, content] of Object.entries(files)) {
    await mkdir(join(snapshot, name, ".."), { recursive: true });
    await writeFile(join(snapshot, name), content);
  }
  return {
    version: 1,
    kind,
    repo: "org/code",
    revision,
    fetched_at: new Date(0).toISOString(),
    snapshot_path: snapshot,
    size_bytes: 0,
    files: Object.fromEntries(Object.keys(files).map((name) => [name, "sha256:x"])),
  };
}

async function withRoot(run: (root: string) => Promise<void>) {
  const root = await mkdtemp(join(tmpdir(), "tt-dataset-preview-"));
  try {
    await run(root);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
}

const jsonl = (rows: unknown[]) => `${rows.map((row) => typeof row === "string" ? row : JSON.stringify(row)).join("\n")}\n`;

test("JSONL preview reports fields, quality problems and a mapping check", async () => {
  await withRoot(async (root) => {
    const asset = await download(root, {
      "data/train.jsonl": jsonl([
        { prompt: "Reverse a string", completion: "s[::-1]" },
        { prompt: "Reverse a string", completion: "s[::-1]" },
        { prompt: "Add numbers", completion: "" },
        "{not json",
        { prompt: "Echo", completion: "echo" },
        "",
        { prompt: "Sort a list", completion: "sorted(xs)", lang: "python" },
      ]),
      "data/test.jsonl": jsonl([{ prompt: "x", completion: "y" }]),
      "README.md": "# docs",
    });
    const preview = await previewDataset({ asset, rows: 10, columns: { input: "prompt", output: "completion" } });
    assert.equal(preview.file, "data/train.jsonl");
    assert.equal(preview.format, "jsonl");
    assert.deepEqual(preview.other_files, ["data/test.jsonl"]);
    assert.equal(preview.records, 5);
    assert.equal(preview.scanned, 5);
    assert.equal(preview.scan_complete, true);
    assert.equal(preview.malformed_lines, 1);
    assert.equal(preview.duplicate_records, 1);
    const completion = preview.fields.find((field) => field.name === "completion")!;
    assert.deepEqual({ present: completion.present, blank: completion.blank, types: completion.types }, { present: 5, blank: 1, types: ["string"] });
    assert.deepEqual(completion.chars, { min: 0, median: 7, p95: 10, max: 10 });
    assert.equal(preview.fields.find((field) => field.name === "lang")?.present, 1);
    assert.deepEqual(preview.mapping, {
      columns: { input: "prompt", output: "completion" },
      missing_columns: [],
      usable: 4,
      skipped_missing_fields: 1,
      duplicate_inputs: 1,
      input_equals_output: 1,
    });
    assert.deepEqual(preview.samples.map((sample) => sample.index), [0, 1, 2, 3, 4]);
  });
});

test("preview samples deterministically and bounds long values", async () => {
  await withRoot(async (root) => {
    const rows = Array.from({ length: 200 }, (_, index) => ({ text: `${index}:${"x".repeat(5000)}` }));
    const asset = await download(root, { "train.jsonl": jsonl(rows) });
    const first = await previewDataset({ asset, rows: 3 });
    const second = await previewDataset({ asset, rows: 3 });
    assert.deepEqual(first.samples, second.samples);
    assert.equal(first.samples.length, 3);
    assert.ok(first.samples.some((sample) => sample.index > 2), "sample spans the file, not only its head");
    const text = first.samples[0]!.record.text as string;
    assert.ok(text.length < 1000);
    assert.match(text, /… \[\+\d+ chars\]$/);
  });
});

test("chat rows with many long turns stay inside the tool-output budget", async () => {
  await withRoot(async (root) => {
    const turn = (role: string) => ({ role, content: "y".repeat(4000) });
    const messages = Array.from({ length: 8 }, (_, index) => turn(index % 2 ? "assistant" : "user"));
    const asset = await download(root, { "train.jsonl": jsonl(Array.from({ length: 40 }, () => ({ messages }))) });
    const preview = await previewDataset({ asset, rows: 20, columns: { messages: "messages" } });
    assert.equal(preview.samples.length, 20);
    assert.ok(JSON.stringify(preview).length < 32_000);
    assert.equal(preview.mapping?.usable, 40);
    assert.equal(preview.duplicate_records, 39);
  });
});

test("scan limit stops statistics but keeps an exact record count", async () => {
  await withRoot(async (root) => {
    const asset = await download(root, { "train.jsonl": jsonl(Array.from({ length: 30 }, (_, index) => ({ n: index }))) });
    const preview = await previewDataset({ asset, scanLimit: 10 });
    assert.equal(preview.scanned, 10);
    assert.equal(preview.records, 30);
    assert.equal(preview.scan_complete, false);
    assert.ok(preview.samples.every((sample) => sample.index < 10));
  });
});

test("Parquet preview reads every column and checks a messages mapping", async () => {
  await withRoot(async (root) => {
    const parquet = new Uint8Array(parquetWriteBuffer({
      columnData: [
        { name: "instruction", data: ["one", "two", "three", ""], type: "STRING" },
        { name: "label", data: [1n, 2n, 3n, 4n], type: "INT64" },
      ],
      rowGroupSize: 2,
    }));
    const asset = await download(root, { "data/train-00000.parquet": parquet });
    const preview = await previewDataset({ asset, columns: { messages: "messages" } });
    assert.equal(preview.format, "parquet");
    assert.equal(preview.records, 4);
    assert.equal(preview.scan_complete, true);
    assert.deepEqual(preview.fields.map((field) => [field.name, field.types, field.blank]), [
      ["instruction", ["string"], 1],
      ["label", ["integer"], 0],
    ]);
    assert.deepEqual(preview.mapping?.missing_columns, ["messages"]);
    assert.equal(preview.mapping?.usable, 0);
    assert.equal(preview.samples[0]?.record.label, "1");
  });
});

test("preview only reads downloaded dataset files", async () => {
  await withRoot(async (root) => {
    const asset = await download(root, { "train.jsonl": jsonl([{ a: 1 }]), "notes.csv": "a\n1\n" });
    await assert.rejects(previewDataset({ asset, file: "../../etc/passwd" }), /not a downloaded \.jsonl or \.parquet file.*train\.jsonl/);
    await assert.rejects(previewDataset({ asset, file: "notes.csv" }), /not a downloaded/);
    await assert.rejects(previewDataset({ asset: { ...asset, files: { "notes.csv": "sha256:x" } } }), /no downloaded \.jsonl or \.parquet files/);
    await assert.rejects(previewDataset({ asset: { ...asset, kind: "model" } }), /model, not a dataset/);
    await rm(join(asset.snapshot_path, "train.jsonl"));
    await assert.rejects(previewDataset({ asset }), /missing from/);
  });
});
