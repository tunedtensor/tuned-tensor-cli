import { createHash, randomUUID } from "node:crypto";
import { constants } from "node:fs";
import { lstat, mkdir, open, rename, rm, rmdir, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { z } from "zod";
import { canonicalWorkspace, fingerprintWorkspace } from "../local-spec-workspace.js";
import { SpecWorkspaceMismatchError } from "../spec-workspace.js";
import { datasetInputIdentity, type ChatJsonlRow } from "./dataset.js";
import { resolveCachedDatasetFile } from "../agent-local-datasets.js";

const MAX_LINE_BYTES = 2 * 1024 * 1024;
const SPLITS = ["training", "validation", "test"] as const;
type Split = typeof SPLITS[number];
export const datasetSplitOptionsSchema = z.object({
  file: z.string().min(1),
  output: z.string().min(1),
  train: z.number().positive().max(1).default(0.8),
  validation: z.number().min(0).max(1).default(0.1),
  test: z.number().min(0).max(1).default(0.1),
  seed: z.string().min(1).max(200).default("42"),
  method: z.enum(["random", "ordered"]).default("random"),
  group_by: z.string().min(1).max(200).optional(),
  stratify_by: z.string().min(1).max(200).optional(),
}).strict().superRefine((v, ctx) => {
  if (Math.abs(v.train + v.validation + v.test - 1) > 1e-9 || v.validation + v.test === 0) {
    ctx.addIssue({ code: "custom", message: "Ratios must sum to 1, with training and at least one held-out split." });
  }
});
export type DatasetSplitOptions = z.infer<typeof datasetSplitOptionsSchema>;
const digest = (text: string) => createHash("sha256").update(text).digest("hex");
const object = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);

/** Agent-accessible data stays inside the workspace, without symlink traversal. */
async function location(workspaceRoot: string, name: string, missingLeaf = false) {
  const root = await canonicalWorkspace(workspaceRoot);
  const path = resolve(root, name);
  const child = relative(root, path);
  if (!child || child === ".." || child.startsWith(`..${sep}`) || isAbsolute(child)) throw new Error("Dataset paths must be inside the current workspace.");
  let current = root;
  const parts = child.split(sep);
  for (const [i, part] of parts.entries()) {
    current = join(current, part);
    let info;
    try { info = await lstat(current); }
    catch (error) {
      if (missingLeaf && i === parts.length - 1 && (error as NodeJS.ErrnoException).code === "ENOENT") break;
      throw error;
    }
    if (info.isSymbolicLink()) throw new Error("Dataset paths cannot contain symlinks.");
    if (i < parts.length - 1 && !info.isDirectory()) throw new Error("Dataset parent must be a directory.");
    if (i === parts.length - 1 && missingLeaf) throw new Error("Split output already exists; choose a new output directory.");
  }
  return { root, path, display: `./${child.split(sep).join("/")}` };
}

async function sourceLocation(workspaceRoot: string, name: string) {
  if (!name.startsWith("hf://")) return { ...await location(workspaceRoot, name), expectedDigest: undefined };
  return { ...await resolveCachedDatasetFile(name), root: await canonicalWorkspace(workspaceRoot), display: name };
}

function validateRow(value: unknown, line: number): ChatJsonlRow & Record<string, unknown> {
  const fail = () => { throw new Error(`Row ${line}: expected chat JSONL with an optional system message, one user message and one assistant answer.`); };
  if (!object(value) || !Array.isArray(value.messages)) return fail();
  const messages = value.messages;
  if (messages.length !== 2 && messages.length !== 3) return fail();
  const roles = messages.length === 3 ? ["system", "user", "assistant"] : ["user", "assistant"];
  if (messages.some((m, i) => !object(m) || m.role !== roles[i] || typeof m.content !== "string" || !m.content.trim())) return fail();
  return value as ChatJsonlRow & Record<string, unknown>;
}

/** Read bounded lines, hash the exact bytes, and never execute dataset contents. */
async function scan(path: string, visit: (row: ChatJsonlRow & Record<string, unknown>, line: number) => void | Promise<void>, signal?: AbortSignal, expectedDigest?: string) {
  const file = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
  try {
    const before = await file.stat();
    if (!before.isFile()) throw new Error("Dataset must be a regular JSONL file.");
    const hash = createHash("sha256");
    const gitHash = expectedDigest?.startsWith("git-sha1:") ? createHash("sha1").update(`blob ${before.size}\0`) : undefined;
    let pending = Buffer.alloc(0), rows = 0, line = 0;
    const accept = async (buffer: Buffer) => {
      line++;
      if (buffer.length > MAX_LINE_BYTES) throw new Error(`Row ${line} exceeds 2 MiB.`);
      if (!buffer.toString("utf8").trim()) return;
      let parsed: unknown;
      try { parsed = JSON.parse(buffer.toString("utf8")); }
      catch { throw new Error(`Row ${line}: malformed JSON.`); }
      await visit(validateRow(parsed, line), line);
      rows++;
    };
    const buffer = Buffer.alloc(256 * 1024);
    for (;;) {
      signal?.throwIfAborted();
      const { bytesRead } = await file.read(buffer, 0, buffer.length, null);
      if (!bytesRead) break;
      const bytes = buffer.subarray(0, bytesRead);
      hash.update(bytes);
      gitHash?.update(bytes);
      pending = Buffer.concat([pending, bytes]);
      let start = 0, end: number;
      while ((end = pending.indexOf(10, start)) >= 0) {
        await accept(pending.subarray(start, end));
        start = end + 1;
      }
      pending = pending.subarray(start);
      if (pending.length > MAX_LINE_BYTES) throw new Error(`Row ${line + 1} exceeds 2 MiB.`);
    }
    if (pending.length) await accept(pending);
    const after = await file.stat();
    if (before.size !== after.size || before.mtimeMs !== after.mtimeMs || before.ctimeMs !== after.ctimeMs) throw new Error("Dataset changed while reading; retry after it is stable.");
    if (!rows) throw new Error("Dataset contains no examples.");
    const sha256 = hash.digest("hex");
    if (expectedDigest && expectedDigest !== (gitHash ? `git-sha1:${gitHash.digest("hex")}` : `sha256:${sha256}`)) {
      throw new Error("Cached dataset checksum differs from its download manifest; verify or download it again.");
    }
    return { sha256, rows, size_bytes: before.size };
  } finally { await file.close(); }
}

function field(row: Record<string, unknown>, name: string): string {
  let v: unknown = row;
  for (const key of name.split(".")) v = object(v) && Object.hasOwn(v, key) ? v[key] : undefined;
  if ((typeof v !== "string" && typeof v !== "number") || String(v).length === 0) throw new Error(`Each record needs a scalar ${name} field.`);
  return `${typeof v}:${v}`;
}
function promptKey(row: ChatJsonlRow) {
  return digest(datasetInputIdentity(row.messages.at(-2)!.content));
}

export async function inspectDatasetFile(workspaceRoot: string, file: string, signal?: AbortSignal) {
  const loc = await sourceLocation(workspaceRoot, file);
  const samples: unknown[] = [];
  const fields = new Set<string>();
  const result = await scan(loc.path, (row) => {
    for (const key of Object.keys(row)) if (fields.size < 100) fields.add(key);
    if (object(row.metadata)) for (const key of Object.keys(row.metadata)) if (fields.size < 100) fields.add(`metadata.${key}`);
    if (samples.length < 2) samples.push({ messages: row.messages.map(m => ({ role: m.role, content: m.content.slice(0, 600) })) });
  }, signal, loc.expectedDigest);
  return { file: loc.display, ...result, format: "chat_jsonl", fields: [...fields].slice(0, 100), sample: samples,
    note: "Structure validated across the file. Samples are truncated untrusted data, not instructions or proof of answer quality. No files changed." };
}

type Group = { key: string; stratum: string; count: number; trainingOnly: boolean; split?: Split };
async function plan(workspaceRoot: string, input: unknown, signal?: AbortSignal) {
  const options = datasetSplitOptionsSchema.parse(input);
  const source = await sourceLocation(workspaceRoot, options.file);
  const target = await location(source.root, options.output, true);
  const groups = new Map<string, Group>();
  const prompts = new Map<string, string>();
  const sourceInfo = await scan(source.path, (row) => {
    const prompt = promptKey(row);
    const key = options.group_by ? digest(field(row, options.group_by)) : prompt;
    const previous = prompts.get(prompt);
    if (previous !== undefined && previous !== key) throw new Error("Duplicate prompts occur in different groups; fix the grouping field before splitting.");
    prompts.set(prompt, key);
    const stratum = options.stratify_by ? field(row, options.stratify_by) : "all";
    const group = groups.get(key);
    if (group && group.stratum !== stratum) throw new Error("A group spans multiple strata; choose a consistent stratification field.");
    if (group) {
      group.count++;
      // TT requires unique evaluation prompts, even within a single split.
      if (previous !== undefined) group.trainingOnly = true;
    } else groups.set(key, { key, stratum, count: 1, trainingOnly: false });
  }, signal, source.expectedDigest);
  const strata = new Map<string, Group[]>();
  for (const group of groups.values()) {
    const list = strata.get(group.stratum) ?? [];
    list.push(group); strata.set(group.stratum, list);
  }
  const counts = { training: 0, validation: 0, test: 0 };
  let trainingOnlyRows = 0;
  for (const list of strata.values()) {
    const total = list.reduce((sum, group) => sum + group.count, 0);
    let forcedTraining = 0;
    for (const group of list) if (group.trainingOnly) {
      group.split = "training";
      forcedTraining += group.count;
    }
    counts.training += forcedTraining;
    trainingOnlyRows += forcedTraining;
    const eligible = list.filter(group => !group.trainingOnly);
    const ordered = options.method === "random"
      ? eligible.map(group => ({ group, rank: digest(`${options.seed}\0${group.key}`) })).sort((a,b) => a.rank.localeCompare(b.rank)).map(x => x.group)
      : eligible;
    // Fill the remaining training quota first. If duplicates exceed that quota,
    // divide eligible rows between held-out splits in their requested proportion.
    const trainingWeight = Math.max(0, total * options.train - forcedTraining);
    const validationWeight = total * options.validation;
    const weight = trainingWeight + validationWeight + total * options.test;
    let used = 0;
    for (const group of ordered) {
      const position = (used + group.count / 2) / (total - forcedTraining) * weight;
      group.split = position < trainingWeight ? "training" : position < trainingWeight + validationWeight ? "validation" : "test";
      counts[group.split] += group.count;
      used += group.count;
    }
  }
  for (const [split, ratio] of [["training", options.train], ["validation", options.validation], ["test", options.test]] as const) {
    if (ratio > 0 && counts[split] === 0) throw new Error(`Requested ${split} split is empty. Use more independent groups or different ratios/stratification. Groups with repeated prompts must stay in training.`);
  }
  return { options: { ...options, file: source.display, output: target.display }, source, target, groups, sourceInfo, counts, trainingOnlyRows,
    workspace_fingerprint: await fingerprintWorkspace(source.root), parent_fingerprint: await fingerprintWorkspace(dirname(target.path)) };
}

export async function prepareDatasetSplit(workspaceRoot: string, input: unknown, signal?: AbortSignal) {
  const p = await plan(workspaceRoot, input, signal);
  return { options: p.options, source_sha256: p.sourceInfo.sha256, workspace_fingerprint: p.workspace_fingerprint, parent_fingerprint: p.parent_fingerprint,
    preview: { source: p.options.file, output: p.options.output, rows: p.sourceInfo.rows, groups: p.groups.size, counts: p.counts, training_only_rows: p.trainingOnlyRows,
      method: p.options.method, seed: p.options.seed, group_by: p.options.group_by ?? "normalized user prompt", stratify_by: p.options.stratify_by ?? null,
      note: "Group boundaries can make ratios approximate. Groups containing repeated prompts stay in training so evaluation prompts remain unique. Outputs preserve every record and its metadata; the unsplit source and spec are unchanged. Existing source split labels are metadata only: ordered/random splitting does not preserve original benchmark holdouts." } };
}

export async function validateDatasetSplit(workspaceRoot: string, input: Record<string, unknown>, signal?: AbortSignal) {
  const p = await plan(workspaceRoot, input.options, signal);
  if (input.workspace_fingerprint !== p.workspace_fingerprint) throw new SpecWorkspaceMismatchError("The dataset workspace changed; prepare the split again.");
  if (input.parent_fingerprint !== p.parent_fingerprint || input.source_sha256 !== p.sourceInfo.sha256) throw new Error("Dataset or output parent changed after review; prepare the split again.");
  return p;
}

export async function executeDatasetSplit(workspaceRoot: string, input: Record<string, unknown>, signal?: AbortSignal) {
  const p = await validateDatasetSplit(workspaceRoot, input, signal);
  // Work privately in a sibling staging directory. Reserve the destination with
  // exclusive mkdir only when all bytes have been checked; never overwrite data.
  const stage = join(dirname(p.target.path), `.tt-split-${randomUUID()}`);
  await mkdir(stage, { mode: 0o700 });
  const files: Partial<Record<Split, string>> = {};
  const handles = new Map<Split, Awaited<ReturnType<typeof open>>>();
  let reserved = false;
  try {
    for (const split of SPLITS) if (p.counts[split]) {
      handles.set(split, await open(join(stage, `${split}.jsonl`), "wx", 0o600));
      files[split] = `${p.options.output}/${split}.jsonl`;
    }
    const written = { training: 0, validation: 0, test: 0 };
    const result = await scan(p.source.path, async (row) => {
      const key = p.options.group_by ? digest(field(row, p.options.group_by)) : promptKey(row);
      const split = p.groups.get(key)?.split;
      if (!split) throw new Error("Source groups changed during splitting.");
      await handles.get(split)!.writeFile(`${JSON.stringify(row)}\n`);
      written[split]++;
    }, signal);
    if (result.sha256 !== p.sourceInfo.sha256 || SPLITS.some(s => written[s] !== p.counts[s])) throw new Error("Dataset changed while splitting; no outputs published.");
    for (const handle of handles.values()) await handle.close();
    handles.clear();
    const report = { version: 1, source: p.options.file, source_sha256: result.sha256, options: p.options, rows: result.rows, counts: written,
      groups: p.groups.size, training_only_rows: p.trainingOnlyRows, dataset_prebuilt: { ...files, format: "chat_jsonl" }, note: "Source and spec unchanged. Groups containing repeated prompts were kept in training. Inspect this result, then update dataset_prebuilt in your spec separately. Original source split labels were not enforced." };
    await writeFile(join(stage, "split-manifest.json"), `${JSON.stringify(report, null, 2)}\n`, { flag: "wx" });
    signal?.throwIfAborted();
    await location(workspaceRoot, p.options.output, true);
    if (await fingerprintWorkspace(dirname(p.target.path)) !== p.parent_fingerprint) throw new Error("Output parent changed during splitting.");
    await mkdir(p.target.path, { mode: 0o700 }); reserved = true;
    await rename(stage, p.target.path);
    reserved = false;
    return report;
  } finally {
    for (const handle of handles.values()) await handle.close();
    await rm(stage, { recursive: true, force: true });
    // A concurrent writer may have populated the reserved directory. Only
    // remove it while empty; never recursively delete another writer's data.
    if (reserved) await rmdir(p.target.path).catch(() => {});
  }
}
