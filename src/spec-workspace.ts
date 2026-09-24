import { randomUUID } from "node:crypto";
import { constants } from "node:fs";
import { lstat, mkdir, open, readdir, rename, unlink, type FileHandle } from "node:fs/promises";
import { isAbsolute, join, relative, sep } from "node:path";
import { canonicalWorkspace, fingerprintWorkspace, LocalSpecMutationError } from "./local-spec-workspace.js";
import { parseLocalRunInput, specHash, validateBehaviorSpec, type SpecValidation } from "./local-runtime/local-project.js";

import { pipelineForRunInput } from "./pipeline.js";

const MAX_BYTES = 200_000;
const EDIT_KEYS = new Set([
  "name", "description", "system_prompt", "guidelines", "constraints", "examples",
  "base_model", "hyperparameters", "foundation", "dataset_prebuilt", "runtime", "evaluation", "pipeline",
]);
function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** Parse once for review; retain the original document so edits preserve its fields. */
function validateDocument(document: unknown, path: string): SpecValidation {
  try {
    const input = parseLocalRunInput(document, path);
    const result = validateBehaviorSpec(input);
    if (result.valid) pipelineForRunInput(input);
    return result;
  } catch (error) {
    return { valid: false, errors: [(error as Error).message], warnings: [] };
  }
}

async function readBounded(path: string, limit = MAX_BYTES): Promise<string> {
  const handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
  try {
    const info = await handle.stat();
    if (!info.isFile() || info.size > limit) throw new Error(`Spec/revision must be a regular file of at most ${limit} bytes.`);
    const buffer = Buffer.alloc(limit + 1);
    let size = 0;
    while (size < buffer.length) {
      const result = await handle.read(buffer, size, buffer.length - size, null);
      if (!result.bytesRead) break;
      size += result.bytesRead;
    }
    if (size > limit) throw new Error(`Spec/revision exceeds ${limit} bytes.`);
    return buffer.subarray(0, size).toString("utf8");
  } finally {
    await handle.close();
  }
}

/** Restrict agent reads and writes to a named spec under real workspace directories. */
async function specLocation(workspaceRoot: string, specPath: string) {
  const root = await canonicalWorkspace(workspaceRoot);
  const parts = specPath.replace(/^\.\//, "").split(/[\\/]/);
  if (isAbsolute(specPath) || parts.some(part => !part || part === "." || part === "..") || parts.at(-1) !== "tunedtensor.json") {
    throw new Error("spec_path must name tunedtensor.json inside the current workspace (no traversal or symlinks).");
  }
  let parent = root;
  for (const part of parts.slice(0, -1)) {
    parent = join(parent, part);
    const info = await lstat(parent);
    if (info.isSymbolicLink() || !info.isDirectory()) throw new Error("Spec directories must be real directories, not symlinks.");
  }
  return {
    root,
    path: join(parent, "tunedtensor.json"),
    parent,
    displayPath: `./${parts.join("/")}`,
    workspaceFingerprint: await fingerprintWorkspace(root),
    parentFingerprint: await fingerprintWorkspace(parent),
  };
}

async function assertCurrentLocation(location: Awaited<ReturnType<typeof specLocation>>): Promise<void> {
  const root = await canonicalWorkspace(location.root);
  if (await fingerprintWorkspace(root) !== location.workspaceFingerprint
    || await fingerprintWorkspace(location.parent) !== location.parentFingerprint) {
    throw new Error("The spec workspace changed during editing; inspect the revision journal before retrying.");
  }
}

async function readSpecSource(location: Awaited<ReturnType<typeof specLocation>>): Promise<string> {
  if (process.platform !== "linux") return await readBounded(location.path);
  const handles: FileHandle[] = [];
  try {
    let directory = await openDirectory(location.root);
    handles.push(directory);
    if (await fingerprintWorkspace(`/proc/self/fd/${directory.fd}/.`) !== location.workspaceFingerprint) throw new Error("Workspace changed during spec inspection.");
    for (const part of relative(location.root, location.parent).split(sep).filter(Boolean)) {
      directory = await openDirectory(`/proc/self/fd/${directory.fd}/${part}`);
      handles.push(directory);
    }
    if (await fingerprintWorkspace(`/proc/self/fd/${directory.fd}/.`) !== location.parentFingerprint) throw new Error("Spec directory changed during inspection.");
    return await readBounded(`/proc/self/fd/${directory.fd}/tunedtensor.json`);
  } finally {
    for (const handle of handles.reverse()) await handle.close();
  }
}

export async function inspectLocalSpec(workspaceRoot: string, specPath = "tunedtensor.json") {
  const location = await specLocation(workspaceRoot, specPath);
  const source = await readSpecSource(location);
  let document: unknown;
  let validation: SpecValidation;
  try {
    document = JSON.parse(source);
    validation = validateDocument(document, location.path);
  } catch (error) {
    validation = { valid: false, errors: [`Invalid JSON: ${(error as Error).message}`], warnings: [] };
  }
  return { ...location, source, sha256: specHash(source), document, validation };
}

export function specDiff(before: unknown, after: unknown): string {
  const left = isObject(before) ? before : {};
  const right = isObject(after) ? after : {};
  const keys = [...new Set([...Object.keys(left), ...Object.keys(right)])].sort();
  const lines: string[] = [];
  for (const key of keys) {
    if (JSON.stringify(left[key]) === JSON.stringify(right[key])) continue;
    lines.push(
      `@@ ${key} @@`,
      ...JSON.stringify(left[key] ?? null, null, 2).split("\n").map(line => `- ${line}`),
      ...JSON.stringify(right[key] ?? null, null, 2).split("\n").map(line => `+ ${line}`),
    );
  }
  return lines.join("\n") || "No changes.";
}

export interface SpecUpdate {
  spec_path: string;
  expected_sha256: string;
  after_sha256: string;
  workspace_fingerprint: string;
  parent_fingerprint: string;
  changes: Record<string, unknown>;
}

export async function prepareSpecUpdate(workspaceRoot: string, specPath: string, expectedSha256: string, changes: unknown) {
  const current = await inspectLocalSpec(workspaceRoot, specPath);
  if (current.sha256 !== expectedSha256) throw new Error("The behavior spec changed; inspect it and prepare the edit again.");
  if (!isObject(current.document)) throw new Error("Repair the spec JSON before preparing an edit.");
  if (!isObject(changes) || !Object.keys(changes).length || Object.keys(changes).some(key => !EDIT_KEYS.has(key))) {
    throw new Error("Provide supported spec field changes; identity and engine cannot be changed by an edit.");
  }
  const next = { ...current.document, ...changes };
  function mergeSettings(before: Record<string, unknown>, patch: Record<string, unknown>): Record<string, unknown> {
    const result = { ...before };
    for (const [key, value] of Object.entries(patch)) {
      if (value === null) delete result[key];
      else result[key] = isObject(value) && isObject(result[key]) ? mergeSettings(result[key], value) : value;
    }
    return result;
  }
  for (const key of ["runtime", "evaluation", "pipeline"]) {
    if (changes[key] === null) delete next[key];
    else if (key !== "pipeline" && isObject(changes[key])) next[key] = mergeSettings(isObject(current.document[key]) ? current.document[key] : {}, changes[key]);
  }
  for (const key of ["hyperparameters", "foundation"] as const) {
    if (isObject(changes[key])) {
      const previous = isObject(current.document[key]) ? current.document[key] : {};
      next[key] = { ...previous, ...changes[key] };
    }
  }
  const validation = validateDocument(next, current.path);
  if (!validation.valid) throw new Error(`Invalid behavior spec:\n${validation.errors.join("\n")}`);
  const source = `${JSON.stringify(next, null, 2)}\n`;
  if (Buffer.byteLength(source) > MAX_BYTES) throw new Error("Updated spec is too large.");
  const diff = specDiff(current.document, next);
  if (diff === "No changes.") throw new Error("The proposed edit makes no changes.");
  const update: SpecUpdate = {
    spec_path: current.displayPath,
    expected_sha256: current.sha256,
    after_sha256: specHash(source),
    workspace_fingerprint: current.workspaceFingerprint,
    parent_fingerprint: current.parentFingerprint,
    changes,
  };
  return { current, next, source, diff, validation, update };
}

export class SpecWorkspaceMismatchError extends Error {}

export async function validateSpecUpdate(workspaceRoot: string, update: SpecUpdate) {
  const root = await canonicalWorkspace(workspaceRoot);
  if (await fingerprintWorkspace(root) !== update.workspace_fingerprint) {
    throw new SpecWorkspaceMismatchError("The local workspace changed; return to the original workspace before approving.");
  }
  const prepared = await prepareSpecUpdate(workspaceRoot, update.spec_path, update.expected_sha256, update.changes);
  if (prepared.current.workspaceFingerprint !== update.workspace_fingerprint || prepared.current.parentFingerprint !== update.parent_fingerprint) {
    throw new Error("The spec workspace changed; return to the original workspace and prepare the edit again.");
  }
  if (prepared.update.after_sha256 !== update.after_sha256) {
    throw new Error("The proposed spec edit changed after review; prepare it again.");
  }
  return prepared;
}

async function openDirectory(path: string) {
  return await open(path, constants.O_RDONLY | constants.O_DIRECTORY | constants.O_NOFOLLOW);
}
async function privateChild(parent: FileHandle, name: string) {
  const path = `/proc/self/fd/${parent.fd}/${name}`;
  await mkdir(path, { mode: 0o700 }).catch(error => {
    if (error.code !== "EEXIST") throw error;
  });
  return await openDirectory(path);
}
async function atomicText(parent: FileHandle, name: string, source: string) {
  const base = `/proc/self/fd/${parent.fd}`;
  const temporary = join(base, `.tt-${randomUUID()}.tmp`);
  try {
    const file = await open(temporary, "wx", 0o600);
    try {
      await file.writeFile(source, "utf8");
      await file.sync();
    } finally {
      await file.close();
    }
    await rename(temporary, join(base, name));
    await parent.sync();
  } finally {
    await unlink(temporary).catch(() => {});
  }
}

/** Cooperative writer lock, reviewed-byte check, atomic replacement and durable revision journal. */
export async function applySpecUpdate(workspaceRoot: string, update: SpecUpdate) {
  let committing = false;
  let lock: FileHandle | undefined;
  const handles: FileHandle[] = [];
  let lockPath: string | undefined;
  try {
    if (process.platform !== "linux") throw new Error("Secure local spec editing currently requires Linux filesystem handle support.");
    const prepared = await validateSpecUpdate(workspaceRoot, update);
    // Open every component relative to an already-open directory to prevent path redirection.
    let parent = await openDirectory(prepared.current.root);
    handles.push(parent);
    for (const part of relative(prepared.current.root, prepared.current.parent).split(sep).filter(Boolean)) {
      parent = await openDirectory(`/proc/self/fd/${parent.fd}/${part}`);
      handles.push(parent);
    }
    const stable = `/proc/self/fd/${parent.fd}`;
    if (await fingerprintWorkspace(`${stable}/.`) !== update.parent_fingerprint) throw new Error("Spec directory changed before editing.");
    lockPath = `${stable}/.tunedtensor-spec.lock`;
    lock = await open(lockPath, "wx", 0o600);
    const before = await readBounded(`${stable}/tunedtensor.json`);
    if (specHash(before) !== update.expected_sha256) throw new Error("The behavior spec changed; inspect it and prepare the edit again.");
    const state = await privateChild(parent, ".tuned-tensor");
    handles.push(state);
    const history = await privateChild(state, "spec-history");
    handles.push(history);
    const id = `${Date.now()}-${randomUUID()}`;
    const revision: SpecRevision = {
      id,
      status: "prepared",
      at: new Date().toISOString(),
      before_sha256: update.expected_sha256,
      after_sha256: update.after_sha256,
      before: prepared.current.document,
      after: prepared.next,
    };
    await atomicText(history, `${id}.json`, JSON.stringify(revision));
    // Recheck after journal IO, immediately before replacing the reviewed file.
    if (specHash(await readBounded(`${stable}/tunedtensor.json`)) !== update.expected_sha256) throw new Error("The behavior spec changed before commit; prepare the edit again.");
    await assertCurrentLocation(prepared.current);
    committing = true;
    await atomicText(parent, "tunedtensor.json", prepared.source);
    await assertCurrentLocation(prepared.current);
    await atomicText(history, `${id}.json`, JSON.stringify({ ...revision, status: "applied" }));
    return { updated: true, path: update.spec_path, revision: id, sha256: revision.after_sha256, diff: prepared.diff };
  } catch (error) {
    throw new LocalSpecMutationError((error as Error).message, committing ? "unknown" : "not_applied", error);
  } finally {
    if (lock) {
      await lock.close();
      await unlink(lockPath!).catch(() => {});
    }
    for (const handle of handles.reverse()) await handle.close();
  }
}

export interface SpecRevision {
  id: string;
  status: "prepared" | "applied";
  at: string;
  before_sha256: string;
  after_sha256: string;
  before: unknown;
  after: unknown;
}
export async function readSpecHistory(workspaceRoot: string, specPath = "tunedtensor.json"): Promise<SpecRevision[]> {
  const location = await specLocation(workspaceRoot, specPath);
  let path = location.parent;
  for (const part of [".tuned-tensor", "spec-history"]) {
    path = join(path, part);
    try {
      const info = await lstat(path);
      if (!info.isDirectory() || info.isSymbolicLink()) {
        throw new Error("Spec history must use real directories.");
      }
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
      throw error;
    }
  }
  const names = (await readdir(path)).filter(name => /^\d+-[a-f0-9-]+\.json$/.test(name)).sort().reverse().slice(0, 50);
  return await Promise.all(names.map(async name => JSON.parse(await readBounded(join(path, name), MAX_BYTES * 3)) as SpecRevision));
}
