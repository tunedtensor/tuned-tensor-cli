import { defaultLocalHome } from "./store.js";
import { lstat, readFile, rename, writeFile, unlink } from "node:fs/promises";
import { dirname, join } from "node:path";
import { parseLocalRunInput } from "./local-project.js";
import { pipelineForRunInput } from "../pipeline.js";
import { parseLocalRunnerConfig } from "./orchestrator.js";

/** Validate everything before writing; retain recoverable copies of every input. */
export async function migrateProjectWorkflow(specPath: string) {
  const info = await lstat(specPath);
  if (!info.isFile() || info.isSymbolicLink()) throw new Error("Migration requires a regular spec file.");
  const original = await readFile(specPath, "utf8");
  const document = JSON.parse(original);
  const input = parseLocalRunInput(document, specPath);
  if (input.kind === "request") throw new Error("Migration requires a behavior spec.");
  const files: string[] = [];
  const sources = new Map<string, string>([[specPath, original]]);
  for (const name of ["tunedtensor.pipeline.json", "local-runner.json"]) {
    const path = join(dirname(specPath), name);
    const entry = await lstat(path).catch((error) => { if (error.code === "ENOENT") return undefined; throw error; });
    if (!entry) continue;
    if (!entry.isFile() || entry.isSymbolicLink()) throw new Error(`Migration requires regular files: ${path}`);
    const source = await readFile(path, "utf8");
    sources.set(path, source);
    const raw = JSON.parse(source);
    if (name === "tunedtensor.pipeline.json") {
      pipelineForRunInput(input, raw);
      document.pipeline = raw;
    } else {
      if (document.runtime || document.evaluation) throw new Error("Conflicting runtime/evaluation settings; reconcile them before migration.");
      const resolved = parseLocalRunnerConfig(raw, path);
      if (resolved.dryRun) throw new Error("Legacy dryRun is enabled. Use --dry-run explicitly and remove dryRun before migration.");
      if (input.kind === "foundation-spec") {
        if (Object.keys(raw).some(key => !["gpu", "dryRun"].includes(key))) {
          throw new Error("Foundation migration supports runner gpu only; reconcile adapter-only runner settings before migration.");
        }
        document.runtime = resolved.gpu ? { gpu: resolved.gpu } : {};
      } else {
        const { evaluation, dryRun: _dryRun, ...runtime } = resolved;
        document.runtime = { ...runtime, storeRoot: runtime.storeRoot ?? defaultLocalHome() };
        document.evaluation = evaluation;
      }
    }
    files.push(path);
  }
  if (!files.length) return { migrated: false, path: specPath, backups: [] };
  pipelineForRunInput(parseLocalRunInput(document, specPath));
  const backups: string[] = [];
  const temp = `${specPath}.migrating`;
  let createdTemp = false;
  try {
    for (const path of [specPath, ...files]) {
      await writeFile(`${path}.bak`, sources.get(path)!, { flag: "wx", mode: 0o600 });
      backups.push(`${path}.bak`);
    }
    await writeFile(temp, `${JSON.stringify(document, null, 2)}\n`, { flag: "wx", mode: info.mode & 0o777 });
    createdTemp = true;
    for (const [path, source] of sources) {
      const current = await lstat(path);
      if (!current.isFile() || current.isSymbolicLink() || await readFile(path, "utf8") !== source) {
        throw new Error(`Configuration changed during migration: ${path}; retry after reviewing it.`);
      }
    }
    await rename(temp, specPath);
  } catch (error) {
    if (createdTemp) await unlink(temp).catch(() => {});
    for (const backup of backups) await unlink(backup).catch(() => {});
    throw error;
  }
  for (const path of files) {
    if (await readFile(path, "utf8") !== sources.get(path)) throw new Error(`Legacy config changed during migration: ${path}; retained with backups for inspection.`);
    await unlink(path);
  }
  return { migrated: true, path: specPath, backups };
}
