import { mkdtemp, mkdir, readFile, rename, rm, symlink, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { applySpecUpdate, inspectLocalSpec, prepareSpecUpdate, readSpecHistory } from "../spec-workspace.js";
import { parseLocalRunInput, validateBehaviorSpec } from "../local-runtime/local-project.js";
import { reviewSpec } from "../commands/spec.js";
import { createShellSession } from "../shell.js";

const spec = {
  name: "Feedback", base_model: "Qwen/Qwen3.5-2B", system_prompt: "Classify feedback.",
  guidelines: ["Return one label."], constraints: [],
  examples: [{ input: "Great", output: "positive" }, { input: "Bad", output: "negative" }],
  hyperparameters: { n_epochs: 2, learning_rate: 0.0002 },
};
let root: string;
beforeEach(async () => { root = await mkdtemp(join(tmpdir(), "tt-spec-review-")); await writeFile(join(root, "tunedtensor.json"), JSON.stringify(spec)); });
afterEach(async () => { await rm(root, { recursive: true, force: true }); });
async function proposal(changes: Record<string, unknown> = { system_prompt: "Classify feedback with no commentary." }) {
  return await prepareSpecUpdate(root, "tunedtensor.json", (await inspectLocalSpec(root)).sha256, changes);
}

describe("behavior spec review and editing", () => {
  it("reads and validates without writes, preserves unrelated settings, and records the approved diff", async () => {
    const original = await readFile(join(root, "tunedtensor.json"), "utf8");
    const prepared = await proposal({ guidelines: ["Return one lowercase label."], hyperparameters: { n_epochs: 3 } });
    expect(prepared.diff).toContain('@@ guidelines @@');
    expect(prepared.diff).toContain('-   "Return one label."');
    expect(await readFile(join(root, "tunedtensor.json"), "utf8")).toBe(original);
    expect(await readSpecHistory(root)).toEqual([]);
    const result = await applySpecUpdate(root, prepared.update);
    const updated = await inspectLocalSpec(root);
    expect(result.sha256).toBe(updated.sha256);
    expect(updated.document).toEqual({ ...spec, guidelines: ["Return one lowercase label."], hyperparameters: { n_epochs: 3, learning_rate: 0.0002 } });
    expect(await readSpecHistory(root)).toEqual([expect.objectContaining({ status: "applied", before: spec, after: updated.document })]);
    expect((await reviewSpec(root, "diff")).text).toContain(prepared.diff);
    await writeFile(join(root, "tunedtensor.json"), JSON.stringify({ ...prepared.next, name: "External edit" }));
    expect((await reviewSpec(root, "diff")).text).toContain("Changes since the last approved revision");
    expect((await reviewSpec(root, "diff")).text).toContain('@@ name @@');
  });

  it.each([
    { system_prompt: "   " }, { examples: [{ input: "a", output: "b" }] },
    { examples: [{ input: "a", output: "b" }, { input: "a", output: "c" }] },
    { hyperparameters: { n_epocs: 2 } }, { engine: "foundation" },
  ])("refuses an invalid edit before mutation: %j", async (changes) => {
    await expect(proposal(changes)).rejects.toThrow();
    expect((await inspectLocalSpec(root)).document).toEqual(spec);
    expect(await readSpecHistory(root)).toEqual([]);
  });

  it("rejects stale or altered proposals without overwriting the spec", async () => {
    const prepared = await proposal();
    await expect(applySpecUpdate(root, { ...prepared.update, changes: { name: "Tampered" } })).rejects.toThrow(/changed after review/);
    await writeFile(join(root, "tunedtensor.json"), JSON.stringify({ ...spec, name: "Someone else's edit" }));
    await expect(applySpecUpdate(root, prepared.update)).rejects.toThrow(/changed/);
    expect((await inspectLocalSpec(root)).document).toMatchObject({ name: "Someone else's edit" });
    expect(await readSpecHistory(root)).toEqual([]);
  });

  it("serializes concurrent approvals so only one reviewed edit wins", async () => {
    const first = await proposal({ name: "First" });
    const second = await proposal({ name: "Second" });
    const results = await Promise.allSettled([applySpecUpdate(root, first.update), applySpecUpdate(root, second.update)]);
    expect(results.filter(result => result.status === "fulfilled")).toHaveLength(1);
    expect((await readSpecHistory(root)).filter(item => item.status === "applied")).toHaveLength(1);
  });

  it("refuses traversal, spec symlinks and symlinked parent/history directories", async () => {
    await expect(inspectLocalSpec(root, "../tunedtensor.json")).rejects.toThrow(/traversal/);
    await mkdir(join(root, "real"));
    await writeFile(join(root, "real/tunedtensor.json"), JSON.stringify(spec));
    await symlink(join(root, "real"), join(root, "alias"));
    await expect(inspectLocalSpec(root, "alias/tunedtensor.json")).rejects.toThrow(/symlink/);
    const prepared = await proposal();
    await mkdir(join(root, "elsewhere"));
    await symlink(join(root, "elsewhere"), join(root, ".tuned-tensor"));
    await expect(applySpecUpdate(root, prepared.update)).rejects.toThrow();
    expect((await inspectLocalSpec(root)).document).toEqual(spec);
    await rm(join(root, "tunedtensor.json"));
    await symlink(join(root, "real/tunedtensor.json"), join(root, "tunedtensor.json"));
    await expect(inspectLocalSpec(root)).rejects.toThrow();
  });


  it("rejects a replaced nested directory even when its spec bytes are identical", async () => {
    await mkdir(join(root, "nested"));
    const source = JSON.stringify(spec);
    await writeFile(join(root, "nested/tunedtensor.json"), source);
    const read = await inspectLocalSpec(root, "nested/tunedtensor.json");
    const prepared = await prepareSpecUpdate(root, "nested/tunedtensor.json", read.sha256, { name: "Changed" });
    await rename(join(root, "nested"), join(root, "old"));
    await mkdir(join(root, "nested"));
    await writeFile(join(root, "nested/tunedtensor.json"), source);
    await expect(applySpecUpdate(root, prepared.update)).rejects.toThrow(/workspace changed/);
    expect(await readFile(join(root, "nested/tunedtensor.json"), "utf8")).toBe(source);
    expect(await readFile(join(root, "old/tunedtensor.json"), "utf8")).toBe(source);
  });

  it("reports malformed JSON for review without silently repairing it", async () => {
    await writeFile(join(root, "tunedtensor.json"), '{"name":');
    const read = await inspectLocalSpec(root);
    expect(read.validation.valid).toBe(false);
    expect((await reviewSpec(root)).text).toContain('Invalid JSON');
    expect(await reviewSpec(root)).toMatchObject({ source: '{"name":' });
    await expect(reviewSpec(root, "diff")).rejects.toThrow(/Cannot compare malformed spec JSON/);
    await expect(prepareSpecUpdate(root, "tunedtensor.json", read.sha256, { name: "Fixed" })).rejects.toThrow(/Repair/);
  });

  it("uses the same validation for foundation settings and placeholder specs", () => {
    expect(validateBehaviorSpec(parseLocalRunInput({ ...spec, system_prompt: " " }, "tunedtensor.json")).valid).toBe(false);
    expect(() => parseLocalRunInput({ ...spec, hyperparameters: { unknown: true } }, "tunedtensor.json")).toThrow();
    expect(validateBehaviorSpec(parseLocalRunInput(spec, "tunedtensor.json"))).toMatchObject({ valid: true, warnings: [expect.stringMatching(/quality/)] });
  });

  it("reviews the current spec, pending diff and history in the shell without a model call", async () => {
    const output: string[] = []; const errors: string[] = [];
    const prepared = await proposal();
    const agent = { busy: false, handleLine: vi.fn(async () => "continue" as const), interrupt: () => false,
      snapshot: () => ({ pendingActions: [{ id: "reviewed-action", operation: "update_local_spec", title: "Edit", summary: "", risk: "medium", arguments: prepared.update, preview: { diff: prepared.diff } }] }) };
    const shell = await createShellSession({ cwd: root, env: { HOME: root, TUNED_TENSOR_HOME: join(root, "home") }, agent,
      runner: vi.fn(), io: { write: text => output.push(text), writeError: text => errors.push(text), clear: () => {} } });
    for (const command of ["/spec", "/spec diff", "/spec validate", "/spec history"]) await shell.handleLine(command);
    expect(output.join("\n")).toContain("Pending edit reviewed");
    expect(output.join("\n")).toContain(prepared.diff);
    expect(output.join("\n")).toContain("No reviewed edits recorded");
    expect(errors).toEqual([]);
    expect(agent.handleLine).not.toHaveBeenCalled();
  });
});
