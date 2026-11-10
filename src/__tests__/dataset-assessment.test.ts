import { mkdtemp, writeFile, rm, symlink, readdir } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { afterEach, expect, it } from "vitest";
import { assessLocalDataset } from "../local-runtime/dataset-split.js";
const roots: string[] = [];
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
const rows = (pairs: string[][]) => pairs.map(([input, output]) => JSON.stringify({ messages: [{ role: "user", content: input }, { role: "assistant", content: output }] })).join("\n");
it("reports conflicting labels and normalized overlap without changing the source", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-assess-")); roots.push(root);
  await writeFile(join(root, "train.jsonl"), rows([["Hello", "yes"], ["  HELLO ", "no"], ["World", "yes"]]));
  await writeFile(join(root, "test.jsonl"), rows([["hello", "yes"], ["new", "no"], ["NEW", "no"]]));
  const before = await readdir(root);
  expect(await assessLocalDataset(root, { training: "train.jsonl", evaluation: "test.jsonl" })).toMatchObject({
    training: { rows: 3, duplicate_prompts: 1, conflicting_answers: 1 },
    evaluation: { rows: 3, duplicate_prompts: 1, training_overlap: 1 }, structural_checks_passed: false,
  });
  expect(await readdir(root)).toEqual(before);
});
it("distinguishes clean structure from missing evaluation and refuses traversal", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-assess-")); roots.push(root);
  await writeFile(join(root, "train.jsonl"), rows([["a", "yes"], ["b", "no"]]));
  const result = await assessLocalDataset(root, { training: "train.jsonl" });
  expect(result.structural_checks_passed).toBe(true);
  expect(result.warnings.join(" ")).toContain("No held-out");
  await expect(assessLocalDataset(root, { training: "../train.jsonl" })).rejects.toThrow(/inside/);
  if (process.platform !== "win32") {
    await symlink(join(root, "train.jsonl"), join(root, "link.jsonl"));
    await expect(assessLocalDataset(root, { training: "link.jsonl" })).rejects.toThrow(/symlink/);
  }
});
