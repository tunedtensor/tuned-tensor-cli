import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const script = fileURLToPath(new URL("../../scripts/test-npm-windows.ps1", import.meta.url));
const cleanup = (workspace: string) => new Promise<number>((resolve, reject) => {
  execFile(join(process.env.SystemRoot ?? "C:\\Windows", "System32", "WindowsPowerShell", "v1.0", "powershell.exe"),
    ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", script, "-Phase", "Cleanup", "-Workspace", workspace],
    { timeout: 15_000, windowsHide: true }, (error) => {
      if (error && typeof error.code !== "number") return reject(error);
      resolve(typeof error?.code === "number" ? error.code : 0);
    });
});
const options = { skip: process.platform !== "win32" };
test("npm cleanup refuses a workspace outside temporary roots", options, async () => {
  const root = await mkdtemp(join(process.cwd(), "tt npm outside-"));
  try {
    await writeFile(join(root, "sentinel"), "preserve");
    assert.notEqual(await cleanup(root), 0);
    assert.equal(await readFile(join(root, "sentinel"), "utf8"), "preserve");
  } finally { await rm(root, { recursive: true, force: true }); }
});
test("npm cleanup refuses an unrelated directory inside a temporary root", options, async () => {
  const root = await mkdtemp(join(tmpdir(), "tt npm unowned-"));
  try {
    await writeFile(join(root, "sentinel"), "preserve");
    assert.notEqual(await cleanup(root), 0);
    assert.equal(await readFile(join(root, "sentinel"), "utf8"), "preserve");
  } finally { await rm(root, { recursive: true, force: true }); }
});
test("npm cleanup refuses a junction even when its target has an ownership marker", options, async () => {
  const root = await mkdtemp(join(tmpdir(), "tt npm links-"));
  const outside = await mkdtemp(join(tmpdir(), "tt npm target-"));
  const link = join(root, "tt npm redirect");
  try {
    await writeFile(join(outside, "sentinel"), "preserve");
    await writeFile(join(outside, ".tt-npm-smoke.json"), JSON.stringify({ kind: "tt-npm-smoke-v1", root: link, installs: 1 }));
    await symlink(outside, link, "junction");
    assert.notEqual(await cleanup(link), 0);
    assert.equal(await readFile(join(outside, "sentinel"), "utf8"), "preserve");
  } finally { await rm(root, { recursive: true, force: true }); await rm(outside, { recursive: true, force: true }); }
});
