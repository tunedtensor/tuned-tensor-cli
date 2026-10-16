import { afterEach, expect, it, vi } from "vitest";
import { chmod, mkdir, mkdtemp, readdir, rm, stat, symlink, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { hardenExistingLocalStore } from "../local-runtime/store.js";

vi.mock("node:fs/promises", async (importOriginal) => {
  const fs = await importOriginal<typeof import("node:fs/promises")>();
  return { ...fs, readdir: vi.fn(fs.readdir) };
});

let root: string;
afterEach(async () => {
  if (root) await rm(root, { recursive: true, force: true });
});

it.each(["file", "directory"])("tolerates a transient %s disappearing after listing the store", async (kind) => {
  root = await mkdtemp(join(tmpdir(), "tt-store-hardening-"));
  const transient = join(root, "transient");
  const persistent = join(root, "state.json");
  if (kind === "directory") await mkdir(transient);
  else await writeFile(transient, "temporary atomic write");
  await writeFile(persistent, "{}");
  await chmod(persistent, 0o644);
  const fs = await vi.importActual<typeof import("node:fs/promises")>("node:fs/promises");
  // A concurrent writer renames a temporary state file or releases its lock
  // between the permission scan's directory listing and its next lstat.
  vi.mocked(readdir).mockImplementationOnce(async (...args) => {
    const entries = await fs.readdir(...args);
    await rm(transient, { recursive: true });
    return entries;
  });
  expect(await hardenExistingLocalStore(root)).toBe(true);
  expect((await stat(persistent)).mode & 0o777).toBe(0o600);
});

it("continues to reject unsafe entries during permission repair", async () => {
  root = await mkdtemp(join(tmpdir(), "tt-store-hardening-"));
  await symlink(join(root, "missing"), join(root, "link"));
  await expect(hardenExistingLocalStore(root)).rejects.toThrow(/symbolic link/);
});
