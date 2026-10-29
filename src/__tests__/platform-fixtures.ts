import { mkdtempSync, rmSync, statSync, symlinkSync as nativeSymlinkSync, writeFileSync } from "node:fs";
import { copyFile, symlink as nativeSymlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";

// Directory junctions exercise Windows reparse-point checks without requiring
// administrator rights or changing the computer's Developer Mode setting.
function linkType(target: Parameters<typeof nativeSymlinkSync>[0], type?: string | null): "dir" | "file" | "junction" | undefined {
  if (type == null) type = undefined;
  if (type !== undefined && type !== "dir" && type !== "file" && type !== "junction") throw new Error("Invalid fixture symlink type.");
  if (process.platform !== "win32") return type;
  try { if (type === "dir" || type === "junction" || statSync(target).isDirectory()) return "junction" as const; }
  catch { /* A dangling file link keeps its requested type. */ }
  return type;
}
export const symlinkSync: typeof nativeSymlinkSync = (target, path, type) => nativeSymlinkSync(target, path, linkType(target, type));
export const symlink: typeof nativeSymlink = async (target, path, type) => await nativeSymlink(target, path, linkType(target, type));

export const canCreateFileSymlinks = (() => {
  if (process.platform !== "win32") return true;
  const root = mkdtempSync(join(tmpdir(), "tt-link-capability-"));
  try {
    writeFileSync(join(root, "target"), "fixture");
    nativeSymlinkSync(join(root, "target"), join(root, "link"), "file");
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "EPERM") return false;
    throw error;
  } finally { rmSync(root, { recursive: true, force: true }); }
})();

/** huggingface_hub uses ordinary snapshot copies on Windows without symlinks. */
export async function linkOrCopySnapshot(target: string, path: string): Promise<void> {
  if (canCreateFileSymlinks) await nativeSymlink(target, path);
  else await copyFile(resolve(dirname(path), target), path);
}
