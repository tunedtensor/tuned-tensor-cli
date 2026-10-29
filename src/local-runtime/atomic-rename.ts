import { rename } from "node:fs/promises";

/** Windows readers and permission scans can briefly deny replacing a file. */
export async function renameAtomicFile(source: string, destination: string): Promise<void> {
  for (let attempt = 0; ; attempt++) {
    try { await rename(source, destination); return; }
    catch (error) {
      if (process.platform !== "win32" || attempt >= 10
        || !["EPERM", "EACCES", "EBUSY"].includes((error as NodeJS.ErrnoException).code ?? "")) throw error;
      await new Promise(resolve => setTimeout(resolve, 10 * (attempt + 1)));
    }
  }
}
