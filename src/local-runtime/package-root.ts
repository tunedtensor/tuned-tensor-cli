import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

/** Resolve the unified CLI package root from source tests or bundled dist files. */
export function localRuntimePackageRoot(moduleUrl: string): string {
  const moduleDirectory = dirname(fileURLToPath(moduleUrl));
  const candidates = [
    dirname(moduleDirectory),
    dirname(dirname(moduleDirectory)),
  ];
  const root = candidates.find((candidate) =>
    existsSync(resolve(candidate, "package.json"))
    && existsSync(resolve(candidate, "training/adapter")),
  );
  if (!root) {
    throw new Error("Unable to locate the bundled Tuned Tensor local runtime assets.");
  }
  return root;
}
