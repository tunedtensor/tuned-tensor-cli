import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { pythonEnvironmentPath } from "../paths.js";
import { localRuntimePackageRoot } from "./package-root.js";

const root = localRuntimePackageRoot(import.meta.url);
const project = join(root, "training/serving");
const hash = createHash("sha256");
for (const name of ["pyproject.toml", "uv.lock"]) hash.update(readFileSync(join(project, name)));
export const SERVING_PYTHON_ENVIRONMENT = pythonEnvironmentPath("uv-serving", hash.digest("hex").slice(0, 20));

export function buildServingPythonCommand() {
  const commandArgs = ["run", "--frozen", "--quiet", "--project", project,
    "python", join(root, "training/adapter/src/serve.py")];
  return { command: "uv", commandArgs, displayCommand: ["uv", ...commandArgs] };
}

export function withServingPythonEnvironment(env: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  return {
    ...env,
    UV_PROJECT_ENVIRONMENT: SERVING_PYTHON_ENVIRONMENT,
    VLLM_NO_USAGE_STATS: "1",
    DO_NOT_TRACK: "1",
    VLLM_ALLOW_RUNTIME_LORA_UPDATING: "0",
  };
}
