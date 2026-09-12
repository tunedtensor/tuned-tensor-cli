import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { runDoctor } from "../local-runtime/doctor.js";
import { localRunnerConfigSchema, localFoundationSpecFileSchema } from "../local-runtime/contracts.js";
import { checkAwsGpu, runGpuProcess } from "../local-runtime/gpu-executor.js";
vi.mock("../local-runtime/gpu-executor.js", () => ({ checkAwsGpu: vi.fn(), runGpuProcess: vi.fn() }));
let root: string;
beforeEach(async () => {
  vi.clearAllMocks();
  root = await mkdtemp(join(tmpdir(), "tt-gpu-doctor-"));
  await mkdir(join(root, "bin"));
  await writeFile(join(root, "bin/uv"), `#!/bin/sh
if [ "$1" = "--version" ]; then echo uv; exit 0; fi
echo 'Missing local Python dependency' >&2
exit 1
`, { mode: 0o700 });
  vi.stubEnv("PATH", `${join(root, "bin")}:${process.env.PATH}`);
  vi.mocked(checkAwsGpu).mockResolvedValue();
  vi.mocked(runGpuProcess).mockResolvedValue({ exitCode: 0, stderr: "" });
});
afterEach(async () => { vi.unstubAllEnvs(); await rm(root, { recursive: true, force: true }); });

it.each(["adapter", "foundation"])("checks local dependencies for %s even when the AWS GPU is ready", async (engine) => {
  const config = localRunnerConfigSchema.parse({ artifactRoot: join(root, "artifacts"), storeRoot: join(root, "store"),
    paths: { modelCache: join(root, "cache") }, evaluation: { inference: { device: "cpu" } },
    gpu: { provider: "aws", instanceId: "i-1234567890abcdef0", user: "ubuntu" } });
  const foundation = engine === "foundation" ? localFoundationSpecFileSchema.parse({ engine: "foundation", name: "Arithmetic",
    examples: [{ input: "2 + 2?", output: "4" }, { input: "3 + 3?", output: "6" }], foundation: { depth: 2 } }) : undefined;
  const checks = await runDoctor(config, undefined, foundation);
  expect(checks.find((check) => check.name === "python-runtime")).toMatchObject({ ok: false, message: expect.stringContaining("Missing local Python dependency") });
  expect(checks.find((check) => check.name === "aws-gpu")?.ok).toBe(true);
  expect(runGpuProcess).toHaveBeenCalledOnce();
});
