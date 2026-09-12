import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { localRunnerConfigSchema } from "../local-runtime/contracts.js";
import { parseLocalRunnerConfig } from "../local-runtime/orchestrator.js";
import { buildBundledPythonCommand, runLoggedProcess, ProcessCancelledError } from "../local-runtime/process-runner.js";
import { gpuBaseModelPath, resolveGpuHost, runGpuProcess, shellQuote } from "../local-runtime/gpu-executor.js";

vi.mock("../local-runtime/process-runner.js", async (original) => ({
  ...await original<typeof import("../local-runtime/process-runner.js")>(), runLoggedProcess: vi.fn(),
}));
const gpu = localRunnerConfigSchema.parse({ gpu: { provider: "aws", instanceId: "i-1234567890abcdef0", user: "ubuntu", profile: "research", region: "eu-west-1" } }).gpu!;
let root: string;
let uploadedDocument: Record<string, unknown> | undefined;
let remoteFailure: Error | undefined;
let retrievalFailure = false;

beforeEach(async () => {
  vi.clearAllMocks();
  root = await mkdtemp(join(tmpdir(), "tt-gpu-test-"));
  uploadedDocument = undefined;
  remoteFailure = undefined;
  retrievalFailure = false;
  vi.mocked(runLoggedProcess).mockImplementation(async (args) => {
    if (args.command === "aws") args.onLine?.(JSON.stringify({ Reservations: [{ Instances: [{
      InstanceId: gpu.instanceId, State: { Name: "running" }, PublicIpAddress: "203.0.113.5", PrivateIpAddress: "10.0.0.1",
    }] }] }), "stdout");
    if (args.command === "rsync" && args.commandArgs.at(-2)?.endsWith("document.json")) {
      uploadedDocument = JSON.parse(await readFile(args.commandArgs.at(-2)!, "utf8"));
    }
    if (args.command === "rsync" && args.commandArgs.at(-2)?.startsWith("ubuntu@")) {
      if (retrievalFailure) return { exitCode: 23, stderr: "connection lost" };
      await writeFile(join(root, "out", "checkpoint"), "recovered");
    }
    if (args.commandArgs.includes("-tt") && remoteFailure) throw remoteFailure;
    return { exitCode: 0, stderr: "" };
  });
});
afterEach(async () => { await rm(root, { recursive: true, force: true }); });

async function execute(overrides: Partial<Parameters<typeof runGpuProcess>[0]> = {}) {
  const input = join(root, "config.json");
  await writeFile(input, JSON.stringify({ output_dir: join(root, "out"), prompt: "/keep/this/user/text" }));
  await mkdir(join(root, "out"), { recursive: true });
  const command = buildBundledPythonCommand("train.py");
  return runGpuProcess({ ...command, gpu, runtime: "adapter", stage: "training",
    env: { AWS_SECRET_ACCESS_KEY: "never-forward", TT_TOKEN: "never-forward", HF_TOKEN: "never-forward", HOME: "/laptop/home" },
    files: [{ path: input, direction: "input" }, { path: join(root, "out"), direction: "both", directory: true }],
    document: { path: input, value: JSON.parse(await readFile(input, "utf8")), pathKeys: ["output_dir"] },
    ...overrides });
}

describe("AWS GPU process boundary", () => {
  it("uses named AWS credentials locally and resolves private networking explicitly", async () => {
    expect(await resolveGpuHost({ ...gpu, privateIp: true })).toBe("ubuntu@10.0.0.1");
    expect(vi.mocked(runLoggedProcess).mock.calls[0]![0].commandArgs).toEqual(expect.arrayContaining(["--profile", "research", "--region", "eu-west-1"]));
  });

  it("rejects stopped instances without attempting SSH", async () => {
    vi.mocked(runLoggedProcess).mockImplementationOnce(async (args) => {
      args.onLine?.(JSON.stringify({ Reservations: [{ Instances: [{ InstanceId: gpu.instanceId, State: { Name: "stopped" } }] }] }), "stdout");
      return { exitCode: 0, stderr: "" };
    });
    await expect(resolveGpuHost(gpu)).rejects.toThrow("must already be running");
    expect(runLoggedProcess).toHaveBeenCalledTimes(1);
  });

  it("transfers declared files, rewrites only path fields, and retrieves output before cleanup", async () => {
    await execute();
    expect(uploadedDocument?.output_dir).toMatch(/^\/tmp\/tt-gpu-.*\/files\/1\/out$/);
    expect(uploadedDocument?.prompt).toBe("/keep/this/user/text");
    expect(JSON.parse(await readFile(join(root, "config.json"), "utf8")).output_dir).toBe(join(root, "out"));
    expect(await readFile(join(root, "out", "checkpoint"), "utf8")).toBe("recovered");
    const calls = vi.mocked(runLoggedProcess).mock.calls.map(([call]) => call);
    const launched = calls.find((call) => call.commandArgs.includes("-tt"))!;
    expect(launched.commandArgs).toEqual(expect.arrayContaining(["StrictHostKeyChecking=yes", "BatchMode=yes"]));
    expect(launched.commandArgs.at(-1)).toContain("setsid timeout -k 120 86400s");
    expect(launched.commandArgs.at(-1)).not.toContain("never-forward");
    expect(launched.commandArgs.at(-1)).not.toContain("/laptop/home");
    expect(calls.at(-1)?.commandArgs.at(-1)).toMatch(/then rm -rf -- \/tmp\/tt-gpu-/);
  });

  it("stops remote work and returns checkpoints on cancellation", async () => {
    remoteFailure = new ProcessCancelledError();
    await expect(execute()).rejects.toBeInstanceOf(ProcessCancelledError);
    expect(await readFile(join(root, "out", "checkpoint"), "utf8")).toBe("recovered");
    expect(vi.mocked(runLoggedProcess).mock.calls.some(([call]) => call.commandArgs.at(-1)?.includes("kill -KILL"))).toBe(true);
  });

  it("keeps remote artifacts and reports their location when download fails", async () => {
    retrievalFailure = true;
    await expect(execute()).rejects.toThrow(/Remote files remain at ubuntu@203.0.113.5:\/tmp\/tt-gpu-/);
    expect(vi.mocked(runLoggedProcess).mock.calls.some(([call]) => call.commandArgs.at(-1)?.includes("then rm -rf"))).toBe(false);
  });

  it("does not contact AWS when already cancelled", async () => {
    await expect(execute({ shouldCancel: () => true })).rejects.toBeInstanceOf(ProcessCancelledError);
    expect(runLoggedProcess).not.toHaveBeenCalled();
  });

  it("rejects unsafe config and resolves identity files relative to the config", () => {
    expect(() => localRunnerConfigSchema.parse({ gpu: { ...gpu, user: "ubuntu; whoami" } })).toThrow();
    expect(() => localRunnerConfigSchema.parse({ gpu: { ...gpu, maxSeconds: 0 } })).toThrow();
    const config = parseLocalRunnerConfig({ gpu: { ...gpu, identityFile: "./keys/gpu" } }, join(root, "local-runner.json"));
    expect(config.gpu?.identityFile).toBe(join(root, "keys/gpu"));
    expect(shellQuote("a'b $(oops)")).toBe("'a'\\''b $(oops)'");
  });

  it("selects just a pinned snapshot instead of copying the credential-bearing cache", () => {
    const config = localRunnerConfigSchema.parse({ paths: { modelCache: root } });
    expect(gpuBaseModelPath(config, "org/model", "a".repeat(40))).toBe(join(root, "hub/models--org--model/snapshots", "a".repeat(40)));
    expect(() => gpuBaseModelPath(config, "org/model")).toThrow("immutable");
  });
});
