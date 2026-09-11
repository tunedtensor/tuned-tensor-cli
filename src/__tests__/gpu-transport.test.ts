import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { mkdtemp, mkdir, readFile, rm, writeFile, access } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { runGpuProcess } from "../local-runtime/gpu-executor.js";
import { localRunnerConfigSchema } from "../local-runtime/contracts.js";

// Real process and rsync integration, with a local shell standing in for the SSH server.
let root: string;
beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), "tt-transport-"));
  const bin = join(root, "bin");
  await mkdir(bin);
  await writeFile(join(bin, "aws"), `#!/bin/sh\nprintf '%s\\n' '{"Reservations":[{"Instances":[{"InstanceId":"i-1234567890abcdef0","State":{"Name":"running"},"PublicIpAddress":"127.0.0.1"}]}]}'\n`, { mode: 0o700 });
  await writeFile(join(bin, "ssh"), `#!/bin/bash
while [[ "$1" == -* ]]; do
  case "$1" in -o|-i|-l) shift 2;; *) shift;; esac
done
shift
exec /bin/bash -c "$*"
`, { mode: 0o700 });
  vi.stubEnv("PATH", `${bin}:${process.env.PATH}`);
});
afterEach(async () => { vi.unstubAllEnvs(); await rm(root, { recursive: true, force: true }); });

it("executes a real process, copies artifacts home, and removes staging", async () => {
  const output = join(root, "output with ' quotes");
  let remote = "";
  const gpu = localRunnerConfigSchema.parse({ gpu: { provider: "aws", instanceId: "i-1234567890abcdef0", user: "ubuntu" } }).gpu!;
  const result = await runGpuProcess({ gpu, runtime: "adapter", command: "python3",
    commandArgs: ["-c", "import os, pathlib; pathlib.Path(os.environ['SM_MODEL_DIR'], 'model.txt').write_text('weights')"],
    env: { SM_MODEL_DIR: output }, stage: "training", files: [{ path: output, direction: "output", directory: true }],
    reporter: { onEvent(event) { remote = String(event.details?.remote_directory); } },
  });
  expect(result.exitCode).toBe(0);
  expect(await readFile(join(output, "model.txt"), "utf8")).toBe("weights");
  await expect(access(remote)).rejects.toThrow();
}, 20_000);

it("terminates remote descendants on cancellation and recovers the checkpoint", async () => {
  const output = join(root, "output");
  let ready = false;
  const gpu = localRunnerConfigSchema.parse({ gpu: { provider: "aws", instanceId: "i-1234567890abcdef0", user: "ubuntu" } }).gpu!;
  const result = runGpuProcess({ gpu, runtime: "adapter", command: "python3",
    commandArgs: ["-u", "-c", "import os, pathlib, time; pathlib.Path(os.environ['SM_MODEL_DIR'], 'checkpoint').write_text('saved'); print('READY', flush=True); time.sleep(300)"],
    env: { SM_MODEL_DIR: output }, stage: "training", files: [{ path: output, direction: "output", directory: true }],
    onLine(line) { if (line.includes('READY')) ready = true; }, shouldCancel: () => ready,
  });
  await expect(result).rejects.toThrow(/cancelled/);
  expect(await readFile(join(output, "checkpoint"), "utf8")).toBe("saved");
}, 20_000);

it("keeps corpus extensions, paths and nanosecond timestamps stable across resume", async () => {
  const corpus = join(root, "corpus.jsonl");
  const output = join(root, "output");
  const configPath = join(root, "config.json");
  await writeFile(corpus, '{"text":"example"}\n');
  await writeFile(configPath, '{}');
  const gpu = localRunnerConfigSchema.parse({ gpu: { provider: "aws", instanceId: "i-1234567890abcdef0", user: "ubuntu" } }).gpu!;
  const run = () => runGpuProcess({ gpu, runtime: "foundation", command: "python3",
    commandArgs: ["-c", "import sys,json,pathlib; c=json.load(open(sys.argv[1])); p=pathlib.Path(c['corpus_path']); assert p.suffix == '.jsonl'; pathlib.Path(c['output_dir'], 'manifest').write_text(str(p)+':'+str(p.stat().st_mtime_ns))", configPath],
    logPath: join(root, "step.log"), stage: "pretrain",
    files: [{ path: corpus, direction: "input" }, { path: configPath, direction: "input" }, { path: output, direction: "both", directory: true }],
    document: { path: configPath, value: { corpus_path: corpus, output_dir: output }, pathKeys: ["corpus_path", "output_dir"] },
  });
  await run();
  const first = await readFile(join(output, "manifest"), "utf8");
  await run();
  expect(await readFile(join(output, "manifest"), "utf8")).toBe(first);
}, 20_000);
