import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { createHash } from "node:crypto";
import { access, mkdir, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { localRunnerConfigSchema } from "../local-runtime/contracts.js";
import { runGpuProcess } from "../local-runtime/gpu-executor.js";
import { ProcessCancelledError } from "../local-runtime/process-runner.js";

let root: string;
let remote: string;
let logPath: string;
const gpu = localRunnerConfigSchema.parse({
  gpu: { provider: "aws", instanceId: "i-1234567890abcdef0", user: "ubuntu" },
}).gpu!;

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), "tt-gpu-review-"));
  logPath = join(root, "training.log");
  remote = `/tmp/tt-gpu-${createHash("sha256").update(logPath).digest("hex").slice(0, 32)}`;
  const bin = join(root, "bin");
  await mkdir(bin);
  await writeFile(join(bin, "aws"), `#!/bin/sh
touch "$TT_GPU_REVIEW_AWS_CALLED"
printf '%s\\n' '{"Reservations":[{"Instances":[{"InstanceId":"${gpu.instanceId}","State":{"Name":"running"},"PublicIpAddress":"127.0.0.1"}]}]}'
`, { mode: 0o700 });
  await writeFile(join(bin, "ssh"), `#!/bin/bash
while [[ "$1" == -* ]]; do
  case "$1" in -o|-i|-l) shift 2;; *) shift;; esac
done
shift
if [[ -n "$TT_GPU_REVIEW_DELAY_RESERVATION" && ! -e "$TT_GPU_REVIEW_FIRST_CALL" ]]; then
  touch "$TT_GPU_REVIEW_FIRST_CALL"
  /bin/bash -c "$*"
  result=$?
  if [[ "$result" -eq 0 ]]; then
    touch "$TT_GPU_REVIEW_READY"
    sleep 30
  fi
  exit "$result"
fi
exec /bin/bash -c "$*"
`, { mode: 0o700 });
  vi.stubEnv("PATH", `${bin}:${process.env.PATH}`);
  vi.stubEnv("TT_GPU_REVIEW_FIRST_CALL", join(root, "first-call"));
  vi.stubEnv("TT_GPU_REVIEW_READY", join(root, "ready"));
  vi.stubEnv("TT_GPU_REVIEW_AWS_CALLED", join(root, "aws-called"));
});

afterEach(async () => {
  vi.unstubAllEnvs();
  await rm(remote, { recursive: true, force: true });
  await rm(root, { recursive: true, force: true });
});

it("preserves cancellation and releases a reservation whose SSH acknowledgement was interrupted", async () => {
  vi.stubEnv("TT_GPU_REVIEW_DELAY_RESERVATION", "1");
  await expect(runGpuProcess({
    gpu, runtime: "adapter", command: "python3", commandArgs: ["-c", "print('unused')"],
    stage: "training", logPath, files: [],
    shouldCancel: () => access(join(root, "ready")).then(() => true, () => false),
  })).rejects.toBeInstanceOf(ProcessCancelledError);
  await expect(access(remote)).rejects.toThrow();
}, 15_000);

it("preserves an existing staging directory and its recoverable artifacts", async () => {
  await mkdir(remote);
  await writeFile(join(remote, "checkpoint"), "previous run");
  await expect(runGpuProcess({
    gpu, runtime: "adapter", command: "python3", commandArgs: ["-c", "print('unused')"],
    stage: "training", logPath, files: [],
  })).rejects.toThrow(/Cannot reserve GPU staging directory/);
  expect(await readFile(join(remote, "checkpoint"), "utf8")).toBe("previous run");
});

it("returns partial artifacts after the remote deadline and removes staging", async () => {
  const output = join(root, "output");
  // Exercise the production timeout command without making the test wait the
  // configuration schema's 60-second minimum for user-supplied deadlines.
  const result = await runGpuProcess({
    gpu: { ...gpu, maxSeconds: 1 }, runtime: "adapter", command: "python3",
    commandArgs: ["-u", "-c", "import os,pathlib,time; pathlib.Path(os.environ['SM_MODEL_DIR'], 'checkpoint').write_text('saved'); time.sleep(300)"],
    env: { SM_MODEL_DIR: output }, stage: "training", logPath,
    files: [{ path: output, direction: "output", directory: true }],
  });
  expect(result.exitCode).toBe(124);
  expect(await readFile(join(output, "checkpoint"), "utf8")).toBe("saved");
  await expect(access(remote)).rejects.toThrow();
}, 20_000);

it("returns checkpoint pruning from a directory that is uploaded and downloaded", async () => {
  const output = join(root, "recovery");
  await mkdir(output);
  await writeFile(join(output, "old-checkpoint"), "old");
  await writeFile(join(output, "retained-checkpoint"), "retained");
  const result = await runGpuProcess({
    gpu, runtime: "adapter", command: "python3",
    commandArgs: ["-c", "import os,pathlib; p=pathlib.Path(os.environ['SM_MODEL_DIR']); p.joinpath('old-checkpoint').unlink(); p.joinpath('new-checkpoint').write_text('new')"],
    env: { SM_MODEL_DIR: output }, stage: "training", logPath,
    files: [{ path: output, direction: "both", directory: true }],
  });
  expect(result.exitCode).toBe(0);
  await expect(access(join(output, "old-checkpoint"))).rejects.toThrow();
  expect(await readFile(join(output, "retained-checkpoint"), "utf8")).toBe("retained");
  expect(await readFile(join(output, "new-checkpoint"), "utf8")).toBe("new");
}, 20_000);

it.each(["same", "parent", "symlink"])("rejects %s writable directory mappings before contacting AWS", async (overlap) => {
  const output = join(root, "recovery");
  await mkdir(output);
  await writeFile(join(output, "state"), "old");
  const alias = join(root, "backup-alias");
  if (overlap === "symlink") await symlink(output, alias);
  await expect(runGpuProcess({
    gpu, runtime: "adapter", command: "python3", commandArgs: ["-c", "print('unused')"],
    env: { SM_MODEL_DIR: output }, stage: "training", logPath,
    files: [
      { path: output, direction: "both", directory: true },
      { path: overlap === "same" ? output : overlap === "symlink" ? alias : root, direction: "both", directory: true },
    ],
  })).rejects.toThrow(/overlap/i);
  expect(await readFile(join(output, "state"), "utf8")).toBe("old");
  await expect(access(join(root, "aws-called"))).rejects.toThrow();
});

it("keeps prior local checkpoints and remote recovery files when a download is interrupted", async () => {
  const output = join(root, "recovery");
  await mkdir(output);
  await writeFile(join(output, "a-state"), "old");
  await writeFile(join(output, "old-checkpoint"), "recoverable");
  // Use actual rsync, throttled and terminated only on the download leg. The
  // small state file arrives before the large payload but must stay uncommitted.
  vi.stubEnv("TT_GPU_REVIEW_ORIGINAL_PATH", process.env.PATH!.split(":").slice(1).join(":"));
  await writeFile(join(root, "bin", "rsync"), `#!/bin/bash
actual_rsync="$(PATH="$TT_GPU_REVIEW_ORIGINAL_PATH" command -v rsync)"
if [[ "\${@: -2:1}" == ubuntu@127.0.0.1:* ]]; then
  "$actual_rsync" --bwlimit=1024 "$@" &
  pid=$!
  sleep 0.5
  kill -TERM "$pid" 2>/dev/null || true
  wait "$pid"
  exit "$?"
fi
exec "$actual_rsync" "$@"
`, { mode: 0o700 });
  await expect(runGpuProcess({
    gpu, runtime: "adapter", command: "python3",
    commandArgs: ["-c", "import os,pathlib; p=pathlib.Path(os.environ['SM_MODEL_DIR']); p.joinpath('a-state').write_text('new'); p.joinpath('old-checkpoint').unlink(); p.joinpath('z-payload').write_bytes(b'x' * 4_194_304)"],
    env: { SM_MODEL_DIR: output }, stage: "training", logPath,
    files: [{ path: output, direction: "both", directory: true }],
  })).rejects.toThrow(/Remote files remain/);
  expect(await readFile(join(output, "a-state"), "utf8")).toBe("old");
  expect(await readFile(join(output, "old-checkpoint"), "utf8")).toBe("recoverable");
  expect(await readFile(join(remote, "files", "0", "recovery", "a-state"), "utf8")).toBe("new");
}, 20_000);
