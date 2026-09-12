import { createHash, randomUUID } from "node:crypto";
import { mkdtemp, mkdir, realpath, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import type { LocalRunnerConfig } from "./contracts.js";
import { runLoggedProcess, ProcessCancelledError } from "./process-runner.js";
import { localRuntimePackageRoot } from "./package-root.js";
import { withHuggingFaceCacheEnvironment } from "./huggingface-cache.js";

export type AwsGpuConfig = NonNullable<LocalRunnerConfig["gpu"]>;
type ProcessArgs = Parameters<typeof runLoggedProcess>[0];
export interface GpuFile {
  path: string;
  direction: "input" | "output" | "both";
  directory?: boolean;
}

export function shellQuote(value: string): string {
  return `'${value.replaceAll("'", "'\\''")}'`;
}

/** Resolve with the user's normal AWS CLI credential chain; credentials never go to EC2. */
export async function resolveGpuHost(gpu: AwsGpuConfig): Promise<string> {
  let stdout = "";
  const result = await runLoggedProcess({
    command: "aws",
    commandArgs: ["ec2", "describe-instances", "--instance-ids", gpu.instanceId,
      ...(gpu.profile ? ["--profile", gpu.profile] : []),
      ...(gpu.region ? ["--region", gpu.region] : []), "--output", "json", "--no-cli-pager"],
    timeoutMs: 30_000,
    stage: "gpu_connect",
    onLine: (line, stream) => { if (stream === "stdout") stdout += line; },
  });
  if (result.exitCode !== 0) throw new Error(`AWS GPU lookup failed: ${result.stderr}`);
  const instances = JSON.parse(stdout).Reservations?.flatMap((r: { Instances?: unknown[] }) => r.Instances ?? []);
  const instance = instances?.find((i: { InstanceId: string }) => i.InstanceId === gpu.instanceId);
  if (instance?.State?.Name !== "running") throw new Error(`AWS GPU instance ${gpu.instanceId} must already be running.`);
  const host = gpu.privateIp ? instance.PrivateIpAddress : instance.PublicIpAddress;
  if (typeof host !== "string" || !/^[0-9.]+$/.test(host)) {
    throw new Error("AWS GPU instance has no selected IP address. Use privateIp with a reachable VPC/VPN, or assign a public IP.");
  }
  return `${gpu.user}@${host}`;
}

function sshOptions(gpu: AwsGpuConfig): string[] {
  return ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=15",
    "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
    ...(gpu.identityFile ? ["-i", gpu.identityFile] : [])];
}

export async function checkAwsGpu(gpu: AwsGpuConfig): Promise<void> {
  const host = await resolveGpuHost(gpu);
  const result = await runLoggedProcess({ command: "ssh", commandArgs: [...sshOptions(gpu), host,
    `bash -lc ${shellQuote("command -v uv && command -v rsync && command -v timeout && command -v setsid && nvidia-smi")}`],
    stage: "gpu_check", timeoutMs: 30_000 });
  if (result.exitCode !== 0) throw new Error(`AWS GPU preflight failed: ${result.stderr}`);
}

/** Only transfer the verified snapshot, never the Hugging Face cache's token files. */
export function gpuBaseModelPath(config: LocalRunnerConfig, model: string, revision?: string): string {
  if (config.paths.baseModel) return resolve(config.paths.baseModel);
  if (!revision || !/^[a-f0-9]{40}$/i.test(revision)) {
    throw new Error("AWS GPU execution needs an immutable base-model revision. Run tt models prefetch first or configure paths.baseModel.");
  }
  const env = withHuggingFaceCacheEnvironment(process.env, config.paths.modelCache);
  return join(env.HF_HUB_CACHE!, `models--${model.replaceAll("/", "--")}`, "snapshots", revision);
}

/** Resolve existing parents too, so a symlink cannot disguise overlapping output trees. */
async function physicalTransferPath(path: string): Promise<string> {
  try { return await realpath(path); }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT" || dirname(path) === path) throw error;
    return join(await physicalTransferPath(dirname(path)), basename(path));
  }
}

async function assertSeparateTransferPaths(files: GpuFile[]): Promise<void> {
  const paths = await Promise.all(files.map((file) => physicalTransferPath(resolve(file.path))));
  for (let i = 0; i < files.length; i += 1) {
    for (let j = 0; j < i; j += 1) {
      if (files[i]!.direction === "input" && files[j]!.direction === "input") continue;
      const left = paths[i]!;
      const right = paths[j]!;
      if (left === right || (files[i]!.directory && right.startsWith(`${left}/`))
        || (files[j]!.directory && left.startsWith(`${right}/`))) {
        throw new Error(`GPU transfer paths overlap: ${files[i]!.path} and ${files[j]!.path}. Use separate input, output, recovery, and backup directories.`);
      }
    }
  }
}

/** One process boundary: the orchestrator and all output validation remain local. */
export async function runGpuProcess(args: ProcessArgs & {
  gpu: AwsGpuConfig;
  runtime: "adapter" | "foundation";
  files: GpuFile[];
  document?: { path: string; value: Record<string, unknown>; pathKeys: string[] };
}): Promise<Awaited<ReturnType<typeof runLoggedProcess>>> {
  if (await args.shouldCancel?.()) throw new ProcessCancelledError();
  await assertSeparateTransferPaths(args.files);
  const host = await resolveGpuHost(args.gpu);
  const options = sshOptions(args.gpu);
  // Stable paths preserve foundation checkpoint corpus identities across resume.
  // An exclusive directory below prevents simultaneous use or accidental deletion of old work.
  const identity = args.logPath ? createHash("sha256").update(resolve(args.logPath)).digest("hex").slice(0, 32) : randomUUID();
  const root = `/tmp/tt-gpu-${identity}`;
  const scratch = await mkdtemp(join(tmpdir(), "tt-gpu-"));
  const runtime = join(localRuntimePackageRoot(import.meta.url), "training", args.runtime);
  const mappings = args.files.map((file, index) => ({ ...file, path: resolve(file.path), remote: `${root}/files/${index}/${basename(file.path)}` }));
  const mapPath = (path: string): string => {
    const absolute = resolve(path);
    const match = [...mappings].sort((a, b) => b.path.length - a.path.length)
      .find((file) => absolute === file.path || (file.directory && absolute.startsWith(`${file.path}/`)));
    if (match) return match.remote + absolute.slice(match.path.length);
    if (absolute === runtime || absolute.startsWith(`${runtime}/`)) return `${root}/runtime${absolute.slice(runtime.length)}`;
    throw new Error(`GPU process path is not declared for transfer: ${path}`);
  };
  const transport = async (command: string, commandArgs: string[], cancellable = true) => {
    const result = await runLoggedProcess({ command, commandArgs, stage: args.stage,
      reporter: args.reporter, timeoutMs: 30 * 60_000,
      shouldCancel: cancellable ? args.shouldCancel : undefined });
    if (result.exitCode !== 0) throw new Error(`${command} failed for ${host}: ${result.stderr}`);
  };
  const ssh = async (script: string, cancellable = true) => transport("ssh", [...options, host, script], cancellable);
  const sync = async (source: string, destination: string, upload: boolean, cancellable = true, mirror = false) => {
    await transport("rsync", ["-rlt", "--copy-links", "--protect-args", "--modify-window=-1", "--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=",
      ...(!upload ? ["--delay-updates", ...(mirror ? ["--delete-delay"] : [])] : []),
      "--exclude=.venv", "--exclude=__pycache__", "--exclude=.git",
      "-e", ["ssh", ...options].map(shellQuote).join(" "), "--",
      upload ? source : `${host}:${source}`, upload ? `${host}:${destination}` : destination], cancellable);
  };
  const grace = Math.ceil((args.shutdownGraceMs ?? 5_000) / 1000);
  const stop = `kill -TERM -- -"$pid" 2>/dev/null || true; remaining=${grace}; while kill -0 -- -"$pid" 2>/dev/null && [ "$remaining" -gt 0 ]; do sleep 1; remaining=$((remaining - 1)); done; kill -KILL -- -"$pid" 2>/dev/null || true`;
  const owner = randomUUID();
  let reservationAttempted = false;
  let started = false;
  let completed = false;
  try {
    reservationAttempted = true;
    await ssh(`umask 077; mkdir ${root} && printf '%s' ${shellQuote(owner)} > ${root}/.owner && mkdir ${root}/runtime ${root}/files`).catch((error) => {
      if (error instanceof ProcessCancelledError) throw error;
      throw new Error(`Cannot reserve GPU staging directory ${host}:${root}. Another process or unrecovered artifacts may be present. ${error instanceof Error ? error.message : error}`);
    });
    for (const name of ["pyproject.toml", "uv.lock", "src"]) {
      await sync(join(runtime, name), `${root}/runtime/`, true);
    }
    for (const file of mappings) {
      await ssh(`mkdir -p ${shellQuote(file.directory ? file.remote : dirname(file.remote))}`);
      if (file.direction !== "output") {
        const exists = await stat(file.path).catch((error: NodeJS.ErrnoException) => {
          if (error.code === "ENOENT" && file.direction === "both") return undefined;
          throw error;
        });
        if (exists) await sync(file.path + (file.directory ? "/" : ""), file.remote + (file.directory ? "/" : ""), true);
      }
    }
    if (args.document) {
      const value = { ...args.document.value };
      for (const key of args.document.pathKeys) if (typeof value[key] === "string") value[key] = mapPath(value[key]);
      const path = join(scratch, "document.json");
      await writeFile(path, JSON.stringify(value), { mode: 0o600 });
      await sync(path, mapPath(args.document.path), true);
    }
    // Deliberately forward only process settings, not the laptop's environment or credentials.
    const env: Record<string, string> = { UV_PROJECT_ENVIRONMENT: `${root}/venv`, TORCH_DISABLE_NATIVE_JIT: "1",
      HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1" };
    for (const [key, value] of Object.entries(args.env ?? {})) {
      if (value === undefined) continue;
      if (key === "BACKEND") env[key] = value;
      if (["SM_CHANNEL_TRAINING", "SM_CHANNEL_BASE_MODEL", "SM_MODEL_DIR", "SM_OUTPUT_DIR", "TT_HYPERPARAMETERS_PATH"].includes(key)) env[key] = mapPath(value);
    }
    const commandArgs = args.commandArgs.map((value) => value.startsWith("/") ? mapPath(value) : value);
    const command = ["env", ...Object.entries(env).map(([key, value]) => `${key}=${value}`), args.command, ...commandArgs].map(shellQuote).join(" ");
    // The remote deadline also bounds orphaned work after a lost connection or laptop power loss.
    const script = `umask 077; cd ${root}; setsid timeout -k 120 ${args.gpu.maxSeconds}s ${command} & pid=$!; echo "$pid" > pid; trap ${shellQuote(stop)} HUP INT TERM; wait "$pid"`;
    await args.reporter?.onEvent?.({ stage: args.stage, status: "running", message: `Running GPU process on ${args.gpu.instanceId}.`, details: { remote_directory: root } });
    started = true;
    const result = await runLoggedProcess({ ...args, env: process.env, command: "ssh",
      commandArgs: [...options, "-tt", host, `bash -lc ${shellQuote(script)}`] });
    completed = result.exitCode === 0;
    return result;
  } finally {
    try {
      if (started) {
        await ssh(`bash -c ${shellQuote(`if test -f ${root}/pid; then pid=$(cat ${root}/pid); ${stop}; fi`)}`, false);
        for (const file of mappings.filter((file) => file.direction !== "input")) {
          await mkdir(file.directory ? file.path : dirname(file.path), { recursive: true });
          if (!completed && !file.directory) {
            const exists = await runLoggedProcess({ command: "ssh", commandArgs: [...options, host, `test -e ${shellQuote(file.remote)}`], stage: args.stage, timeoutMs: 30_000 });
            if (exists.exitCode === 1) continue;
            if (exists.exitCode !== 0) throw new Error("Cannot inspect remote outputs.");
          }
          await sync(file.remote + (file.directory ? "/" : ""), file.path + (file.directory ? "/" : ""), false, false, file.directory === true && file.direction === "both");
        }
      }
      // SSH may have been interrupted after reserving the directory but before replying.
      // Only remove a reservation made by this invocation, never another run's files.
      if (reservationAttempted) await ssh(`if [ "$(cat ${root}/.owner 2>/dev/null)" = ${shellQuote(owner)} ]; then rm -rf -- ${root}; fi`, false);
    } catch (error) {
      throw new Error(`GPU cleanup or artifact retrieval failed. Remote files remain at ${host}:${root}. ${error instanceof Error ? error.message : error}`);
    } finally {
      await rm(scratch, { recursive: true, force: true });
    }
  }
}
