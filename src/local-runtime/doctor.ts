import { constants } from "node:fs";
import { access, lstat, mkdir, statfs, unlink, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join, resolve } from "node:path";
import { spawn } from "node:child_process";
import type { FineTuneRunRequest, LocalFoundationSpecFile, LocalRunnerConfig } from "./contracts.js";
import { foundationPlaceholderIssues } from "./local-project.js";
import {
  minimalMachineLearningEnvironment,
  withHuggingFaceCacheEnvironment,
} from "./huggingface-cache.js";
import {
  buildBundledPythonCommand,
  buildFoundationPythonCommand,
  withBundledPythonEnvironment,
  withFoundationPythonEnvironment,
} from "./process-runner.js";
import { resolveTrainingModel } from "./model-registry.js";
import { defaultLocalHome } from "./store.js";
import { verifyLocalBaseModel } from "./prefetch.js";

export interface DoctorCheck {
  name: string;
  ok: boolean;
  message: string;
  details?: Record<string, unknown>;
}

interface CommandResult {
  code: number | null;
  stdout: string;
  stderr: string;
  error?: string;
}

interface CommandOptions {
  cwd?: string;
  env?: NodeJS.ProcessEnv;
  timeoutMs?: number;
}

async function runCommand(command: string, args: string[], options: CommandOptions = {}): Promise<CommandResult> {
  const timeoutMs = options.timeoutMs ?? 10_000;
  let interrupted: string | undefined;
  const result = await new Promise<CommandResult>((resolveResult) => {
    let settled = false;
    let stdout = "";
    let stderr = "";
    let stopError: string | undefined;
    let forceKillTimer: NodeJS.Timeout | null = null;
    const child = spawn(command, args, {
      cwd: options.cwd,
      env: options.env ?? process.env,
      stdio: ["ignore", "pipe", "pipe"],
      detached: process.platform !== "win32",
    });
    const killProcessGroup = (signal: NodeJS.Signals) => {
      if (child.pid && process.platform !== "win32") {
        try {
          process.kill(-child.pid, signal);
          return;
        } catch {
          // The child may have exited between the check and signal.
        }
      }
      child.kill(signal);
    };
    const requestStop = (signal: NodeJS.Signals = "SIGTERM") => {
      killProcessGroup(signal);
      if (!forceKillTimer) {
        forceKillTimer = setTimeout(() => killProcessGroup("SIGKILL"), 5_000);
        forceKillTimer.unref();
      }
    };
    const onSigint = () => {
      interrupted = "interrupted by SIGINT";
      stopError = interrupted;
      requestStop("SIGINT");
    };
    const onSigterm = () => {
      interrupted = "interrupted by SIGTERM";
      stopError = interrupted;
      requestStop("SIGTERM");
    };
    const finish = (result: CommandResult) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (forceKillTimer) clearTimeout(forceKillTimer);
      process.off("SIGINT", onSigint);
      process.off("SIGTERM", onSigterm);
      resolveResult(result);
    };
    const timer = setTimeout(() => {
      stopError = `timed out after ${timeoutMs}ms`;
      requestStop();
    }, timeoutMs);
    timer.unref();
    process.once("SIGINT", onSigint);
    process.once("SIGTERM", onSigterm);
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", (error) => {
      finish({ code: null, stdout, stderr, error: error.message });
    });
    child.on("close", (code) => {
      finish({ code: stopError ? null : code, stdout, stderr, error: stopError });
    });
  });
  if (interrupted) throw new Error(interrupted);
  return result;
}

function firstLine(value: string): string {
  return value.trim().split(/\r?\n/)[0] ?? "";
}

function nvidiaSmiSummary(stdout: string): string {
  const lines = stdout
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  const queried = lines.find((line) =>
    !/^(?:mon|tue|wed|thu|fri|sat|sun)\b/i.test(line)
    && !line.startsWith("+---")
    && !line.startsWith("|")
  );
  if (queried) return queried;
  return firstLine(stdout) || "nvidia-smi reported a GPU";
}

function commandText(command: string, args: string[]): string {
  return [command, ...args].map((part) => /\s/.test(part) ? JSON.stringify(part) : part).join(" ");
}

async function writableDirectoryCheck(name: string, path: string): Promise<DoctorCheck> {
  const resolvedPath = resolve(path);
  const probePath = join(resolvedPath, `.tt-local-write-probe-${process.pid}-${Date.now()}`);
  try {
    await mkdir(resolvedPath, { recursive: true });
    await access(resolvedPath, constants.R_OK | constants.W_OK);
    await writeFile(probePath, "ok\n", { flag: "wx" });
    await unlink(probePath);
    const fs = await statfs(resolvedPath);
    const freeBytes = Number(fs.bavail) * Number(fs.bsize);
    return {
      name,
      ok: true,
      message: `${resolvedPath} is writable (${Math.round(freeBytes / (1024 ** 3) * 10) / 10} GiB available)`,
      details: { path: resolvedPath, free_bytes: freeBytes },
    };
  } catch (error) {
    await unlink(probePath).catch(() => undefined);
    return {
      name,
      ok: false,
      message: `${resolvedPath}: ${error instanceof Error ? error.message : String(error)}`,
      details: { path: resolvedPath },
    };
  }
}

interface PythonProbePlan {
  name: "python-runtime";
  command: string;
  args: string[];
  env: NodeJS.ProcessEnv;
}

function foundationPythonProbePlan(requireBf16: boolean): PythonProbePlan {
  const source = [
    "import json",
    "import numpy, safetensors, tokenizers, torch",
    "assert torch.cuda.is_available(), 'TT Foundation training requires CUDA but torch.cuda.is_available() is false'",
    ...(requireBf16
      ? ["assert torch.cuda.is_bf16_supported(), 'TT Foundation BF16 training was requested but this device does not support BF16'"]
      : []),
    "props = torch.cuda.get_device_properties(0)",
    "print(json.dumps({'python_ok': True, 'torch': torch.__version__, 'cuda_available': True, 'bf16': torch.cuda.is_bf16_supported(), 'cuda_device': torch.cuda.get_device_name(0), 'compute_capability': list(torch.cuda.get_device_capability(0)), 'total_memory_bytes': props.total_memory}))",
  ].join("; ");
  const entrypoint = buildFoundationPythonCommand("-c", [source]);
  return {
    name: "python-runtime",
    command: entrypoint.command,
    args: entrypoint.commandArgs,
    env: withFoundationPythonEnvironment(process.env),
  };
}

function pythonProbeSource(device: LocalRunnerConfig["evaluation"]["inference"]["device"]): string {
  return [
    "import json",
    "import torch, transformers, peft, huggingface_hub",
    `requested = ${JSON.stringify(device)}`,
    "assert torch.cuda.is_available(), 'TT Local training requires CUDA but torch.cuda.is_available() is false'",
    "props = torch.cuda.get_device_properties(0)",
    "print(json.dumps({'python_ok': True, 'torch': torch.__version__, 'transformers': transformers.__version__, 'cuda_available': True, 'cuda_device': torch.cuda.get_device_name(0), 'compute_capability': list(torch.cuda.get_device_capability(0)), 'total_memory_bytes': props.total_memory, 'evaluation_device': requested}))",
  ].join("; ");
}

function pythonProbePlan(args: {
  device: LocalRunnerConfig["evaluation"]["inference"]["device"];
  modelCache?: string;
}): PythonProbePlan {
  const entrypoint = buildBundledPythonCommand(
    "-c",
    [pythonProbeSource(args.device)],
  );
  const env = withBundledPythonEnvironment(
    withHuggingFaceCacheEnvironment(
      minimalMachineLearningEnvironment(process.env),
      args.modelCache,
    ),
  );
  return {
    name: "python-runtime",
    command: entrypoint.command,
    args: entrypoint.commandArgs,
    env,
  };
}

/** Build the exact bundled uv environment that every real stage uses. */
export function buildDoctorPythonPlans(config: LocalRunnerConfig): PythonProbePlan[] {
  if (config.dryRun) return [];
  return [
    pythonProbePlan({
      device: config.evaluation.inference.device,
      modelCache: config.paths.modelCache,
    }),
  ];
}

function foundationPlaceholderCheck(spec: LocalFoundationSpecFile): DoctorCheck {
  const issues = foundationPlaceholderIssues(spec);
  return {
    name: "spec-content",
    ok: issues.length === 0,
    message: issues.length > 0
      ? "The spec still contains generated placeholder content; edit it before training."
      : `${spec.examples.length} spec example(s); foundation engine`,
  };
}

async function foundationCorpusCheck(name: string, path: string): Promise<DoctorCheck> {
  const resolvedPath = resolve(path);
  try {
    const info = await lstat(resolvedPath);
    const supported = info.isFile() || info.isDirectory();
    return {
      name,
      ok: supported && !info.isSymbolicLink(),
      message: supported && !info.isSymbolicLink()
        ? `${resolvedPath} is available for streaming.`
        : `${resolvedPath} must be a regular file or directory, not a symbolic link.`,
      details: { path: resolvedPath },
    };
  } catch (error) {
    return {
      name,
      ok: false,
      message: `${resolvedPath}: ${error instanceof Error ? error.message : String(error)}`,
      details: { path: resolvedPath },
    };
  }
}

function placeholderSpecCheck(request: FineTuneRunRequest): DoctorCheck {
  const placeholder = request.spec_snapshot.examples.some((example) =>
    /replace this with/i.test(example.input) || /replace this with/i.test(example.output)
  ) || /describe the behavior this local model should learn/i.test(request.spec_snapshot.system_prompt);
  return {
    name: "spec-content",
    ok: !placeholder,
    message: placeholder
      ? "The spec still contains generated placeholder content; edit it before training."
      : `${request.spec_snapshot.examples.length} spec example(s); base model ${request.spec_snapshot.base_model}`,
  };
}

async function localBaseModelCheck(
  path: string,
  expectedModelId?: string,
  modelCache?: string,
): Promise<DoctorCheck> {
  const resolvedPath = resolve(path);
  try {
    const verified = await verifyLocalBaseModel(resolvedPath, expectedModelId, modelCache);
    return {
      name: "local-base-model",
      ok: true,
      message: `${resolvedPath} is a valid local Hugging Face snapshot directory.`,
      details: { path: resolvedPath, file_count: verified.fileCount, size_bytes: verified.sizeBytes },
    };
  } catch (error) {
    return {
      name: "local-base-model",
      ok: false,
      message: error instanceof Error ? error.message : String(error),
      details: { path: resolvedPath },
    };
  }
}

export async function runDoctor(
  config: LocalRunnerConfig,
  request?: FineTuneRunRequest,
  foundationSpec?: LocalFoundationSpecFile,
): Promise<DoctorCheck[]> {
  const checks: DoctorCheck[] = [];
  const nodeVersion = process.versions.node;
  const nodeMajor = Number(nodeVersion.split(".")[0]);
  checks.push({ name: "node", ok: nodeMajor >= 22, message: `Node ${nodeVersion}` });

  checks.push(await writableDirectoryCheck("artifact-root", config.artifactRoot));
  checks.push(await writableDirectoryCheck("store-root", config.storeRoot ?? defaultLocalHome()));
  checks.push(await writableDirectoryCheck(
    "model-cache",
    config.paths.modelCache ?? process.env.HF_HOME ?? join(homedir(), ".cache", "huggingface"),
  ));
  if (config.paths.baseModel) {
    checks.push(await localBaseModelCheck(
      config.paths.baseModel,
      request?.spec_snapshot.base_model,
      config.paths.modelCache,
    ));
  }

  if (foundationSpec) {
    checks.push(foundationPlaceholderCheck(foundationSpec));
    if (foundationSpec.foundation.corpus_path) {
      checks.push(await foundationCorpusCheck("pretraining-corpus", foundationSpec.foundation.corpus_path));
    }
    if (foundationSpec.foundation.validation_path) {
      checks.push(await foundationCorpusCheck("validation-corpus", foundationSpec.foundation.validation_path));
    }
    if (
      foundationSpec.foundation.corpus_path
      && foundationSpec.foundation.validation_path
      && resolve(foundationSpec.foundation.corpus_path) === resolve(foundationSpec.foundation.validation_path)
    ) {
      checks.push({
        name: "held-out-corpus",
        ok: false,
        message: "foundation.corpus_path and foundation.validation_path must be distinct.",
      });
    }
    if (foundationSpec.foundation.checkpoint_backup_dir) {
      checks.push(await writableDirectoryCheck(
        "checkpoint-backup",
        foundationSpec.foundation.checkpoint_backup_dir,
      ));
    }
  } else if (request) {
    checks.push(placeholderSpecCheck(request));
    resolveTrainingModel(request.spec_snapshot.base_model);
  }
  const pythonPlans = foundationSpec && !config.dryRun
    ? [foundationPythonProbePlan(foundationSpec.foundation.bf16 !== false)]
    : buildDoctorPythonPlans(config);
  if (pythonPlans.length > 0) {
    const uvVersion = await runCommand("uv", ["--version"]);
    checks.push({
      name: "uv",
      ok: uvVersion.code === 0,
      message: uvVersion.code === 0
        ? firstLine(uvVersion.stdout)
        : uvVersion.error ?? (firstLine(uvVersion.stderr) || "uv is not available"),
    });

    if (uvVersion.code === 0) {
      const uniquePlans = new Map<string, PythonProbePlan>();
      for (const plan of pythonPlans) {
        const key = JSON.stringify([plan.command, plan.args, plan.env]);
        if (!uniquePlans.has(key)) uniquePlans.set(key, plan);
      }
      for (const plan of uniquePlans.values()) {
        const result = await runCommand(plan.command, plan.args, {
          env: plan.env,
          timeoutMs: 1_800_000,
        });
        checks.push({
          name: plan.name,
          ok: result.code === 0,
          message: result.code === 0
            ? firstLine(result.stdout)
            : result.error ?? (firstLine(result.stderr) || `${plan.command} exited ${result.code}`),
          details: { command: commandText(plan.command, plan.args) },
        });
      }
    }
  } else {
    checks.push({
      name: "python-runtime",
      ok: true,
      message: config.dryRun
        ? "Python dependency checks skipped because dryRun is enabled."
        : "Python dependency checks were skipped.",
    });
  }

  const device = config.evaluation.inference.device;
  if (!config.dryRun) {
    const queried = await runCommand("nvidia-smi", [
      "--query-gpu=name,driver_version",
      "--format=csv,noheader",
    ]);
    const nvidiaSmi = queried.code === 0 ? queried : await runCommand("nvidia-smi", []);
    checks.push({
      name: "nvidia-smi",
      ok: nvidiaSmi.code === 0,
      message: nvidiaSmi.code === 0
        ? nvidiaSmiSummary(nvidiaSmi.stdout)
        : nvidiaSmi.error ?? (firstLine(nvidiaSmi.stderr) || "nvidia-smi not available"),
    });
  } else {
    checks.push({
      name: "nvidia-smi",
      ok: true,
      message: config.dryRun
        ? "GPU checks skipped because dryRun is enabled."
        : `nvidia-smi is not required for device=${device}.`,
    });
  }

  const trainingCommand = foundationSpec
    ? buildFoundationPythonCommand("pretrain.py").displayCommand
    : buildBundledPythonCommand("train.py").displayCommand;
  checks.push({
    name: "effective-plan",
    ok: true,
    message: "Resolved the configured training and evaluation plan.",
    details: {
      training_command: trainingCommand ?? null,
      evaluation_provider: foundationSpec ? "foundation" : "transformers",
      evaluation_device: device,
      artifact_root: resolve(config.artifactRoot),
      store_root: resolve(config.storeRoot ?? defaultLocalHome()),
      model_cache: resolve(config.paths.modelCache ?? process.env.HF_HOME ?? join(homedir(), ".cache", "huggingface")),
      base_model: request?.spec_snapshot.base_model ?? null,
      scoring_mode: foundationSpec ? "bpb/chat" : config.evaluation.scoring.mode,
    },
  });

  return checks;
}
