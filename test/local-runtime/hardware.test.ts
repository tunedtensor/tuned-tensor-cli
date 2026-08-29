import assert from "node:assert/strict";
import { chmod, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";
import { test } from "node:test";
import { localRunnerConfigSchema } from "../../src/local-runtime/contracts.js";
import { assessHardware } from "../../src/local-runtime/hardware.js";
import { hardwareSnapshotPath } from "../../src/local-runtime/hardware-snapshot.js";

test("hardware inventory uses nvidia-smi and writes a snapshot without requiring CUDA torch", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-hardware-"));
  const previousPath = process.env.PATH;
  const previousHome = process.env.TUNED_TENSOR_HOME;
  try {
    const bin = join(root, "bin");
    await mkdir(bin);
    await writeFile(join(bin, "uv"), `#!/bin/sh
if [ "$1" = "--version" ]; then
  echo "uv 0.test"
  exit 0
fi
echo "should not run python in quick mode" >&2
exit 1
`, "utf8");
    await writeFile(join(bin, "nvidia-smi"), `#!/bin/sh
if echo "$*" | grep -q query-gpu; then
  echo "0, NVIDIA GB10, 580.00, 131072, 120000, 12.1"
  exit 0
fi
echo "NVIDIA GB10"
exit 0
`, "utf8");
    await chmod(join(bin, "uv"), 0o755);
    await chmod(join(bin, "nvidia-smi"), 0o755);
    process.env.PATH = `${bin}${delimiter}${previousPath ?? ""}`;
    process.env.TUNED_TENSOR_HOME = join(root, "home");

    const report = await assessHardware({
      quick: true,
      cwd: root,
      env: { ...process.env, TUNED_TENSOR_HOME: join(root, "home") },
      config: localRunnerConfigSchema.parse({
        artifactRoot: join(root, "artifacts"),
        storeRoot: join(root, "store"),
        paths: { modelCache: join(root, "cache") },
      }),
    });

    assert.equal(report.quick, true);
    assert.equal(report.capabilities.cuda_available, true);
    assert.match(report.capabilities.gpu?.name ?? "", /GB10/);
    const qwen = report.capabilities.adapters.find((item) => item.id === "Qwen/Qwen3.5-2B");
    const nemotron = report.capabilities.adapters.find((item) => item.id.includes("Nemotron"));
    assert.equal(qwen?.train.status, "ready");
    assert.equal(nemotron?.train.status, "ready");
    const saved = JSON.parse(await readFile(hardwareSnapshotPath({
      ...process.env,
      TUNED_TENSOR_HOME: join(root, "home"),
    }), "utf8")) as { summary: string };
    assert.match(saved.summary, /CUDA yes/);
  } finally {
    if (previousPath === undefined) delete process.env.PATH;
    else process.env.PATH = previousPath;
    if (previousHome === undefined) delete process.env.TUNED_TENSOR_HOME;
    else process.env.TUNED_TENSOR_HOME = previousHome;
    await rm(root, { recursive: true, force: true });
  }
});

test("hardware inventory reports no GPU as CUDA unavailable", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-hardware-cpu-"));
  const previousPath = process.env.PATH;
  const previousHome = process.env.TUNED_TENSOR_HOME;
  try {
    const bin = join(root, "bin");
    await mkdir(bin);
    await writeFile(join(bin, "uv"), "#!/bin/sh\necho uv 0.test\n", "utf8");
    await writeFile(join(bin, "nvidia-smi"), "#!/bin/sh\necho 'nvidia-smi not found' >&2\nexit 127\n", "utf8");
    await chmod(join(bin, "uv"), 0o755);
    await chmod(join(bin, "nvidia-smi"), 0o755);
    process.env.PATH = `${bin}${delimiter}${previousPath ?? ""}`;
    process.env.TUNED_TENSOR_HOME = join(root, "home");

    const report = await assessHardware({
      quick: true,
      cwd: root,
      env: { ...process.env, TUNED_TENSOR_HOME: join(root, "home") },
    });
    assert.equal(report.capabilities.cuda_available, false);
    assert.equal(report.capabilities.foundation.train.status, "not_possible");
  } finally {
    if (previousPath === undefined) delete process.env.PATH;
    else process.env.PATH = previousPath;
    if (previousHome === undefined) delete process.env.TUNED_TENSOR_HOME;
    else process.env.TUNED_TENSOR_HOME = previousHome;
    await rm(root, { recursive: true, force: true });
  }
});

test("hardware inventory parses the nvidia-smi table instead of the weekday header", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-hardware-table-"));
  const previousPath = process.env.PATH;
  const previousHome = process.env.TUNED_TENSOR_HOME;
  try {
    const bin = join(root, "bin");
    await mkdir(bin);
    await writeFile(join(bin, "uv"), `#!/bin/sh
echo "uv 0.test"
exit 0
`, "utf8");
    await writeFile(join(bin, "nvidia-smi"), `#!/bin/sh
if echo "$*" | grep -q query-gpu; then
  echo "Field compute_cap is not a valid field to query." >&2
  exit 2
fi
cat <<'EOF'
Sat Aug 29 15:30:00 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.86.16              Driver Version: 570.86.16      CUDA Version: 12.8     |
|   0  NVIDIA GB10                     On  |   00000000:01:00.0 Off |                  N/A |
+-----------------------------------------------------------------------------------------+
EOF
exit 0
`, "utf8");
    await chmod(join(bin, "uv"), 0o755);
    await chmod(join(bin, "nvidia-smi"), 0o755);
    process.env.PATH = `${bin}${delimiter}${previousPath ?? ""}`;
    process.env.TUNED_TENSOR_HOME = join(root, "home");

    const report = await assessHardware({
      quick: true,
      cwd: root,
      env: { ...process.env, TUNED_TENSOR_HOME: join(root, "home") },
    });
    assert.match(report.capabilities.gpu?.name ?? "", /GB10/);
    assert.doesNotMatch(report.capabilities.gpu?.name ?? "", /Sat|Aug/);
    assert.equal(report.capabilities.cuda_available, true);
  } finally {
    if (previousPath === undefined) delete process.env.PATH;
    else process.env.PATH = previousPath;
    if (previousHome === undefined) delete process.env.TUNED_TENSOR_HOME;
    else process.env.TUNED_TENSOR_HOME = previousHome;
    await rm(root, { recursive: true, force: true });
  }
});

test("a quick probe does not replace a fresh full hardware snapshot", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-hardware-keep-"));
  const previousPath = process.env.PATH;
  const previousHome = process.env.TUNED_TENSOR_HOME;
  try {
    const bin = join(root, "bin");
    const home = join(root, "home");
    await mkdir(bin);
    await mkdir(home);
    await writeFile(join(bin, "uv"), "#!/bin/sh\necho uv 0.test\nexit 0\n", "utf8");
    await writeFile(join(bin, "nvidia-smi"), `#!/bin/sh
echo "0, NVIDIA GB10, 580.00, 131072, 120000"
exit 0
`, "utf8");
    await chmod(join(bin, "uv"), 0o755);
    await chmod(join(bin, "nvidia-smi"), 0o755);
    process.env.PATH = `${bin}${delimiter}${previousPath ?? ""}`;
    process.env.TUNED_TENSOR_HOME = home;

    await writeFile(join(home, "hardware.json"), `${JSON.stringify({
      version: 1,
      collected_at: new Date().toISOString(),
      quick: false,
      gpu_fingerprint: { name: "NVIDIA GB10", total_memory_bytes: 137438953472 },
      inventory: { collected_at: new Date().toISOString(), quick: false },
      capabilities: {
        cuda_available: false,
        gpu: { index: 0, name: "NVIDIA GB10", memory_total_bytes: 137438953472, unified_memory: true },
        adapters: [],
        foundation: {
          default_depth: 2,
          suggested_max_depth: 2,
          train: { status: "not_possible", reason: "Foundation pretrain, SFT, and RL require CUDA" },
          finetune: { status: "not_possible", reason: "Foundation pretrain, SFT, and RL require CUDA" },
          inference: { status: "ready" },
          serve: { status: "not_possible", reason: "tt serve cannot host foundation checkpoints yet" },
        },
        notes: [],
      },
      summary: "GPU NVIDIA GB10, CUDA no",
    })}\n`);

    const report = await assessHardware({
      quick: true,
      cwd: root,
      env: { ...process.env, TUNED_TENSOR_HOME: home },
    });
    assert.equal(report.quick, false);
    assert.equal(report.capabilities.cuda_available, false);
    assert.equal(report.summary, "GPU NVIDIA GB10, CUDA no");
  } finally {
    if (previousPath === undefined) delete process.env.PATH;
    else process.env.PATH = previousPath;
    if (previousHome === undefined) delete process.env.TUNED_TENSOR_HOME;
    else process.env.TUNED_TENSOR_HOME = previousHome;
    await rm(root, { recursive: true, force: true });
  }
});
