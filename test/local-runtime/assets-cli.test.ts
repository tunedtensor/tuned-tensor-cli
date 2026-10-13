import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { chmod, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const projectRoot = dirname(dirname(dirname(fileURLToPath(import.meta.url))));
const cliPath = join(projectRoot, "src", "local-runtime", "index.ts");
const tsxLoader = import.meta.resolve("tsx");
const revision = "0123456789abcdef0123456789abcdef01234567";
const modelRevision = "1111111111111111111111111111111111111111";

/**
 * A stand-in for `uv run … prefetch*.py`: it lays files out the way
 * huggingface_hub does (snapshot links into content-addressed blobs) and
 * writes the report the Node side expects.
 */
const FAKE_UV = `#!/usr/bin/env node
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const args = process.argv.slice(2);
const script = args.find((value) => value.endsWith(".py")) ?? "";
const input = JSON.parse(fs.readFileSync(args[args.indexOf("--input") + 1], "utf8"));
const output = args[args.indexOf("--output") + 1];
const hub = process.env.HF_HUB_CACHE;
fs.appendFileSync(path.join(process.env.HF_HOME, "payloads.jsonl"), JSON.stringify(input) + "\\n");
function place(kind, repo, rev, name, content) {
  const repository = path.join(hub, kind + "--" + repo.replace("/", "--"));
  const blob = path.join(repository, "blobs", crypto.createHash("sha256").update(content).digest("hex"));
  const link = path.join(repository, "snapshots", rev, name);
  fs.mkdirSync(path.dirname(blob), { recursive: true });
  fs.mkdirSync(path.dirname(link), { recursive: true });
  fs.writeFileSync(blob, content);
  if (!fs.existsSync(link)) fs.symlinkSync(blob, link);
  fs.mkdirSync(path.join(repository, "refs"), { recursive: true });
  fs.writeFileSync(path.join(repository, "refs", "main"), rev);
  return { link, content, snapshot: path.join(repository, "snapshots", rev) };
}
if (script.endsWith("prefetch_dataset.py")) {
  const rev = input.revision ?? ${JSON.stringify(revision)};
  const files = {};
  let snapshot;
  for (const [split, name] of Object.entries(input.files)) {
    const content = JSON.stringify({ messages: [{ role: "user", content: split }, { role: "assistant", content: "ok" }] }) + "\\n";
    const placed = place("datasets", input.repo, rev, name, content);
    snapshot = placed.snapshot;
    files[split] = { path: name, size_bytes: Buffer.byteLength(content), sha256: crypto.createHash("sha256").update(content).digest("hex") };
  }
  fs.writeFileSync(output, JSON.stringify({ ok: true, repo: input.repo, snapshot_revision: rev, snapshot_path: snapshot, files }));
} else if (script.endsWith("prefetch.py")) {
  const rev = input.revision ?? ${JSON.stringify(modelRevision)};
  let snapshot;
  for (const [name, content] of [["config.json", "{}"], ["tokenizer_config.json", "{}"], ["model.safetensors", "weights"]]) {
    snapshot = place("models", input.base_model, rev, name, content).snapshot;
  }
  fs.writeFileSync(output, JSON.stringify({
    ok: true, base_model: input.base_model, model_cache: process.env.HF_HOME, hf_home: process.env.HF_HOME,
    hub_cache: hub, snapshot_path: snapshot, snapshot_revision: rev, file_count: 3, size_bytes: 11,
  }));
} else {
  process.exit(23);
}
`;

function tt(args: string[], cwd: string, root: string) {
  const result = spawnSync(process.execPath, ["--import", tsxLoader, cliPath, ...args], {
    cwd,
    encoding: "utf8",
    env: {
      ...process.env,
      PATH: `${join(root, "bin")}:${process.env.PATH ?? ""}`,
      TUNED_TENSOR_HOME: join(root, "tt-home"),
      TT_LOCAL_HOME: join(root, "store"),
      HF_HOME: join(root, "hf"),
    },
  });
  return { status: result.status, stdout: result.stdout, stderr: result.stderr };
}

test("downloads by repo, spec downloads pin the lock, and assets list, locate and verify them", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-assets-cli-"));
  try {
    const project = join(root, "project");
    await mkdir(project, { recursive: true });
    await mkdir(join(root, "bin"), { recursive: true });
    await mkdir(join(root, "hf"), { recursive: true });
    await writeFile(join(root, "bin", "uv"), FAKE_UV);
    await chmod(join(root, "bin", "uv"), 0o755);
    const specPath = join(project, "tunedtensor.json");
    await writeFile(specPath, JSON.stringify({
      name: "Chat",
      system_prompt: "Be helpful.",
      base_model: "Qwen/Qwen3.5-2B",
      examples: [],
      dataset_prebuilt: {
        huggingface: { repo: "org/chat", columns: { messages: "messages" } },
        training: "data/train.jsonl",
        test: "data/test.jsonl",
      },
    }));

    // 1. Any dataset can be downloaded without touching the spec or its lock.
    let result = tt(["datasets", "prefetch", "--repo", "org/other", "--training", "train.jsonl", "--quiet"], project, root);
    assert.equal(result.status, 0, result.stderr);
    assert.equal(JSON.parse(result.stdout).revision, revision);
    assert.equal(existsSync(join(project, "tunedtensor.lock.json")), false);

    // 2. Any model can be downloaded by repo; an uncertified one resolves its latest commit.
    result = tt(["models", "prefetch", "--repo", "org/model", "--quiet"], project, root);
    assert.equal(result.status, 0, result.stderr);
    assert.equal(JSON.parse(result.stdout).snapshot_revision, modelRevision);
    const payloads = (await readFile(join(root, "hf", "payloads.jsonl"), "utf8")).trim().split("\n").map((line) => JSON.parse(line));
    assert.equal(payloads[1].base_model, "org/model");
    assert.equal(payloads[1].resolve_latest, true);
    assert.equal(payloads[1].revision, undefined);

    // 3. The spec's own dataset download pins it in the lock beside the spec.
    result = tt(["datasets", "prefetch", "tunedtensor.json", "--quiet"], project, root);
    assert.equal(result.status, 0, result.stderr);
    const lock = JSON.parse(await readFile(join(project, "tunedtensor.lock.json"), "utf8"));
    assert.equal(lock.version, 1);
    assert.equal(lock.dataset.repo, "org/chat");
    assert.equal(lock.dataset.revision, revision);
    assert.deepEqual(Object.keys(lock.dataset.files), ["data/test.jsonl", "data/train.jsonl"]);
    assert.match(lock.dataset.files["data/train.jsonl"], /^sha256:[0-9a-f]{64}$/);
    // Prefetching again keeps an unpinned spec dataset on its locked commit.
    result = tt(["datasets", "prefetch", "tunedtensor.json", "--quiet"], project, root);
    assert.equal(result.status, 0, result.stderr);
    const last = (await readFile(join(root, "hf", "payloads.jsonl"), "utf8")).trim().split("\n").map((line) => JSON.parse(line)).at(-1);
    assert.equal(last.revision, revision);

    // 4. Downloads are listed in one place, can be located and verified.
    result = tt(["assets", "list"], project, root);
    const assets = JSON.parse(result.stdout) as Array<{ kind: string; repo: string; manifest_path: string }>;
    assert.deepEqual(assets.map((asset) => asset.repo).sort(), ["org/chat", "org/model", "org/other"]);
    assert.ok(assets.every((asset) => asset.manifest_path.startsWith(join(root, "tt-home", "assets"))));
    result = tt(["assets", "path", "org/model"], project, root);
    assert.equal(result.stdout.trim(), join(root, "hf", "hub", "models--org--model", "snapshots", modelRevision));
    result = tt(["assets", "verify"], project, root);
    assert.equal(result.status, 0, result.stderr);
    assert.equal(JSON.parse(result.stdout).ok, true);

    // 5. Changed bytes are reported and fail the command.
    await writeFile(join(root, "hf", "hub", "models--org--model", "snapshots", modelRevision, "model.safetensors"), "tampered");
    result = tt(["assets", "verify", "org/model"], project, root);
    assert.equal(result.status, 2);
    assert.match(JSON.parse(result.stdout).assets[0].problems[0], /model\.safetensors: content does not match/);

    // An invalid workspace spec cannot block an independent download or alter its lock.
    const originalLock = await readFile(join(project, "tunedtensor.lock.json"), "utf8");
    await writeFile(specPath, "{unfinished");
    result = tt(["datasets", "prefetch", "--repo", "org/direct", "--training", "train.jsonl", "--quiet"], project, root);
    assert.equal(result.status, 0, result.stderr);
    assert.equal(JSON.parse(result.stdout).repo, "org/direct");
    assert.equal(await readFile(join(project, "tunedtensor.lock.json"), "utf8"), originalLock);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
