import assert from "node:assert/strict";
import { writeFileSync } from "node:fs";
import { chmod, mkdtemp, readFile, readdir, rm, stat, symlink, unlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { delimiter, join, resolve } from "node:path";
import { test } from "node:test";
import { localRunnerConfigSchema } from "../../src/local-runtime/contracts.js";
import {
  baselineCacheKey,
  deriveSampleSeed,
  evaluateExamples,
  INFERENCE_PROTOCOL_VERSION,
  sampleExamples,
  splitSpecExamples,
  tokenF1,
} from "../../src/local-runtime/evaluation.js";
import { QWEN_3_5_2B_REVISION } from "../../src/local-runtime/model-registry.js";

const inheritedPath = process.env.PATH ?? "";

async function writeFakeUv(root: string): Promise<string> {
  const path = join(root, "uv");
  await writeFile(path, `#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

if sys.argv[1:] == ["--version"]:
    print("uv 0.test")
    raise SystemExit(0)

input_path = sys.argv[sys.argv.index("--input") + 1]
output_path = sys.argv[sys.argv.index("--output") + 1]
payload = json.load(open(input_path))
assert payload["protocol_version"] == 2
assert payload["model_loader"] == "causal_lm"
assert payload["trust_remote_code"] is False
assert all(set(example) == {"id", "input"} for example in payload["examples"])

config_path = Path(__file__).with_name("fake-uv-config.json")
config = json.loads(config_path.read_text()) if config_path.exists() else {}
if config.get("base_model"):
    assert payload["base_model"] == config["base_model"]
counter_path = config.get("counter")
if counter_path:
    current = int(open(counter_path).read()) if os.path.exists(counter_path) else 0
    open(counter_path, "w").write(str(current + 1))

mode = config.get("mode", "success")
if mode == "no-output":
    raise SystemExit(0)
if mode == "invalid-json":
    open(output_path, "w").write("{")
    raise SystemExit(0)

answers = config.get("answers", {})
results = [
    {
        "id": example["id"],
        "actual": answers.get(example["input"], example["input"]),
        "latency_ms": 10 + index,
    }
    for index, example in enumerate(payload["examples"])
]
results.reverse()
if mode == "short":
    results = results[:1]
elif mode == "duplicate" and len(results) > 1:
    results[1]["id"] = results[0]["id"]
elif mode == "bad-actual":
    results[0]["actual"] = 42
elif mode == "bad-latency":
    results[0]["latency_ms"] = -1

print(json.dumps({
    "HF_HOME": os.environ.get("HF_HOME"),
    "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
}))
json.dump({
    "provider": "transformers",
    "model_id": payload["model_id"],
    "base_model": payload["base_model"],
    "adapter_path": payload.get("adapter_path"),
    "generation_config": payload["generation"],
    "results": results,
}, open(output_path, "w"))
`, "utf8");
  await chmod(path, 0o755);
  process.env.PATH = `${root}${delimiter}${inheritedPath}`;
  return path;
}

function configFor(root: string, options: {
  answers?: Record<string, string>;
  mode?: string;
  counter?: string;
  scoring?: { mode: "exact_match" } | { mode: "json_fields"; fields?: string[] };
  baselineCache?: boolean;
  baseModel?: string;
  expectedBaseModel?: string;
} = {}) {
  writeFileSync(join(root, "fake-uv-config.json"), JSON.stringify({
    answers: options.answers ?? {},
    ...(options.mode ? { mode: options.mode } : {}),
    ...(options.counter ? { counter: options.counter } : {}),
    ...(options.expectedBaseModel ? { base_model: options.expectedBaseModel } : {}),
  }));
  return localRunnerConfigSchema.parse({
    storeRoot: join(root, "store"),
    paths: {
      modelCache: join(root, "huggingface"),
      ...(options.baseModel ? { baseModel: options.baseModel } : {}),
    },
    evaluation: {
      inference: {
        device: "cpu",
        maxNewTokens: 32,
        temperature: 0,
        topP: 1,
      },
      scoring: options.scoring ?? { mode: "exact_match" },
      baselineCache: options.baselineCache ?? false,
    },
  });
}

test("Transformers evaluation uses the strict text-only protocol and opaque IDs", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-evaluation-"));
  try {
    await writeFakeUv(root);
    const outputPath = join(root, "candidate.json");
    const report = await evaluateExamples({
      kind: "candidate",
      modelId: "local-adapter",
      baseModelId: "Qwen/Qwen3.5-2B",
      baseModelRevision: QWEN_3_5_2B_REVISION,
      adapterPath: `file://${join(root, "adapter")}`,
      examples: [
        { input: "first", output: "one" },
        { input: "second", output: "two" },
      ],
      system: "Return one word.",
      config: configFor(root, { answers: { first: "one", second: "two" } }),
      outputPath,
    });

    assert.equal(report.inference_provider, "transformers");
    assert.equal(report.scoring_method, "exact_match");
    assert.deepEqual(report.results.map((result) => result.actual), ["one", "two"]);
    assert.equal(report.avg_score, 1);
    assert.equal(report.exact_match_rate, 1);

    const payload = JSON.parse(
      await readFile(`${outputPath}.inference-input.json`, "utf8"),
    ) as Record<string, any>;
    assert.equal(payload.protocol_version, INFERENCE_PROTOCOL_VERSION);
    assert.equal(payload.base_model_revision, QWEN_3_5_2B_REVISION);
    assert.equal(payload.model_loader, "causal_lm");
    assert.equal(payload.trust_remote_code, false);
    assert.equal(payload.adapter_path, join(root, "adapter"));
    assert.deepEqual(payload.examples, [
      { id: "0", input: "first" },
      { id: "1", input: "second" },
    ]);
    assert.deepEqual(payload.generation, {
      max_new_tokens: 32,
      temperature: 0,
      top_p: 1,
    });
    assert.equal(payload.model_cache, resolve(root, "huggingface"));

    const cacheLine = JSON.parse(
      (await readFile(`${outputPath}.inference.log`, "utf8")).trim(),
    ) as Record<string, string>;
    assert.equal(cacheLine.HF_HOME, join(root, "huggingface"));
    assert.equal(cacheLine.HF_HUB_CACHE, join(root, "huggingface", "hub"));
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("local evaluation keeps canonical model identity separate from the snapshot source", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-evaluation-source-"));
  try {
    await writeFakeUv(root);
    const modelSource = join(root, "nemotron-snapshot");
    const outputPath = join(root, "baseline.json");
    await evaluateExamples({
      kind: "baseline",
      modelId: modelSource,
      baseModelId: "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
      modelSource,
      baseModelRevision: "ce38b6ab8b252b4b8ee7165b4605e93191cafd73",
      examples: [{ input: "hello", output: "hello" }],
      system: "Answer.",
      config: configFor(root, {
        baseModel: modelSource,
        expectedBaseModel: "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
      }),
      outputPath,
    });
    const payload = JSON.parse(
      await readFile(`${outputPath}.inference-input.json`, "utf8"),
    ) as Record<string, unknown>;
    assert.equal(
      payload.base_model,
      "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    );
    assert.equal(payload.model_source, modelSource);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("local evaluation rejects a non-certified explicit Nemotron revision before Python starts", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-evaluation-revision-"));
  try {
    await writeFakeUv(root);
    const modelSource = join(root, "nemotron-snapshot");
    await assert.rejects(
      evaluateExamples({
        kind: "baseline",
        modelId: modelSource,
        baseModelId: "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        modelSource,
        baseModelRevision: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        examples: [{ input: "hello", output: "hello" }],
        system: "Answer.",
        config: configFor(root, { baseModel: modelSource }),
        outputPath: join(root, "baseline.json"),
      }),
      /must use certified revision/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("strict evaluator output rejects incomplete, duplicate, and malformed predictions", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-evaluation-invalid-"));
  try {
    await writeFakeUv(root);
    const cases = [
      ["short", /returned 1 predictions; expected 2/],
      ["duplicate", /duplicate id/],
      ["bad-actual", /must include string actual/],
      ["bad-latency", /non-negative integer latency_ms/],
      ["no-output", /did not write valid JSON/],
      ["invalid-json", /did not write valid JSON/],
    ] as const;
    for (const [mode, pattern] of cases) {
      await t.test(mode, async () => {
        await assert.rejects(
          evaluateExamples({
            kind: "candidate",
            modelId: "local-adapter",
            baseModelId: "Qwen/Qwen3.5-2B",
            examples: [
              { input: "first", output: "one" },
              { input: "second", output: "two" },
            ],
            system: "Answer.",
            config: configFor(root, { mode }),
            outputPath: join(root, `${mode}.json`),
          }),
          pattern,
        );
      });
    }
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("JSON-field scoring is deterministic and reports per-field accuracy", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-evaluation-json-"));
  try {
    await writeFakeUv(root);
    const expected = JSON.stringify({
      label: "reply",
      priority: "normal",
      process: true,
    });
    const actual = JSON.stringify({
      label: "reply",
      priority: "low",
      process: true,
    });
    const report = await evaluateExamples({
      kind: "candidate",
      modelId: "local-adapter",
      baseModelId: "Qwen/Qwen3.5-2B",
      examples: [{ input: "classify", output: expected }],
      system: "Return JSON.",
      config: configFor(root, {
        answers: { classify: actual },
        scoring: {
          mode: "json_fields",
          fields: ["label", "priority", "process"],
        },
      }),
      outputPath: join(root, "json-fields.json"),
    });

    assert.equal(report.results[0]?.score, 2 / 3);
    assert.equal(report.results[0]?.passed, false);
    assert.equal(report.results[0]?.scored_by, "json_fields");
    assert.equal(report.json_field_metrics?.valid_json_rate, 1);
    assert.deepEqual(report.json_field_metrics?.field_accuracy.priority, {
      correct: 0,
      total: 1,
      accuracy: 0,
    });

  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("sampling, holdout splitting, and token overlap are reproducible", () => {
  const examples = ["a", "b", "c", "d", "e", "f", "g", "h"];
  const sample = sampleExamples(examples, 3, 42);
  assert.deepEqual(sample, sampleExamples(examples, 3, 42));
  assert.equal(sample.length, 3);
  assert.deepEqual(
    sample.map((item) => examples.indexOf(item)),
    [...sample.map((item) => examples.indexOf(item))].sort((a, b) => a - b),
  );
  assert.deepEqual(sampleExamples(examples, examples.length, 42), examples);

  const split = splitSpecExamples(examples, 42);
  assert.deepEqual(split, splitSpecExamples(examples, 42));
  assert.equal(split.holdout.length, 2);
  assert.equal(split.train.length, 6);
  assert.equal(new Set([...split.train, ...split.holdout]).size, examples.length);

  assert.equal(deriveSampleSeed("run-a"), deriveSampleSeed("run-a"));
  assert.notEqual(deriveSampleSeed("run-a"), deriveSampleSeed("run-b"));
  assert.equal(tokenF1("alpha beta", "alpha gamma"), 0.5);
  assert.equal(tokenF1("", ""), 1);
});

test("evaluateExamples records the deterministic sample seed only when truncated", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-evaluation-sample-"));
  try {
    const examples = Array.from({ length: 6 }, (_, index) => ({
      input: `prompt-${index}`,
      output: `answer-${index}`,
    }));
    const config = localRunnerConfigSchema.parse({
      dryRun: true,
      evaluation: {
        maxExamples: 2,
        sampleSeed: 73,
        baselineCache: false,
      },
    });
    const report = await evaluateExamples({
      kind: "baseline",
      modelId: "Qwen/Qwen3.5-2B",
      examples,
      system: "Answer.",
      config,
      outputPath: join(root, "sampled.json"),
    });
    assert.equal(report.eval_examples_total, 6);
    assert.equal(report.eval_examples_used, 2);
    assert.equal(report.eval_truncated, true);
    assert.equal(report.eval_sample_seed, 73);
    assert.deepEqual(
      report.results.map((result) => result.prompt),
      sampleExamples(examples, 2, 73).map((example) => example.input),
    );

    const all = await evaluateExamples({
      kind: "baseline",
      modelId: "Qwen/Qwen3.5-2B",
      examples: examples.slice(0, 2),
      system: "Answer.",
      config,
      outputPath: join(root, "all.json"),
    });
    assert.equal(all.eval_sample_seed, null);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("baseline cache reuses a report only for identical stable inputs", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-evaluation-cache-"));
  try {
    await writeFakeUv(root);
    const counter = join(root, "count");
    const config = configFor(root, {
      answers: { hello: "hi" },
      counter,
      baselineCache: true,
    });
    const common = {
      kind: "baseline" as const,
      modelId: "Qwen/Qwen3.5-2B",
      baseModelId: "Qwen/Qwen3.5-2B",
      baseModelRevision: QWEN_3_5_2B_REVISION,
      examples: [{ input: "hello", output: "hi" }],
      system: "Answer.",
      config,
    };
    const first = await evaluateExamples({
      ...common,
      outputPath: join(root, "first.json"),
    });
    const second = await evaluateExamples({
      ...common,
      outputPath: join(root, "second.json"),
    });
    assert.equal(first.cached, undefined);
    assert.equal(second.cached, true);
    assert.equal(await readFile(counter, "utf8"), "1");
    if (process.platform !== "win32") {
      const cacheRoot = join(config.storeRoot!, "cache", "baseline-evals");
      const [cacheFile] = await readdir(cacheRoot);
      assert.equal((await stat(config.storeRoot!)).mode & 0o777, 0o700);
      assert.equal((await stat(cacheRoot)).mode & 0o777, 0o700);
      assert.equal((await stat(join(cacheRoot, cacheFile!))).mode & 0o777, 0o600);
    }
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("baseline cache refuses preserved symbolic links before reading", {
  skip: process.platform === "win32",
}, async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-evaluation-cache-symlink-"));
  try {
    await writeFakeUv(root);
    const config = configFor(root, {
      answers: { hello: "hi" },
      baselineCache: true,
    });
    const common = {
      kind: "baseline" as const,
      modelId: "Qwen/Qwen3.5-2B",
      baseModelId: "Qwen/Qwen3.5-2B",
      baseModelRevision: QWEN_3_5_2B_REVISION,
      examples: [{ input: "hello", output: "hi" }],
      system: "Answer.",
      config,
    };
    await evaluateExamples({ ...common, outputPath: join(root, "first.json") });
    const cacheRoot = join(config.storeRoot!, "cache", "baseline-evals");
    const [cacheFile] = await readdir(cacheRoot);
    const cachePath = join(cacheRoot, cacheFile!);
    const externalPath = join(root, "external-cache.json");
    await writeFile(externalPath, "not json");
    await unlink(cachePath);
    await symlink(externalPath, cachePath);

    await assert.rejects(
      evaluateExamples({ ...common, outputPath: join(root, "second.json") }),
      /symbolic link/i,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("baseline cache keys include revision, source fingerprint, inference, and scoring", () => {
  const root = "/tmp/tt-local-cache-key";
  const base = {
    modelId: "Qwen/Qwen3.5-2B",
    baseModelRevision: "revision-a",
    sourceFingerprint: "snapshot-a",
    system: "Answer.",
    examples: [{ input: "hello", output: "hi" }],
    evalExamplesTotal: 1,
    evalSplit: "spec_holdout" as const,
    evalSampleSeed: 123,
    config: localRunnerConfigSchema.parse({}),
    packageVersion: "0.3.0",
  };
  const first = baselineCacheKey(base);
  assert.equal(first, baselineCacheKey(base));
  assert.notEqual(first, baselineCacheKey({ ...base, baseModelRevision: "revision-b" }));
  assert.notEqual(first, baselineCacheKey({ ...base, sourceFingerprint: `${root}-other` }));
  assert.notEqual(first, baselineCacheKey({ ...base, evalExamplesTotal: 2 }));
  assert.notEqual(first, baselineCacheKey({ ...base, evalSampleSeed: 456 }));
  assert.notEqual(first, baselineCacheKey({
    ...base,
    config: localRunnerConfigSchema.parse({
      evaluation: { scoring: { mode: "json_fields", fields: ["label"] } },
    }),
  }));
});
