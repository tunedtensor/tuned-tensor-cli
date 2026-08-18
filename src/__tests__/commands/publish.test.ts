import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { Command } from "commander";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { registerPublishCommand } from "../../commands/publish.js";
import * as client from "../../client.js";
import { setJsonMode } from "../../output.js";

vi.mock("../../client.js", async (importOriginal) => {
  const actual = await importOriginal<typeof client>();
  return {
    ...actual,
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    del: vi.fn(),
  };
});

const FAKE_KEY = "tt_" + "a".repeat(48);
const TEST_ROOT = join(tmpdir(), `tt-publish-${process.pid}`);
const STORE_ROOT = join(TEST_ROOT, "store");
const SPEC_ID = "11111111-1111-4111-8111-111111111111";
const RUN_ID = "22222222-2222-4222-8222-222222222222";
const OLDER_RUN_ID = "33333333-3333-4333-8333-333333333333";
const ORIGINAL_CWD = process.cwd();

function writeStoreFixture() {
  mkdirSync(join(STORE_ROOT, "specs", SPEC_ID), { recursive: true });
  mkdirSync(join(STORE_ROOT, "runs", RUN_ID), { recursive: true });
  mkdirSync(join(STORE_ROOT, "runs", OLDER_RUN_ID), { recursive: true });

  writeFileSync(
    join(STORE_ROOT, "specs", SPEC_ID, "spec.json"),
    JSON.stringify({
      name: "Publish Bot",
      description: "",
      system_prompt: "Answer briefly.",
      guidelines: [],
      examples: [{ input: "Hi", output: "Hello" }],
      constraints: [],
      base_model: "Qwen/Qwen3.5-2B",
    }),
  );

  const report = {
    run_id: RUN_ID,
    behavior_spec_id: SPEC_ID,
    user_id: "local-user",
    run_number: 2,
    base_model: "Qwen/Qwen3.5-2B",
    fine_tuned_model_id: `local-${RUN_ID}`,
    status: "completed",
    baseline: {
      kind: "baseline",
      model_id: "Qwen/Qwen3.5-2B",
      total: 1,
      eval_examples_total: 1,
      eval_examples_used: 1,
      eval_truncated: false,
      avg_score: 0.5,
      pass_rate: 0.5,
      exact_match_rate: 0.5,
      avg_latency_ms: 10,
      results: [
        {
          prompt: "Hi",
          expected: "Hello",
          actual: "Hey",
          passed: false,
          score: 0.5,
          reasoning: null,
          latency_ms: 10,
        },
      ],
      artifact_uri: "file:///tmp/baseline.json",
      scoring_method: "exact_match",
    },
    candidate: {
      kind: "candidate",
      model_id: `local-${RUN_ID}`,
      total: 1,
      eval_examples_total: 1,
      eval_examples_used: 1,
      eval_truncated: false,
      avg_score: 1,
      pass_rate: 1,
      exact_match_rate: 1,
      avg_latency_ms: 11,
      results: [
        {
          prompt: "Hi",
          expected: "Hello",
          actual: "Hello",
          passed: true,
          score: 1,
          reasoning: null,
          latency_ms: 11,
        },
      ],
      artifact_uri: "file:///tmp/candidate.json",
      scoring_method: "exact_match",
    },
    comparison: {
      avg_score_delta: 0.5,
      pass_rate_delta: 0.5,
      exact_match_rate_delta: 0.5,
      regressions: 0,
      improvements: 1,
      regressed_examples: [],
    },
    training: {
      provider: "local-uv",
      training_job_name: "job",
      metrics: null,
      exit_code: 0,
      log_uri: "file:///tmp/train.log",
    },
    artifact_uris: {
      dataset: "file:///tmp/train.jsonl",
      baseline_eval: "file:///tmp/baseline.json",
      candidate_eval: "file:///tmp/candidate.json",
      report: "file:///tmp/report.json",
    },
    run_metadata: {
      base_model: "Qwen/Qwen3.5-2B",
      fine_tuned_model_id: `local-${RUN_ID}`,
      dataset_prebuilt: false,
      dataset_uri: "file:///tmp/train.jsonl",
      spec_example_count: 1,
      training_example_count: 1,
      eval_examples_total: 1,
      eval_examples_used: 1,
      started_at: "2026-08-18T10:00:00.000Z",
      completed_at: "2026-08-18T10:05:00.000Z",
      elapsed_ms: 300000,
      elapsed_seconds: 300,
    },
    created_at: "2026-08-18T10:05:00.000Z",
  };

  writeFileSync(
    join(STORE_ROOT, "runs", RUN_ID, "state.json"),
    JSON.stringify({
      id: RUN_ID,
      behavior_spec_id: SPEC_ID,
      user_id: "local-user",
      run_number: 2,
      status: "completed",
      current_stage: "completed",
      status_message: "Done",
      artifact_dir: "/tmp",
      base_model: "Qwen/Qwen3.5-2B",
      spec_name: "Publish Bot",
      created_at: "2026-08-18T10:00:00.000Z",
      updated_at: "2026-08-18T10:05:00.000Z",
      completed_at: "2026-08-18T10:05:00.000Z",
    }),
  );
  writeFileSync(
    join(STORE_ROOT, "runs", RUN_ID, "run-report.json"),
    JSON.stringify(report),
  );

  writeFileSync(
    join(STORE_ROOT, "runs", OLDER_RUN_ID, "state.json"),
    JSON.stringify({
      id: OLDER_RUN_ID,
      behavior_spec_id: SPEC_ID,
      user_id: "local-user",
      run_number: 1,
      status: "completed",
      current_stage: "completed",
      status_message: "Done",
      artifact_dir: "/tmp",
      base_model: "Qwen/Qwen3.5-2B",
      spec_name: "Publish Bot",
      created_at: "2026-08-17T10:00:00.000Z",
      updated_at: "2026-08-17T10:05:00.000Z",
      completed_at: "2026-08-17T10:05:00.000Z",
    }),
  );
  writeFileSync(
    join(STORE_ROOT, "runs", OLDER_RUN_ID, "run-report.json"),
    JSON.stringify({
      ...report,
      run_id: OLDER_RUN_ID,
      run_number: 1,
    }),
  );

  writeFileSync(
    join(TEST_ROOT, "local-runner.json"),
    JSON.stringify({ storeRoot: STORE_ROOT }),
  );
}

function buildProgram() {
  const program = new Command();
  program.option("--json", "JSON mode");
  registerPublishCommand(program);
  program.exitOverride();
  return program;
}

beforeEach(() => {
  setJsonMode(false);
  rmSync(TEST_ROOT, { recursive: true, force: true });
  mkdirSync(TEST_ROOT, { recursive: true });
  writeStoreFixture();
  process.chdir(TEST_ROOT);
  process.env.TUNED_TENSOR_API_KEY = FAKE_KEY;
  vi.mocked(client.post).mockReset();
  vi.spyOn(console, "log").mockImplementation(() => {});
});

afterEach(() => {
  setJsonMode(false);
  process.chdir(ORIGINAL_CWD);
  rmSync(TEST_ROOT, { recursive: true, force: true });
  delete process.env.TUNED_TENSOR_API_KEY;
});

describe("publish command", () => {
  it("defaults to the latest completed run and posts the evidence bundle", async () => {
    vi.mocked(client.post).mockResolvedValue({
      data: {
        id: "hosted-run",
        spec_id: "hosted-spec",
        run_number: 1,
        origin: "local",
        source_run_id: RUN_ID,
        source_spec_id: SPEC_ID,
      },
    });

    const program = buildProgram();
    await program.parseAsync(["node", "tt", "publish", "--yes"]);

    expect(client.post).toHaveBeenCalledWith(
      "/publish/runs",
      expect.objectContaining({
        spec: expect.objectContaining({
          id: SPEC_ID,
          name: "Publish Bot",
        }),
        report: expect.objectContaining({
          run_id: RUN_ID,
          status: "completed",
        }),
      }),
      expect.anything(),
    );
  });

  it("supports dry-run without calling the API", async () => {
    const program = buildProgram();
    await program.parseAsync(["node", "tt", "publish", "--dry-run"]);
    expect(client.post).not.toHaveBeenCalled();
  });

  it("publishes a specific run id prefix", async () => {
    vi.mocked(client.post).mockResolvedValue({
      data: {
        id: "hosted-run-old",
        spec_id: "hosted-spec",
        run_number: 2,
        origin: "local",
        source_run_id: OLDER_RUN_ID,
        source_spec_id: SPEC_ID,
      },
    });

    const program = buildProgram();
    await program.parseAsync([
      "node",
      "tt",
      "publish",
      OLDER_RUN_ID.slice(0, 8),
      "--yes",
    ]);

    expect(client.post).toHaveBeenCalledWith(
      "/publish/runs",
      expect.objectContaining({
        report: expect.objectContaining({ run_id: OLDER_RUN_ID }),
      }),
      expect.anything(),
    );
  });
});
