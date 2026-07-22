import { describe, it, expect, vi, beforeEach } from "vitest";
import { Command } from "commander";
import { registerRunsCommands } from "../../commands/runs.js";
import * as client from "../../client.js";
import { setJsonMode } from "../../output.js";

vi.mock("../../client.js", async (importOriginal) => {
  const actual = await importOriginal<typeof client>();
  return {
    ...actual,
    get: vi.fn(),
    post: vi.fn(),
  };
});

vi.mock("ora", () => ({
  default: () => ({
    start: vi.fn().mockReturnThis(),
    stop: vi.fn(),
    text: "",
  }),
}));

const FAKE_KEY = "tt_" + "a".repeat(48);

function buildProgram() {
  const program = new Command();
  program
    .option("-k, --api-key <key>", "API key")
    .option("-u, --base-url <url>", "Base URL")
    .option("--json", "JSON mode");
  registerRunsCommands(program);
  program.exitOverride();
  return program;
}

const SPEC_UUID = "11111111-1111-4111-8111-111111111111";
const RUN_UUID = "33333333-3333-4333-8333-333333333333";
const DATASET_UUID = "00000000-0000-4000-8000-000000000123";
const MODEL_UUID = "22222222-2222-4222-8222-222222222222";

const mockRun = {
  id: RUN_UUID,
  behavior_spec_id: SPEC_UUID,
  run_number: 1,
  status: "completed",
  hyperparameters: { n_epochs: 3 },
  eval_summary: { mean_score: 0.85, pass_rate: 0.9 },
  error: null,
  started_at: "2024-01-01T00:00:00Z",
  completed_at: "2024-01-01T01:00:00Z",
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-01-01T01:00:00Z",
  _spec_name: "My Spec",
  _evals: [],
};

const mockEstimate = {
  estimated_training_tokens: 120_000,
  estimated_cost_cents: 22,
  estimated_epochs: 4,
  billing: {
    plan: "free",
    free_run_eligible: true,
    free_run_ineligibility: [],
    free_runs_used: 0,
    free_runs_remaining: 1,
    free_runs_monthly_limit: 1,
    billing_source: "free_quota",
  },
  duration: {
    estimated_minutes: 58,
    range_minutes: { low: 42, high: 78 },
    confidence: "medium",
    sample_count: 12,
    basis: "matched_model",
  },
};

const mockReport = {
  run_id: RUN_UUID,
  status: "completed",
  fine_tuned_model_id: MODEL_UUID,
  baseline: {
    total: 2,
    eval_examples_used: 2,
    avg_score: 0.6,
    pass_rate: 0.5,
    results: [
      {
        prompt: "Label this email.\n\nSubject: Legal notice\nBody: We will file tomorrow unless you pay this invoice.",
        expected: "{\"triage\":\"escalate\",\"priority\":\"high\",\"should_process\":true}",
        actual: "{\"triage\":\"escalate\",\"priority\":\"high\",\"should_process\":true}",
        passed: true,
        score: 1,
        reasoning: "Base escalated the legal threat correctly.",
      },
      {
        prompt: "Label this email.\n\nSubject: FYI office snacks\nBody: Donuts are in the kitchen.",
        expected: "{\"triage\":\"archive\",\"priority\":\"low\",\"should_process\":false}",
        actual: "{\"triage\":\"archive\",\"priority\":\"low\",\"should_process\":false}",
        passed: true,
        score: 1,
      },
    ],
  },
  candidate: {
    total: 2,
    eval_examples_used: 2,
    avg_score: 0.4,
    pass_rate: 0,
    results: [
      {
        prompt: "Label this email.\n\nSubject: Legal notice\nBody: We will file tomorrow unless you pay this invoice.",
        expected: "{\"triage\":\"escalate\",\"priority\":\"high\",\"should_process\":true}",
        actual: "{\"triage\":\"reply\",\"priority\":\"normal\",\"should_process\":true}",
        passed: false,
        score: 0,
        reasoning: "The tuned output under-escalated a legal threat.",
      },
      {
        prompt: "Label this email.\n\nSubject: FYI office snacks\nBody: Donuts are in the kitchen.",
        expected: "{\"triage\":\"archive\",\"priority\":\"low\",\"should_process\":false}",
        actual: "{\"triage\":\"review\",\"priority\":\"normal\",\"should_process\":true}",
        passed: false,
        score: 0.2,
        reasoning: "The tuned output over-processed a low-value FYI.",
      },
    ],
  },
  comparison: {
    avg_score_delta: -0.2,
    pass_rate_delta: -0.5,
    regressions: 1,
    improvements: 0,
    regressed_examples: [
      {
        prompt: "Label this email.\n\nSubject: Legal notice\nBody: We will file tomorrow unless you pay this invoice.",
        old_score: 1,
        new_score: 0,
      },
    ],
  },
};

beforeEach(() => {
  setJsonMode(false);
  process.env.TUNED_TENSOR_API_KEY = FAKE_KEY;
  vi.mocked(client.get).mockReset();
  vi.mocked(client.post).mockReset();
});

describe("runs commands", () => {
  describe("runs list", () => {
    it("fetches all runs", async () => {
      vi.mocked(client.get).mockResolvedValue({
        data: [mockRun],
        meta: { page: 1, per_page: 20, total: 1 },
      });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "runs", "list"]);
      expect(client.get).toHaveBeenCalledWith(
        "/runs",
        expect.objectContaining({ page: "1" }),
        expect.anything(),
      );
    });

    it("filters by spec when --spec is provided", async () => {
      vi.mocked(client.get).mockResolvedValue({
        data: [mockRun],
        meta: { page: 1, per_page: 20, total: 1 },
      });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync([
        "node", "tt", "runs", "list", "--spec", SPEC_UUID,
      ]);
      expect(client.get).toHaveBeenCalledWith(
        `/behavior-specs/${SPEC_UUID}/runs`,
        expect.anything(),
        expect.anything(),
      );
    });

    it("requests compact summaries for all runs", async () => {
      vi.mocked(client.get).mockResolvedValue({
        data: [mockRun],
        meta: { page: 2, per_page: 10, total: 21 },
      });
      vi.spyOn(console, "log").mockImplementation(() => {});

      const program = buildProgram();
      await program.parseAsync([
        "node", "tt", "runs", "list", "--summary", "--page", "2", "--per-page", "10",
      ]);

      expect(client.get).toHaveBeenCalledWith(
        "/runs",
        { page: "2", per_page: "10", view: "summary" },
        expect.anything(),
      );
    });

    it("requests compact summaries for a spec's runs", async () => {
      vi.mocked(client.get).mockResolvedValue({
        data: [mockRun],
        meta: { page: 1, per_page: 20, total: 1 },
      });
      vi.spyOn(console, "log").mockImplementation(() => {});

      const program = buildProgram();
      await program.parseAsync([
        "node", "tt", "runs", "list", "--spec", SPEC_UUID, "--summary",
      ]);

      expect(client.get).toHaveBeenCalledWith(
        `/behavior-specs/${SPEC_UUID}/runs`,
        { page: "1", per_page: "20", view: "summary" },
        expect.anything(),
      );
    });

    it("prints the compact API response unchanged in JSON mode", async () => {
      setJsonMode(true);
      const summaryResponse = {
        data: [{
          id: RUN_UUID,
          behavior_spec_id: SPEC_UUID,
          run_number: 1,
          status: "completed",
          eval_summary: { avg_score: 0.85, pass_rate: 0.9 },
          started_at: "2024-01-01T00:00:00Z",
          completed_at: "2024-01-01T01:00:00Z",
          _spec_name: "My Spec",
        }],
        meta: { page: 1, per_page: 20, total: 1 },
      };
      vi.mocked(client.get).mockResolvedValue(summaryResponse);
      const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});

      const program = buildProgram();
      await program.parseAsync(["node", "tt", "runs", "list", "--summary"]);

      const output = JSON.parse(String(logSpy.mock.calls[0][0]));
      expect(output).toEqual(summaryResponse);
    });
  });

  describe("runs get", () => {
    it("fetches run details", async () => {
      vi.mocked(client.get).mockResolvedValue({ data: mockRun });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "runs", "get", RUN_UUID]);
      expect(client.get).toHaveBeenCalledWith(
        `/runs/${RUN_UUID}`,
        undefined,
        expect.anything(),
      );
    });

    it("displays API avg_score in run details", async () => {
      vi.mocked(client.get).mockResolvedValue({
        data: { ...mockRun, eval_summary: { avg_score: 0.85, pass_rate: 0.9 } },
      });
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "runs", "get", RUN_UUID]);
      expect(spy.mock.calls.flat().join("\n")).toContain("85.0%");
    });

    it("displays output diagnostics when the API provides them", async () => {
      vi.mocked(client.get).mockResolvedValue({
        data: {
          ...mockRun,
          eval_summary: {
            avg_score: 0.08,
            pass_rate: 0.01,
            output_diagnostics: {
              baseline: {
                total: 200,
                valid_json_count: 200,
                valid_json_rate: 1,
                strict_json_count: 200,
                strict_json_rate: 1,
                expected_schema_keys_count: 200,
                expected_schema_keys_rate: 1,
                non_json_prefix_count: 0,
                non_json_prefix_rate: 0,
                visible_reasoning_prefix_count: 0,
                visible_reasoning_prefix_rate: 0,
              },
              candidate: {
                total: 200,
                avg_output_chars: 1096,
                valid_json_count: 0,
                valid_json_rate: 0,
                strict_json_count: 0,
                strict_json_rate: 0,
                expected_schema_keys_count: 0,
                expected_schema_keys_rate: 0,
                non_json_prefix_count: 200,
                non_json_prefix_rate: 1,
                visible_reasoning_prefix_count: 181,
                visible_reasoning_prefix_rate: 0.905,
              },
              insights: [
                "Tuned model output format is the main issue: 100% of primary eval responses did not start with JSON.",
              ],
            },
          },
        },
      });
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "runs", "get", RUN_UUID]);

      const output = spy.mock.calls.flat().join("\n");
      expect(output).toContain("Output Diagnostics");
      expect(output).toContain("Tuned Valid JSON");
      expect(output).toContain("0.0% (0/200)");
      expect(output).toContain("90.5% (181/200)");
      expect(output).toContain("Tuned model output format is the main issue");
    });

    it("outputs JSON when json mode is on", async () => {
      setJsonMode(true);
      vi.mocked(client.get).mockResolvedValue({ data: mockRun });
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "runs", "get", RUN_UUID]);
      const output = JSON.parse(spy.mock.calls[0][0]);
      expect(output.id).toBe(mockRun.id);
    });
  });

  describe("runs estimate", () => {
    it("estimates a run with default options", async () => {
      vi.mocked(client.post).mockResolvedValue({ data: mockEstimate });
      const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "runs", "estimate", SPEC_UUID]);
      expect(client.post).toHaveBeenCalledWith(
        `/behavior-specs/${SPEC_UUID}/runs/estimate`,
        {},
        expect.anything(),
      );
      const output = logSpy.mock.calls.flat().join("\n");
      expect(output).toContain("Plan");
      expect(output).toContain("free");
      expect(output).toContain("Free monthly quota");
      expect(output).toContain("1/1 remaining");
    });

    it("passes the same run configuration options as start", async () => {
      vi.mocked(client.post).mockResolvedValue({ data: mockEstimate });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync([
        "node", "tt", "runs", "estimate", SPEC_UUID,
        "--no-augment",
        "--no-llm-judge",
        "--epochs", "5",
        "--lr", "0.0001",
        "--batch-size", "8",
        "--lora-rank", "16",
        "--lora-alpha", "32",
        "--long-examples", "truncate",
        "--max-seq-length", "4096",
      ]);
      expect(client.post).toHaveBeenCalledWith(
        `/behavior-specs/${SPEC_UUID}/runs/estimate`,
        {
          augment: false,
          use_llm_judge: false,
          hyperparameters: {
            n_epochs: 5,
            learning_rate: 0.0001,
            batch_size: 8,
            lora_rank: 16,
            lora_alpha: 32,
            long_examples: "truncate",
            max_seq_length: 4096,
          },
        },
        expect.anything(),
      );
    });

    it("resolves dataset and spec prefixes before estimating", async () => {
      vi.mocked(client.get)
        .mockResolvedValueOnce({
          data: [{ id: DATASET_UUID, name: "Training data" }],
          meta: { page: 1, per_page: 100, total: 1 },
        })
        .mockResolvedValueOnce({
          data: [{ id: SPEC_UUID, name: "Match" }],
          meta: { page: 1, per_page: 100, total: 1 },
        });
      vi.mocked(client.post).mockResolvedValue({ data: mockEstimate });
      vi.spyOn(console, "log").mockImplementation(() => {});

      const program = buildProgram();
      await program.parseAsync([
        "node", "tt", "runs", "estimate", SPEC_UUID.slice(0, 8),
        "--dataset", DATASET_UUID.slice(0, 8),
        "--train-ratio", "0.7",
        "--validation-ratio", "0.2",
        "--test-ratio", "0.1",
      ]);

      expect(client.post).toHaveBeenCalledWith(
        `/behavior-specs/${SPEC_UUID}/runs/estimate`,
        {
          dataset_id: DATASET_UUID,
          split_ratios: { train: 0.7, validation: 0.2, test: 0.1 },
        },
        expect.anything(),
      );
    });

    it("prints estimate details for humans", async () => {
      vi.mocked(client.post).mockResolvedValue({ data: mockEstimate });
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "runs", "estimate", SPEC_UUID]);
      const output = spy.mock.calls.flat().join("\n");
      expect(output).toContain("Estimated Time");
      expect(output).toContain("58m (42m-1.3h)");
      expect(output).toContain("$0.22");
      expect(output).toContain("medium");
    });

    it("outputs JSON when json mode is on", async () => {
      setJsonMode(true);
      vi.mocked(client.post).mockResolvedValue({ data: mockEstimate });
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "runs", "estimate", SPEC_UUID]);
      const output = JSON.parse(spy.mock.calls[0][0]);
      expect(output.duration.confidence).toBe("medium");
      expect(output.estimated_cost_cents).toBe(22);
    });
  });

  describe("runs start", () => {
    it("starts a run with default options", async () => {
      vi.mocked(client.post).mockResolvedValue({ data: mockRun });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "runs", "start", SPEC_UUID]);
      expect(client.post).toHaveBeenCalledWith(
        `/behavior-specs/${SPEC_UUID}/runs`,
        {},
        expect.anything(),
      );
    });

    it("passes use_llm_judge false at the top level when --no-llm-judge is provided", async () => {
      vi.mocked(client.post).mockResolvedValue({ data: mockRun });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync([
        "node", "tt", "runs", "start", SPEC_UUID, "--no-llm-judge",
      ]);
      expect(client.post).toHaveBeenCalledWith(
        `/behavior-specs/${SPEC_UUID}/runs`,
        { use_llm_judge: false },
        expect.anything(),
      );
    });

    it("passes hyperparameters when provided", async () => {
      vi.mocked(client.post).mockResolvedValue({ data: mockRun });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync([
        "node", "tt", "runs", "start", SPEC_UUID,
        "--epochs", "5", "--lr", "0.0001",
      ]);
      expect(client.post).toHaveBeenCalledWith(
        `/behavior-specs/${SPEC_UUID}/runs`,
        { hyperparameters: { n_epochs: 5, learning_rate: 0.0001 } },
        expect.anything(),
      );
    });

    it("passes eval cap hyperparameters when provided", async () => {
      vi.mocked(client.post).mockResolvedValue({ data: mockRun });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync([
        "node", "tt", "runs", "start", SPEC_UUID,
        "--max-eval-examples", "64",
        "--max-test-eval-examples", "32",
      ]);
      expect(client.post).toHaveBeenCalledWith(
        `/behavior-specs/${SPEC_UUID}/runs`,
        { hyperparameters: { max_eval_examples: 64, max_test_eval_examples: 32 } },
        expect.anything(),
      );
    });

    it("passes long-example controls when provided", async () => {
      vi.mocked(client.post).mockResolvedValue({ data: mockRun });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync([
        "node", "tt", "runs", "start", SPEC_UUID,
        "--long-examples", "truncate",
        "--max-seq-length", "4096",
        "--max-output-tokens", "512",
        "--eval-reserved-output-tokens", "128",
      ]);
      expect(client.post).toHaveBeenCalledWith(
        `/behavior-specs/${SPEC_UUID}/runs`,
        {
          hyperparameters: {
            long_examples: "truncate",
            max_seq_length: 4096,
            max_output_tokens: 512,
            eval_reserved_output_tokens: 128,
          },
        },
        expect.anything(),
      );
    });

    it("rejects unsupported long-example policies", async () => {
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await expect(
        program.parseAsync([
          "node", "tt", "runs", "start", SPEC_UUID,
          "--long-examples", "clip",
        ]),
      ).rejects.toThrow("--long-examples must be one of: error, truncate, skip");
      expect(client.post).not.toHaveBeenCalled();
    });

    it("passes dataset and split ratios when provided", async () => {
      vi.mocked(client.post).mockResolvedValue({ data: mockRun });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync([
        "node", "tt", "runs", "start", SPEC_UUID,
        "--dataset", DATASET_UUID,
        "--train-ratio", "0.7",
        "--validation-ratio", "0.2",
        "--test-ratio", "0.1",
      ]);
      expect(client.post).toHaveBeenCalledWith(
        `/behavior-specs/${SPEC_UUID}/runs`,
        {
          dataset_id: DATASET_UUID,
          split_ratios: { train: 0.7, validation: 0.2, test: 0.1 },
        },
        expect.anything(),
      );
    });

    it("resolves a dataset ID prefix before posting", async () => {
      vi.mocked(client.get).mockResolvedValueOnce({
        data: [{ id: DATASET_UUID, name: "Training data" }],
        meta: { page: 1, per_page: 100, total: 1 },
      });
      vi.mocked(client.post).mockResolvedValue({ data: mockRun });
      vi.spyOn(console, "log").mockImplementation(() => {});

      const program = buildProgram();
      await program.parseAsync([
        "node", "tt", "runs", "start", SPEC_UUID,
        "--dataset", DATASET_UUID.slice(0, 8),
      ]);

      expect(client.get).toHaveBeenCalledWith(
        "/datasets",
        { page: 1, per_page: 100 },
        expect.anything(),
      );
      expect(client.post).toHaveBeenCalledWith(
        `/behavior-specs/${SPEC_UUID}/runs`,
        { dataset_id: DATASET_UUID },
        expect.anything(),
      );
    });

    it("resolves a spec ID prefix before posting", async () => {
      vi.mocked(client.get).mockResolvedValue({
        data: [{ id: SPEC_UUID, name: "Match" }],
        meta: { page: 1, per_page: 100, total: 1 },
      });
      vi.mocked(client.post).mockResolvedValue({ data: mockRun });
      vi.spyOn(console, "log").mockImplementation(() => {});

      const program = buildProgram();
      await program.parseAsync([
        "node", "tt", "runs", "start", SPEC_UUID.slice(0, 8),
      ]);

      expect(client.get).toHaveBeenCalledWith(
        "/behavior-specs",
        { page: 1, per_page: 100 },
        expect.anything(),
      );
      expect(client.post).toHaveBeenCalledWith(
        `/behavior-specs/${SPEC_UUID}/runs`,
        {},
        expect.anything(),
      );
    });

    it("resolves and passes a parent model when provided", async () => {
      vi.mocked(client.get).mockResolvedValueOnce({
        data: [{ id: MODEL_UUID, name: "Tuned model" }],
        meta: { page: 1, per_page: 100, total: 1 },
      });
      vi.mocked(client.post).mockResolvedValue({ data: mockRun });
      vi.spyOn(console, "log").mockImplementation(() => {});

      const program = buildProgram();
      await program.parseAsync([
        "node", "tt", "runs", "start", SPEC_UUID,
        "--parent-model", MODEL_UUID.slice(0, 8),
      ]);

      expect(client.get).toHaveBeenCalledWith(
        "/models",
        { page: 1, per_page: 100 },
        expect.anything(),
      );
      expect(client.post).toHaveBeenCalledWith(
        `/behavior-specs/${SPEC_UUID}/runs`,
        { parent_model_id: MODEL_UUID },
        expect.anything(),
      );
    });
  });

  describe("runs cancel", () => {
    it("cancels a run", async () => {
      vi.mocked(client.post).mockResolvedValue({ data: { ...mockRun, status: "cancelled" } });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "runs", "cancel", RUN_UUID]);
      expect(client.post).toHaveBeenCalledWith(
        `/runs/${RUN_UUID}/cancel`,
        undefined,
        expect.anything(),
      );
    });
  });

  describe("runs watch", () => {
    it("polls until terminal state and displays result", async () => {
      vi.mocked(client.get).mockResolvedValue({
        data: { ...mockRun, status: "completed" },
      });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync([
        "node", "tt", "runs", "watch", RUN_UUID, "--interval", "10",
      ]);
      expect(client.get).toHaveBeenCalledWith(
        `/runs/${RUN_UUID}`,
        undefined,
        expect.anything(),
      );
    });
  });

  describe("runs diagnose", () => {
    it("fetches live diagnostics without exposing backend terms", async () => {
      vi.mocked(client.get).mockResolvedValue({
        data: {
          run_id: RUN_UUID,
          status: "training",
          stage: "training_running",
          stage_label: "Training model",
          progress_pct: 55,
          status_message: "Training is running.",
          summary: "Latest training update arrived 1 minute ago.",
          insights: [
            "Training has reached epoch 0.0810 of 2.00.",
            "Current pace is about 0.0405 epoch every 5 minutes. Estimated training time remaining is about 4.0 hours.",
          ],
          output_diagnostics: {
            baseline: {
              total: 200,
              valid_json_count: 200,
              valid_json_rate: 1,
              strict_json_count: 200,
              strict_json_rate: 1,
              expected_schema_keys_count: 200,
              expected_schema_keys_rate: 1,
              non_json_prefix_count: 0,
              non_json_prefix_rate: 0,
              visible_reasoning_prefix_count: 0,
              visible_reasoning_prefix_rate: 0,
            },
            candidate: {
              total: 200,
              avg_output_chars: 1096,
              valid_json_count: 0,
              valid_json_rate: 0,
              strict_json_count: 0,
              strict_json_rate: 0,
              expected_schema_keys_count: 0,
              expected_schema_keys_rate: 0,
              non_json_prefix_count: 200,
              non_json_prefix_rate: 1,
              visible_reasoning_prefix_count: 181,
              visible_reasoning_prefix_rate: 0.905,
            },
            insights: [
              "Tuned model output format is the main issue: 100% of primary eval responses did not start with JSON.",
            ],
          },
          training: {
            state: "running",
            phase: "Training",
            started_at: "2026-05-28T12:00:00Z",
            completed_at: null,
            last_updated_at: "2026-05-28T12:05:00Z",
            curve: {
              target_epochs: 2,
              latest_epoch: 0.08095,
              latest_loss: 1.9,
              previous_loss: 2.1,
              latest_token_accuracy: 0.5848,
              epoch_rate_per_minute: 0.008094,
              estimated_minutes_remaining: 237.1,
              latest_log_at: "2026-05-28T12:05:00Z",
              samples: [],
            },
          },
          generated_at: "2026-05-28T12:06:00Z",
        },
      });
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "runs", "diagnose", RUN_UUID]);

      expect(client.get).toHaveBeenCalledWith(
        `/runs/${RUN_UUID}/diagnostics`,
        undefined,
        expect.anything(),
      );
      const output = spy.mock.calls.flat().join("\n");
      expect(output).toContain("0.0809 / 2.00");
      expect(output).toContain("58.5%");
      expect(output).toContain("0.0405 epoch / 5m");
      expect(output).toContain("Output Diagnostics");
      expect(output).toContain("90.5% (181/200)");
      expect(output).not.toMatch(/sagemaker|s3:\/\/|aws|ec2/i);
    });
  });

  describe("runs report", () => {
    it("fetches and prints side-by-side regression outputs", async () => {
      vi.mocked(client.get).mockResolvedValue({ data: mockReport });
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "runs", "report", RUN_UUID]);

      expect(client.get).toHaveBeenCalledWith(
        `/runs/${RUN_UUID}/report`,
        undefined,
        expect.anything(),
      );
      const output = spy.mock.calls.flat().join("\n");
      expect(output).toContain("Primary Metrics");
      expect(output).toContain("Primary Regressions");
      expect(output).toContain("score 1.00 -> 0.00");
      expect(output).toContain("Expected:");
      expect(output).toContain('"triage":"escalate"');
      expect(output).toContain("Base:");
      expect(output).toContain("Tuned:");
      expect(output).toContain('"triage":"reply"');
      expect(output).toContain("under-escalated");
    });

    it("can show worst tuned failures instead of regressions", async () => {
      vi.mocked(client.get).mockResolvedValue({ data: mockReport });
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync([
        "node", "tt", "runs", "report", RUN_UUID,
        "--mode", "failures",
        "--limit", "1",
      ]);

      const output = spy.mock.calls.flat().join("\n");
      expect(output).toContain("Primary Tuned Failures");
      expect(output).toContain("tuned score 0.00");
      expect(output).toContain("Legal notice");
      expect(output).not.toContain("FYI office snacks");
    });

    it("outputs the raw report in JSON mode", async () => {
      setJsonMode(true);
      vi.mocked(client.get).mockResolvedValue({ data: mockReport });
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "runs", "report", RUN_UUID]);

      const output = JSON.parse(spy.mock.calls[0][0]);
      expect(output.run_id).toBe(RUN_UUID);
      expect(output.comparison.regressions).toBe(1);
    });
  });
});
