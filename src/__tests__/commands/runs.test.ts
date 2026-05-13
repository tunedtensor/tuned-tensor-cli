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
});
