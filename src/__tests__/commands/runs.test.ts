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

const mockRun = {
  id: "run-12345678-abcd",
  behavior_spec_id: "spec-12345678",
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
        "node", "tt", "runs", "list", "--spec", "spec-123",
      ]);
      expect(client.get).toHaveBeenCalledWith(
        "/behavior-specs/spec-123/runs",
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
      await program.parseAsync(["node", "tt", "runs", "get", "run-1234"]);
      expect(client.get).toHaveBeenCalledWith(
        "/runs/run-1234",
        undefined,
        expect.anything(),
      );
    });

    it("outputs JSON when json mode is on", async () => {
      setJsonMode(true);
      vi.mocked(client.get).mockResolvedValue({ data: mockRun });
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "runs", "get", "run-1234"]);
      const output = JSON.parse(spy.mock.calls[0][0]);
      expect(output.id).toBe(mockRun.id);
    });
  });

  describe("runs start", () => {
    it("starts a run with default options", async () => {
      vi.mocked(client.post).mockResolvedValue({ data: mockRun });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "runs", "start", "spec-123"]);
      expect(client.post).toHaveBeenCalledWith(
        "/behavior-specs/spec-123/runs",
        {},
        expect.anything(),
      );
    });

    it("passes hyperparameters when provided", async () => {
      vi.mocked(client.post).mockResolvedValue({ data: mockRun });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync([
        "node", "tt", "runs", "start", "spec-123",
        "--epochs", "5", "--lr", "0.0001",
      ]);
      expect(client.post).toHaveBeenCalledWith(
        "/behavior-specs/spec-123/runs",
        { hyperparameters: { n_epochs: 5, learning_rate: 0.0001 } },
        expect.anything(),
      );
    });
  });

  describe("runs cancel", () => {
    it("cancels a run", async () => {
      vi.mocked(client.post).mockResolvedValue({ data: { ...mockRun, status: "cancelled" } });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "runs", "cancel", "run-1234"]);
      expect(client.post).toHaveBeenCalledWith(
        "/runs/run-1234/cancel",
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
        "node", "tt", "runs", "watch", "run-1234", "--interval", "10",
      ]);
      expect(client.get).toHaveBeenCalledWith(
        "/runs/run-1234",
        undefined,
        expect.anything(),
      );
    });
  });
});
