import { describe, it, expect, vi, beforeEach } from "vitest";
import { Command } from "commander";
import { registerModelsCommands } from "../../commands/models.js";
import * as client from "../../client.js";
import { setJsonMode } from "../../output.js";

vi.mock("../../client.js", async (importOriginal) => {
  const actual = await importOriginal<typeof client>();
  return {
    ...actual,
    get: vi.fn(),
    del: vi.fn(),
  };
});

const FAKE_KEY = "tt_" + "a".repeat(48);

function buildProgram() {
  const program = new Command();
  program
    .option("-k, --api-key <key>", "API key")
    .option("-u, --base-url <url>", "Base URL")
    .option("--json", "JSON mode");
  registerModelsCommands(program);
  program.exitOverride();
  return program;
}

const mockModel = {
  id: "model-12345678-abcd",
  name: "my-fine-tuned-model",
  provider: "together",
  provider_model_id: "ft:meta-llama/Llama-3.2-3B-Instruct:abc123",
  base_model: "meta-llama/Llama-3.2-3B-Instruct",
  description: null,
  created_at: "2024-01-01T00:00:00Z",
};

beforeEach(() => {
  setJsonMode(false);
  process.env.TUNED_TENSOR_API_KEY = FAKE_KEY;
});

describe("models commands", () => {
  describe("models list", () => {
    it("fetches and displays models", async () => {
      vi.mocked(client.get).mockResolvedValue({
        data: [mockModel],
        meta: { page: 1, per_page: 20, total: 1 },
      });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "models", "list"]);
      expect(client.get).toHaveBeenCalledWith(
        "/models",
        { page: "1", per_page: "20" },
        expect.anything(),
      );
    });

    it("outputs JSON when json mode is on", async () => {
      setJsonMode(true);
      vi.mocked(client.get).mockResolvedValue({
        data: [mockModel],
        meta: { page: 1, per_page: 20, total: 1 },
      });
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "models", "list"]);
      expect(JSON.parse(spy.mock.calls[0][0])).toHaveProperty("data");
    });
  });

  describe("models get", () => {
    it("fetches model details", async () => {
      vi.mocked(client.get).mockResolvedValue({ data: mockModel });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "models", "get", "model-1234"]);
      expect(client.get).toHaveBeenCalledWith(
        "/models/model-1234",
        undefined,
        expect.anything(),
      );
    });

    it("displays all fields", async () => {
      vi.mocked(client.get).mockResolvedValue({ data: mockModel });
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "models", "get", "model-1234"]);
      const allOutput = spy.mock.calls.map((c) => String(c[0])).join("\n");
      expect(allOutput).toContain("my-fine-tuned-model");
      expect(allOutput).toContain("together");
    });
  });

  describe("models delete", () => {
    it("deletes a model", async () => {
      vi.mocked(client.del).mockResolvedValue({ data: null as never });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "models", "delete", "model-1234"]);
      expect(client.del).toHaveBeenCalledWith(
        "/models/model-1234",
        expect.anything(),
      );
    });
  });
});
