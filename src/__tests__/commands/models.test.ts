import { describe, it, expect, vi, beforeEach } from "vitest";
import { Command } from "commander";
import { existsSync, rmSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
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
  provider: "sagemaker",
  provider_model_id: "s3://bucket/models/model.tar.gz",
  base_model: "meta-llama/Llama-3.2-3B-Instruct",
  description: null,
  created_at: "2024-01-01T00:00:00Z",
};

beforeEach(() => {
  vi.clearAllMocks();
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
      vi.mocked(client.get)
        .mockResolvedValueOnce({
          data: [mockModel],
          meta: { page: 1, per_page: 100, total: 1 },
        })
        .mockResolvedValueOnce({ data: mockModel });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "models", "get", "model-1234"]);
      expect(client.get).toHaveBeenNthCalledWith(
        1,
        "/models",
        { page: 1, per_page: 100 },
        expect.anything(),
      );
      expect(client.get).toHaveBeenNthCalledWith(
        2,
        "/models/model-12345678-abcd",
        undefined,
        expect.anything(),
      );
    });

    it("displays all fields", async () => {
      vi.mocked(client.get)
        .mockResolvedValueOnce({
          data: [mockModel],
          meta: { page: 1, per_page: 100, total: 1 },
        })
        .mockResolvedValueOnce({ data: mockModel });
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "models", "get", "model-1234"]);
      const allOutput = spy.mock.calls.map((c) => String(c[0])).join("\n");
      expect(allOutput).toContain("my-fine-tuned-model");
      expect(allOutput).toContain("sagemaker");
    });
  });

  describe("models delete", () => {
    it("deletes a model", async () => {
      vi.mocked(client.get).mockResolvedValue({
        data: [mockModel],
        meta: { page: 1, per_page: 100, total: 1 },
      });
      vi.mocked(client.del).mockResolvedValue({ data: null as never });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "models", "delete", "model-1234"]);
      expect(client.get).toHaveBeenCalledWith(
        "/models",
        { page: 1, per_page: 100 },
        expect.anything(),
      );
      expect(client.del).toHaveBeenCalledWith(
        "/models/model-12345678-abcd",
        expect.anything(),
      );
    });
  });

  describe("models download", () => {
    it("downloads a model artifact to the requested path", async () => {
      const outputPath = join(tmpdir(), `tt-model-download-${process.pid}.tar.gz`);
      if (existsSync(outputPath)) rmSync(outputPath);

      vi.mocked(client.get)
        .mockResolvedValueOnce({
          data: [mockModel],
          meta: { page: 1, per_page: 100, total: 1 },
        })
        .mockResolvedValueOnce({
          data: {
            url: "https://signed.example/model.tar.gz",
            filename: "model.tar.gz",
            expires_at: "2026-01-01T00:10:00Z",
          },
        });
      vi.spyOn(globalThis, "fetch").mockResolvedValue(
        new Response("artifact-bytes", {
          status: 200,
          headers: { "content-length": "14" },
        }),
      );
      vi.spyOn(console, "log").mockImplementation(() => {});

      const program = buildProgram();
      await program.parseAsync([
        "node",
        "tt",
        "models",
        "download",
        "model-1234",
        "--output",
        outputPath,
      ]);

      expect(client.get).toHaveBeenNthCalledWith(
        1,
        "/models",
        { page: 1, per_page: 100 },
        expect.anything(),
      );
      expect(client.get).toHaveBeenNthCalledWith(
        2,
        "/models/model-12345678-abcd/download",
        undefined,
        expect.anything(),
      );
      expect(globalThis.fetch).toHaveBeenCalledWith("https://signed.example/model.tar.gz");
      expect(readFileSync(outputPath, "utf8")).toBe("artifact-bytes");

      rmSync(outputPath);
    });

    it("refuses to overwrite without --force", async () => {
      const outputPath = join(tmpdir(), `tt-model-download-existing-${process.pid}.tar.gz`);
      rmSync(outputPath, { force: true });
      writeFileSync(outputPath, "existing");

      vi.mocked(client.get)
        .mockResolvedValueOnce({
          data: [mockModel],
          meta: { page: 1, per_page: 100, total: 1 },
        })
        .mockResolvedValueOnce({
          data: {
            url: "https://signed.example/model.tar.gz",
            filename: "model.tar.gz",
            expires_at: "2026-01-01T00:10:00Z",
          },
        });

      const program = buildProgram();
      await expect(
        program.parseAsync([
          "node",
          "tt",
          "models",
          "download",
          "model-1234",
          "--output",
          outputPath,
        ]),
      ).rejects.toThrow("Output file already exists");

      rmSync(outputPath, { force: true });
    });
  });
});
