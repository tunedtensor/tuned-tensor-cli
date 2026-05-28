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
  provider: "hosted",
  provider_model_id: "model-12345678-abcd",
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
  describe("models base", () => {
    it("lists supported base models without an API request", async () => {
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "models", "base"]);

      const allOutput = spy.mock.calls.map((c) => String(c[0])).join("\n");
      expect(allOutput).toContain("Qwen/Qwen3.5-2B");
      expect(allOutput).toContain("google/gemma-4-E2B-it");
      expect(client.get).not.toHaveBeenCalled();
    });

    it("outputs JSON when json mode is on", async () => {
      setJsonMode(true);
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "models", "base"]);

      const parsed = JSON.parse(spy.mock.calls[0][0]);
      expect(parsed.data).toContainEqual({
        id: "Qwen/Qwen3.5-2B",
        name: "Qwen3.5-2B",
        type: "base",
      });
      expect(client.get).not.toHaveBeenCalled();
    });
  });

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
      expect(allOutput).toContain("hosted");
      expect(allOutput).not.toMatch(/sagemaker|s3:\/\//i);
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

    it("renders an interactive progress bar with ETA", async () => {
      const outputPath = join(tmpdir(), `tt-model-download-progress-${process.pid}.tar.gz`);
      rmSync(outputPath, { force: true });
      const originalIsTTY = process.stderr.isTTY;
      const originalColumns = process.stderr.columns;
      Object.defineProperty(process.stderr, "isTTY", { value: true, configurable: true });
      Object.defineProperty(process.stderr, "columns", { value: 160, configurable: true });
      const writeSpy = vi.spyOn(process.stderr, "write").mockImplementation(() => true);

      try {
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

        const progressOutput = writeSpy.mock.calls.map((call) => String(call[0])).join("");
        expect(progressOutput).toContain("Downloading [");
        expect(progressOutput).toContain("100.0%");
        expect(progressOutput).toContain("ETA");
      } finally {
        writeSpy.mockRestore();
        Object.defineProperty(process.stderr, "isTTY", {
          value: originalIsTTY,
          configurable: true,
        });
        Object.defineProperty(process.stderr, "columns", {
          value: originalColumns,
          configurable: true,
        });
        rmSync(outputPath, { force: true });
      }
    });

    it("does not render progress in JSON mode", async () => {
      const outputPath = join(tmpdir(), `tt-model-download-json-${process.pid}.tar.gz`);
      rmSync(outputPath, { force: true });
      const originalIsTTY = process.stderr.isTTY;
      Object.defineProperty(process.stderr, "isTTY", { value: true, configurable: true });
      const writeSpy = vi.spyOn(process.stderr, "write").mockImplementation(() => true);

      try {
        setJsonMode(true);
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

        expect(writeSpy).not.toHaveBeenCalled();
      } finally {
        writeSpy.mockRestore();
        Object.defineProperty(process.stderr, "isTTY", {
          value: originalIsTTY,
          configurable: true,
        });
        rmSync(outputPath, { force: true });
      }
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
