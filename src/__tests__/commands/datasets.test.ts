import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { Command } from "commander";
import { writeFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { registerDatasetsCommands } from "../../commands/datasets.js";
import * as client from "../../client.js";
import { setJsonMode } from "../../output.js";

vi.mock("../../client.js", async (importOriginal) => {
  const actual = await importOriginal<typeof client>();
  return {
    ...actual,
    get: vi.fn(),
    del: vi.fn(),
    upload: vi.fn(),
  };
});

const FAKE_KEY = "tt_" + "a".repeat(48);

function buildProgram() {
  const program = new Command();
  program
    .option("-k, --api-key <key>", "API key")
    .option("-u, --base-url <url>", "Base URL")
    .option("--json", "JSON mode");
  registerDatasetsCommands(program);
  program.exitOverride();
  return program;
}

const mockDataset = {
  id: "ds-12345678-abcd",
  name: "training-data",
  description: null,
  storage_path: "/uploads/training-data.jsonl",
  file_size_bytes: 102400,
  row_count: 500,
  format: "jsonl",
  status: "validated",
  validation_errors: null,
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-01-01T00:00:00Z",
};

beforeEach(() => {
  setJsonMode(false);
  process.env.TUNED_TENSOR_API_KEY = FAKE_KEY;
});

describe("datasets commands", () => {
  describe("datasets list", () => {
    it("fetches and displays datasets", async () => {
      vi.mocked(client.get).mockResolvedValue({
        data: [mockDataset],
        meta: { page: 1, per_page: 20, total: 1 },
      });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "datasets", "list"]);
      expect(client.get).toHaveBeenCalledWith(
        "/datasets",
        { page: "1", per_page: "20" },
        expect.anything(),
      );
    });

    it("outputs JSON in json mode", async () => {
      setJsonMode(true);
      vi.mocked(client.get).mockResolvedValue({
        data: [mockDataset],
        meta: { page: 1, per_page: 20, total: 1 },
      });
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "datasets", "list"]);
      expect(JSON.parse(spy.mock.calls[0][0])).toHaveProperty("data");
    });
  });

  describe("datasets get", () => {
    it("fetches dataset details", async () => {
      vi.mocked(client.get).mockResolvedValue({ data: mockDataset });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "datasets", "get", "ds-1234"]);
      expect(client.get).toHaveBeenCalledWith(
        "/datasets/ds-1234",
        undefined,
        expect.anything(),
      );
    });

    it("displays validation errors when present", async () => {
      const dsWithErrors = {
        ...mockDataset,
        status: "invalid",
        validation_errors: ["Row 3 missing 'output' field"],
      };
      vi.mocked(client.get).mockResolvedValue({ data: dsWithErrors });
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "datasets", "get", "ds-1234"]);
      const allOutput = spy.mock.calls.map((c) => String(c[0])).join("\n");
      expect(allOutput).toContain("Row 3 missing");
    });
  });

  describe("datasets upload", () => {
    const tmpFile = join(tmpdir(), `tt-test-upload-${process.pid}.jsonl`);

    beforeEach(() => {
      writeFileSync(tmpFile, '{"input":"hi","output":"hello"}\n');
    });

    afterEach(() => {
      rmSync(tmpFile, { force: true });
    });

    it("uploads a file", async () => {
      vi.mocked(client.upload).mockResolvedValue({ data: mockDataset });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync([
        "node", "tt", "datasets", "upload", tmpFile,
      ]);
      expect(client.upload).toHaveBeenCalledWith(
        "/datasets",
        tmpFile,
        expect.objectContaining({ name: expect.stringContaining("tt-test-upload") }),
        expect.anything(),
      );
    });

    it("exits when file does not exist", async () => {
      vi.spyOn(console, "error").mockImplementation(() => {});
      vi.spyOn(process, "exit").mockImplementation((code) => {
        throw new Error(`process.exit(${code})`);
      });
      const program = buildProgram();
      await expect(
        program.parseAsync([
          "node", "tt", "datasets", "upload", "/nonexistent/data.jsonl",
        ]),
      ).rejects.toThrow();
    });
  });

  describe("datasets delete", () => {
    it("deletes a dataset", async () => {
      vi.mocked(client.del).mockResolvedValue({ data: null as never });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "datasets", "delete", "ds-1234"]);
      expect(client.del).toHaveBeenCalledWith(
        "/datasets/ds-1234",
        expect.anything(),
      );
    });
  });
});
