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
    post: vi.fn(),
  };
});

const FAKE_KEY = "tt_" + "a".repeat(48);
const DATASET_UUID = "44444444-4444-4444-8444-444444444444";

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
  id: DATASET_UUID,
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
  vi.clearAllMocks();
  vi.unstubAllGlobals();
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
      await program.parseAsync(["node", "tt", "datasets", "get", DATASET_UUID]);
      expect(client.get).toHaveBeenCalledWith(
        `/datasets/${DATASET_UUID}`,
        undefined,
        expect.anything(),
      );
    });

    it("resolves a dataset ID prefix before fetching details", async () => {
      vi.mocked(client.get)
        .mockResolvedValueOnce({
          data: [mockDataset],
          meta: { page: 1, per_page: 100, total: 1 },
        })
        .mockResolvedValueOnce({ data: mockDataset });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync([
        "node", "tt", "datasets", "get", DATASET_UUID.slice(0, 8),
      ]);
      expect(client.get).toHaveBeenNthCalledWith(
        1,
        "/datasets",
        { page: 1, per_page: 100 },
        expect.anything(),
      );
      expect(client.get).toHaveBeenNthCalledWith(
        2,
        `/datasets/${DATASET_UUID}`,
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
      await program.parseAsync(["node", "tt", "datasets", "get", DATASET_UUID]);
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
      vi.mocked(client.post)
        .mockResolvedValueOnce({
          data: {
            path: "s3://tt-runs/users/user-1/datasets/upload.jsonl",
            upload_url: "https://uploads.example.com/upload.jsonl",
            method: "PUT",
            headers: { "Content-Type": "application/jsonl" },
          },
        })
        .mockResolvedValueOnce({ data: mockDataset });
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({ ok: true }),
      );
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync([
        "node", "tt", "datasets", "upload", tmpFile,
      ]);
      expect(client.post).toHaveBeenNthCalledWith(
        1,
        "/datasets/upload-url",
        expect.objectContaining({
          name: expect.stringContaining("tt-test-upload"),
          filename: expect.stringContaining("tt-test-upload"),
          format: "jsonl",
        }),
        expect.anything(),
      );
      expect(fetch).toHaveBeenCalledWith(
        "https://uploads.example.com/upload.jsonl",
        expect.objectContaining({ method: "PUT" }),
      );
      expect(client.post).toHaveBeenNthCalledWith(
        2,
        "/datasets/finalize",
        expect.objectContaining({
          path: "s3://tt-runs/users/user-1/datasets/upload.jsonl",
          name: expect.stringContaining("tt-test-upload"),
        }),
        expect.anything(),
      );
    });

    it("uploads a document OCR JSONL file", async () => {
      writeFileSync(
        tmpFile,
        `${JSON.stringify({
          input: {
            prompt: "Extract invoice fields as JSON.",
            assets: [
              {
                mime_type: "image/png",
                data_uri: "data:image/png;base64,iVBORw0KGgo=",
              },
            ],
          },
          output: "{\"invoice_number\":\"INV-123\"}",
        })}\n`,
      );
      vi.mocked(client.post)
        .mockResolvedValueOnce({
          data: {
            path: "s3://tt-runs/users/user-1/datasets/upload.jsonl",
            upload_url: "https://uploads.example.com/upload.jsonl",
            method: "PUT",
            headers: { "Content-Type": "application/jsonl" },
          },
        })
        .mockResolvedValueOnce({
          data: { ...mockDataset, format: "document_ocr_jsonl", row_count: 1 },
        });
      vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true }));
      vi.spyOn(console, "log").mockImplementation(() => {});

      const program = buildProgram();
      await program.parseAsync([
        "node", "tt", "datasets", "upload", tmpFile, "--format", "document_ocr_jsonl",
      ]);

      expect(client.post).toHaveBeenNthCalledWith(
        1,
        "/datasets/upload-url",
        expect.objectContaining({ format: "document_ocr_jsonl" }),
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

    it("rejects OpenAI messages format before uploading", async () => {
      writeFileSync(
        tmpFile,
        '{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}\n',
      );
      const program = buildProgram();

      await expect(
        program.parseAsync(["node", "tt", "datasets", "upload", tmpFile]),
      ).rejects.toThrow(/OpenAI SFT-style "messages".*flat "input" and "output"/s);
      expect(client.post).not.toHaveBeenCalled();
    });

    it("rejects rows missing input or output before uploading", async () => {
      writeFileSync(tmpFile, '{"input":"hi"}\n{"output":"hello"}\n');
      const program = buildProgram();

      await expect(
        program.parseAsync(["node", "tt", "datasets", "upload", tmpFile]),
      ).rejects.toThrow(/missing string "output".*missing string "input"/s);
      expect(client.post).not.toHaveBeenCalled();
    });

    it("rejects invalid JSONL before uploading", async () => {
      writeFileSync(tmpFile, '{"input":"hi","output":"hello"\n');
      const program = buildProgram();

      await expect(
        program.parseAsync(["node", "tt", "datasets", "upload", tmpFile]),
      ).rejects.toThrow(/invalid JSON/);
      expect(client.post).not.toHaveBeenCalled();
    });

    it("rejects empty files before uploading", async () => {
      writeFileSync(tmpFile, "\n");
      const program = buildProgram();

      await expect(
        program.parseAsync(["node", "tt", "datasets", "upload", tmpFile]),
      ).rejects.toThrow(/no JSONL rows/);
      expect(client.post).not.toHaveBeenCalled();
    });

    it("rejects rows containing invisible NEL (U+0085) before uploading", async () => {
      // NEL inside the input string. JSON.parse accepts it, but Python
      // splitlines() will split the JSONL row in half mid-string downstream.
      const row = JSON.stringify({ input: "subjectbody", output: "ok" });
      writeFileSync(tmpFile, `${row}\n`);
      const program = buildProgram();

      await expect(
        program.parseAsync(["node", "tt", "datasets", "upload", tmpFile]),
      ).rejects.toThrow(/U\+0085.*NEXT LINE/);
      expect(client.post).not.toHaveBeenCalled();
    });

    it("rejects rows containing C0 control chars (BELL) before uploading", async () => {
      const row = JSON.stringify({ input: "ab", output: "ok" });
      writeFileSync(tmpFile, `${row}\n`);
      const program = buildProgram();

      await expect(
        program.parseAsync(["node", "tt", "datasets", "upload", tmpFile]),
      ).rejects.toThrow(/U\+0007.*BELL/);
      expect(client.post).not.toHaveBeenCalled();
    });

    it("rejects rows containing C1 control chars (U+0092) before uploading", async () => {
      const row = JSON.stringify({ input: "smartquote", output: "ok" });
      writeFileSync(tmpFile, `${row}\n`);
      const program = buildProgram();

      await expect(
        program.parseAsync(["node", "tt", "datasets", "upload", tmpFile]),
      ).rejects.toThrow(/U\+0092.*C1 CONTROL CHARACTER/);
      expect(client.post).not.toHaveBeenCalled();
    });

    it("rejects rows containing U+2028 LINE SEPARATOR before uploading", async () => {
      const row = JSON.stringify({ input: "a b", output: "ok" });
      writeFileSync(tmpFile, `${row}\n`);
      const program = buildProgram();

      await expect(
        program.parseAsync(["node", "tt", "datasets", "upload", tmpFile]),
      ).rejects.toThrow(/U\+2028.*LINE SEPARATOR/);
      expect(client.post).not.toHaveBeenCalled();
    });

    it("accepts rows containing tab, newline, and carriage return inside strings", async () => {
      const row = JSON.stringify({ input: "line1\nline2\twith tab\rcr", output: "ok" });
      writeFileSync(tmpFile, `${row}\n`);
      vi.mocked(client.post)
        .mockResolvedValueOnce({
          data: {
            path: "s3://tt-runs/users/user-1/datasets/upload.jsonl",
            upload_url: "https://uploads.example.com/upload.jsonl",
            method: "PUT",
            headers: { "Content-Type": "application/jsonl" },
          },
        })
        .mockResolvedValueOnce({ data: mockDataset });
      vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true }));
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "datasets", "upload", tmpFile]);
      expect(client.post).toHaveBeenCalled();
    });
  });

  describe("datasets delete", () => {
    it("deletes a dataset", async () => {
      vi.mocked(client.del).mockResolvedValue({ data: null as never });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "datasets", "delete", DATASET_UUID]);
      expect(client.del).toHaveBeenCalledWith(
        `/datasets/${DATASET_UUID}`,
        expect.anything(),
      );
    });
  });
});
