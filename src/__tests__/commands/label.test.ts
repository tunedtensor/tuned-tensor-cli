import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { Command } from "commander";
import { writeFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import {
  parseCsvHeader,
  registerLabelCommands,
  validateUnlabeledFile,
} from "../../commands/label.js";
import * as client from "../../client.js";
import { setJsonMode } from "../../output.js";

vi.mock("../../client.js", async (importOriginal) => {
  const actual = await importOriginal<typeof client>();
  return {
    ...actual,
    get: vi.fn(),
    post: vi.fn(),
    patch: vi.fn(),
  };
});

const FAKE_KEY = "tt_" + "a".repeat(48);
const JOB_UUID = "55555555-5555-4555-8555-555555555555";
const SPEC_UUID = "66666666-6666-4666-8666-666666666666";
const ROW_UUID = "77777777-7777-4777-8777-777777777777";

function buildProgram() {
  const program = new Command();
  program
    .option("-k, --api-key <key>", "API key")
    .option("-u, --base-url <url>", "Base URL")
    .option("--json", "JSON mode");
  registerLabelCommands(program);
  program.exitOverride();
  return program;
}

const mockJob = {
  id: JOB_UUID,
  name: "support-tickets",
  behavior_spec_id: SPEC_UUID,
  teacher_model: "openai/gpt-5-mini",
  source_format: "jsonl",
  row_count: 3,
  labeled_count: 0,
  failed_count: 0,
  status: "labeling",
  est_cost_cents: 12,
  actual_cost_cents: null,
  promoted_dataset_id: null,
  error: null,
  created_at: "2026-06-01T00:00:00Z",
  updated_at: "2026-06-01T00:00:00Z",
};

const doneCounts = {
  total: 3,
  pending: 0,
  labeled: 3,
  accepted: 0,
  edited: 0,
  rejected: 0,
  failed: 0,
};

beforeEach(() => {
  vi.clearAllMocks();
  vi.unstubAllGlobals();
  setJsonMode(false);
  process.env.TUNED_TENSOR_API_KEY = FAKE_KEY;
  vi.spyOn(console, "log").mockImplementation(() => {});
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("validateUnlabeledFile", () => {
  const tmpFile = join(tmpdir(), `tt-label-test-${process.pid}.jsonl`);
  const tmpCsv = join(tmpdir(), `tt-label-test-${process.pid}.csv`);

  afterEach(() => {
    rmSync(tmpFile, { force: true });
    rmSync(tmpCsv, { force: true });
  });

  it("accepts input-only JSONL", () => {
    writeFileSync(tmpFile, '{"input":"a"}\n{"input":"b","output":"x"}\n');
    expect(validateUnlabeledFile(tmpFile)).toEqual({ format: "jsonl" });
  });

  it("rejects JSONL rows without input", () => {
    writeFileSync(tmpFile, '{"text":"a"}\n');
    expect(() => validateUnlabeledFile(tmpFile)).toThrow(/string "input"/);
  });

  it("rejects invalid JSON with row numbers", () => {
    writeFileSync(tmpFile, '{"input":"a"}\nnot json\n');
    expect(() => validateUnlabeledFile(tmpFile)).toThrow(/Row 2: invalid JSON/);
  });

  it("rejects empty files", () => {
    writeFileSync(tmpFile, "\n\n");
    expect(() => validateUnlabeledFile(tmpFile)).toThrow(/no JSONL rows/);
  });

  it("validates CSV column presence", () => {
    writeFileSync(tmpCsv, "id,text\n1,hello\n");
    expect(validateUnlabeledFile(tmpCsv, "text")).toEqual({ format: "csv" });
    expect(() => validateUnlabeledFile(tmpCsv, "body")).toThrow(
      /"body" not found.*id, text/,
    );
  });

  it("requires --input-column for CSV", () => {
    writeFileSync(tmpCsv, "id,text\n1,hello\n");
    expect(() => validateUnlabeledFile(tmpCsv)).toThrow(/--input-column/);
  });
});

describe("parseCsvHeader", () => {
  it("parses plain headers", () => {
    expect(parseCsvHeader("id,text,category")).toEqual(["id", "text", "category"]);
  });

  it("honours quoted fields with commas", () => {
    expect(parseCsvHeader('id,"text, full",other')).toEqual([
      "id",
      "text, full",
      "other",
    ]);
  });
});

describe("label upload", () => {
  const tmpFile = join(tmpdir(), `tt-label-upload-${process.pid}.jsonl`);

  afterEach(() => {
    rmSync(tmpFile, { force: true });
  });

  it("presigns, PUTs, and creates the job", async () => {
    writeFileSync(tmpFile, '{"input":"a"}\n');
    vi.mocked(client.post)
      .mockResolvedValueOnce({
        data: {
          path: "s3://bucket/users/u/labeling/x.jsonl",
          upload_url: "https://s3/upload",
          method: "PUT",
          source_format: "jsonl",
        },
      })
      .mockResolvedValueOnce({ data: { ...mockJob, teacher_row_count: 1 } });
    const fetchMock = vi.fn().mockResolvedValue({ ok: true });
    vi.stubGlobal("fetch", fetchMock);

    const program = buildProgram();
    await program.parseAsync([
      "node", "tt", "label", "upload", tmpFile, "--spec", SPEC_UUID,
    ]);

    expect(client.post).toHaveBeenNthCalledWith(
      1,
      "/labeling-jobs/upload-url",
      expect.objectContaining({ contentType: "application/jsonl" }),
      expect.anything(),
    );
    expect(fetchMock).toHaveBeenCalledWith(
      "https://s3/upload",
      expect.objectContaining({ method: "PUT" }),
    );
    expect(client.post).toHaveBeenNthCalledWith(
      2,
      "/labeling-jobs",
      expect.objectContaining({
        path: "s3://bucket/users/u/labeling/x.jsonl",
        behavior_spec_id: SPEC_UUID,
        source_format: "jsonl",
      }),
      expect.anything(),
    );
  });
});

describe("label run", () => {
  it("loops until awaiting_review", async () => {
    vi.mocked(client.post)
      .mockResolvedValueOnce({
        data: {
          locked: false,
          status: "labeling",
          counts: { ...doneCounts, pending: 2, labeled: 1 },
          batch_labeled: 1,
          actual_cost_cents: null,
        },
      })
      .mockResolvedValueOnce({
        data: {
          locked: false,
          status: "awaiting_review",
          counts: doneCounts,
          batch_labeled: 2,
          actual_cost_cents: 9,
        },
      });
    vi.mocked(client.get).mockResolvedValue({
      data: { ...mockJob, status: "awaiting_review", labeled_count: 3, actual_cost_cents: 9 },
    });

    const program = buildProgram();
    await program.parseAsync(["node", "tt", "label", "run", JOB_UUID]);

    expect(client.post).toHaveBeenCalledTimes(2);
    expect(client.post).toHaveBeenCalledWith(
      `/labeling-jobs/${JOB_UUID}/process`,
      {},
      expect.anything(),
    );
    expect(client.get).toHaveBeenCalledWith(
      `/labeling-jobs/${JOB_UUID}`,
      undefined,
      expect.anything(),
    );
  });
});

describe("label accept", () => {
  it("bulk accepts with --all", async () => {
    vi.mocked(client.post).mockResolvedValue({ data: { accepted: 3 } });

    const program = buildProgram();
    await program.parseAsync(["node", "tt", "label", "accept", JOB_UUID, "--all"]);

    expect(client.post).toHaveBeenCalledWith(
      `/labeling-jobs/${JOB_UUID}/rows/bulk`,
      { action: "accept", all_labeled: true },
      expect.anything(),
    );
  });

  it("accepts specific rows by index", async () => {
    vi.mocked(client.get).mockResolvedValue({
      data: [
        { id: ROW_UUID, row_index: 1, input: "a", teacher_output: "x", final_output: null, status: "labeled", error: null },
      ],
    });
    vi.mocked(client.patch).mockResolvedValue({ data: {} });

    const program = buildProgram();
    await program.parseAsync([
      "node", "tt", "label", "accept", JOB_UUID, "--rows", "1",
    ]);

    expect(client.get).toHaveBeenCalledWith(
      `/labeling-jobs/${JOB_UUID}/rows`,
      { page: 1, per_page: 100 },
      expect.anything(),
    );
    expect(client.patch).toHaveBeenCalledWith(
      `/labeling-jobs/${JOB_UUID}/rows/${ROW_UUID}`,
      { action: "accept" },
      expect.anything(),
    );
  });
});

describe("label edit", () => {
  it("sends the replacement output", async () => {
    vi.mocked(client.get).mockResolvedValue({
      data: [
        { id: ROW_UUID, row_index: 0, input: "a", teacher_output: "x", final_output: null, status: "labeled", error: null },
      ],
    });
    vi.mocked(client.patch).mockResolvedValue({ data: {} });

    const program = buildProgram();
    await program.parseAsync([
      "node", "tt", "label", "edit", JOB_UUID, "--row", "0", "--output", "better",
    ]);

    expect(client.patch).toHaveBeenCalledWith(
      `/labeling-jobs/${JOB_UUID}/rows/${ROW_UUID}`,
      { action: "edit", output: "better" },
      expect.anything(),
    );
  });
});

describe("label promote", () => {
  it("promotes with the dataset name", async () => {
    vi.mocked(client.post).mockResolvedValue({
      data: { id: ROW_UUID, name: "tickets-labeled", row_count: 3 },
    });

    const program = buildProgram();
    await program.parseAsync([
      "node", "tt", "label", "promote", JOB_UUID, "--name", "tickets-labeled",
      "--include-unreviewed",
    ]);

    expect(client.post).toHaveBeenCalledWith(
      `/labeling-jobs/${JOB_UUID}/promote`,
      {
        name: "tickets-labeled",
        description: null,
        include_unreviewed_labeled: true,
      },
      expect.anything(),
    );
  });
});

describe("label list", () => {
  it("outputs JSON in json mode", async () => {
    setJsonMode(true);
    vi.mocked(client.get).mockResolvedValue({
      data: [mockJob],
      meta: { page: 1, per_page: 20, total: 1 },
    });
    const spy = vi.spyOn(console, "log").mockImplementation(() => {});
    const program = buildProgram();
    await program.parseAsync(["node", "tt", "label", "list"]);
    expect(JSON.parse(spy.mock.calls[0][0])).toHaveProperty("data");
  });
});
