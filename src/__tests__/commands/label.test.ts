import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { Command } from "commander";
import { mkdirSync, mkdtempSync, readFileSync, symlinkSync, writeFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import {
  buildLabelExport,
  downloadLabelingJob,
  teacherInstructionsMatch,
  LabelingNotStartedError,
  parseCsvHeader,
  startLabelingJob,
  registerLabelCommands,
  validateUnlabeledFile,
  withTransientRetry,
} from "../../commands/label.js";
import { buildSystemMessage, examplesFromChatJsonl } from "../../local-runtime/dataset.js";
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
    expect(validateUnlabeledFile(tmpFile)).toEqual({ format: "jsonl", warnings: [] });
  });

  it("rejects rows the service would reject, before uploading", () => {
    writeFileSync(
      tmpFile,
      '{"input":"  "}\n{"input":"b","output":""}\n{"input":"c","output":3}\n',
    );
    expect(() => validateUnlabeledFile(tmpFile)).toThrow(
      /Row 1: "input" is empty\nRow 2: "output" is empty - omit it[^\n]*\nRow 3: "output" must be a string/,
    );
  });

  it("warns about over-long inputs the service marks failed", () => {
    writeFileSync(tmpFile, `${JSON.stringify({ input: "x".repeat(32_001) })}\n{"input":"ok"}\n`);
    expect(validateUnlabeledFile(tmpFile).warnings).toEqual([
      "Row 1: input exceeds 32,000 characters and will be marked failed",
    ]);
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
    expect(validateUnlabeledFile(tmpCsv, "text")).toEqual({ format: "csv", warnings: [] });
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

describe("label upload (.json)", () => {
  const jsonFile = join(tmpdir(), `tt-label-upload-${process.pid}.json`);
  afterEach(() => rmSync(jsonFile, { force: true }));

  it("uploads JSON Lines content in a .json file under a .jsonl name", async () => {
    writeFileSync(jsonFile, '{"input":"a"}\n');
    vi.mocked(client.post)
      .mockResolvedValueOnce({
        data: { path: "s3://b/x.jsonl", upload_url: "https://s3/upload", method: "PUT", source_format: "jsonl" },
      })
      .mockResolvedValueOnce({ data: { ...mockJob, teacher_row_count: 1 } });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true }));

    await buildProgram().parseAsync(["node", "tt", "label", "upload", jsonFile, "--spec", SPEC_UUID]);

    expect(client.post).toHaveBeenNthCalledWith(
      1,
      "/labeling-jobs/upload-url",
      expect.objectContaining({ filename: `tt-label-upload-${process.pid}.jsonl` }),
      expect.anything(),
    );
  });
});

describe("startLabelingJob", () => {
  const file = join(tmpdir(), `tt-label-start-${process.pid}.jsonl`);
  afterEach(() => rmSync(file, { force: true }));

  it("uploads approved bytes without rereading a changed source file", async () => {
    const fileBytes = Buffer.from('{"input":"approved"}\n');
    writeFileSync(file, '{"input":"unapproved"}\n');
    const upload = vi.fn().mockResolvedValue({ ok: true });
    vi.stubGlobal("fetch", upload);
    vi.mocked(client.post)
      .mockResolvedValueOnce({ data: { path: "s3://b/x.jsonl", upload_url: "https://s3/u", method: "PUT" } })
      .mockResolvedValueOnce({ data: mockJob });
    await startLabelingJob({ file, fileBytes, format: "jsonl", specId: SPEC_UUID }, {});
    expect(await (upload.mock.calls[0][1].body as Blob).text()).toBe(fileBytes.toString());
  });

  it("reports refusals before or at job creation as not started", async () => {
    writeFileSync(file, '{"input":"a"}\n');
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true }));
    vi.mocked(client.post)
      .mockResolvedValueOnce({ data: { path: "s3://b/x.jsonl", upload_url: "https://s3/u", method: "PUT", source_format: "jsonl" } })
      .mockRejectedValueOnce(new client.ApiError(402, "insufficient_credits", "Insufficient credits"));
    await expect(startLabelingJob({ file, format: "jsonl", specId: SPEC_UUID }, {}))
      .rejects.toBeInstanceOf(LabelingNotStartedError);

    vi.mocked(client.post)
      .mockResolvedValueOnce({ data: { path: "s3://b/x.jsonl", upload_url: "https://s3/u", method: "PUT", source_format: "jsonl" } })
      .mockRejectedValueOnce(new TypeError("fetch failed"));
    const lost = startLabelingJob({ file, format: "jsonl", specId: SPEC_UUID }, {});
    await expect(lost).rejects.toThrow("fetch failed");
    await expect(lost).rejects.not.toBeInstanceOf(LabelingNotStartedError);

    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 403, statusText: "Forbidden" }));
    vi.mocked(client.post)
      .mockResolvedValueOnce({ data: { path: "s3://b/x.jsonl", upload_url: "https://s3/u", method: "PUT", source_format: "jsonl" } });
    await expect(startLabelingJob({ file, format: "jsonl", specId: SPEC_UUID }, {}))
      .rejects.toThrow(LabelingNotStartedError);
  });
});

describe("withTransientRetry", () => {
  it("retries network and 5xx failures but not client errors", async () => {
    const flaky = vi.fn()
      .mockRejectedValueOnce(new TypeError("fetch failed"))
      .mockRejectedValueOnce(new client.ApiError(503, "unavailable", "busy"))
      .mockResolvedValueOnce("ok");
    await expect(withTransientRetry(flaky, 5, 1)).resolves.toBe("ok");
    expect(flaky).toHaveBeenCalledTimes(3);

    const denied = vi.fn().mockRejectedValue(new client.ApiError(404, "not_found", "gone"));
    await expect(withTransientRetry(denied, 5, 1)).rejects.toThrow("gone");
    expect(denied).toHaveBeenCalledTimes(1);
  });
});

function labelRow(index: number, fields: Record<string, unknown>) {
  return {
    id: `row-${index}`,
    row_index: index,
    input: `input ${index}`,
    teacher_output: `label ${index}`,
    final_output: null,
    status: "accepted",
    error: null,
    sanitization_status: "clean",
    ...fields,
  };
}

describe("buildLabelExport", () => {
  it("keeps reviewed rows, drops unsafe or ambiguous ones, and holds out a stable split", () => {
    const rows = [
      labelRow(0, {}),
      labelRow(1, { status: "edited", final_output: "fixed 1" }),
      labelRow(2, { status: "labeled" }),
      labelRow(3, { status: "rejected" }),
      labelRow(4, { status: "failed", teacher_output: null }),
      labelRow(5, { sanitization_status: "blocked" }),
      labelRow(6, { teacher_output: "   " }),
      labelRow(7, { input: "Same  Email", teacher_output: "spam" }),
      labelRow(8, { input: "same email", teacher_output: "spam" }),
      labelRow(9, { input: "Conflict", teacher_output: "spam" }),
      labelRow(10, { input: "conflict", status: "edited", final_output: "ham" }),
      ...Array.from({ length: 20 }, (_, offset) => labelRow(11 + offset, {})),
    ];
    const first = buildLabelExport(rows, { includeUnreviewed: false, validationRatio: 0.2 });
    const all = [...first.training, ...first.validation];

    expect(first.stats).toMatchObject({
      rows: 31,
      selected: 23,
      skipped_unreviewed: 1,
      skipped_rejected_or_failed: 2,
      skipped_blocked: 1,
      skipped_empty_output: 1,
      dropped_duplicate_inputs: 1,
      dropped_conflicting_inputs: 2,
    });
    expect(all).toHaveLength(23);
    expect(all).toContainEqual({ input: "input 1", output: "fixed 1" });
    expect(all).toContainEqual({ input: "Same  Email", output: "spam" });
    expect(all.some((row) => /conflict/i.test(row.input))).toBe(false);
    expect(first.validation.length).toBeGreaterThan(0);
    const trainingInputs = new Set(first.training.map((row) => row.input));
    expect(first.validation.some((row) => trainingInputs.has(row.input))).toBe(false);

    const shuffled = buildLabelExport([...rows].reverse(), { includeUnreviewed: false, validationRatio: 0.2 });
    expect(shuffled.validation).toEqual(first.validation);

    const withDrafts = buildLabelExport(rows, { includeUnreviewed: true, validationRatio: 0.2 });
    expect(withDrafts.stats.selected).toBe(24);
  });

  it("always holds out at least one row and needs two usable rows", () => {
    const pair = buildLabelExport([labelRow(0, {}), labelRow(1, {})], { includeUnreviewed: false, validationRatio: 0.05 });
    expect(pair.training).toHaveLength(1);
    expect(pair.validation).toHaveLength(1);
    expect(() => buildLabelExport([labelRow(0, {}), labelRow(1, { status: "labeled" })], {
      includeUnreviewed: false, validationRatio: 0.1,
    })).toThrow(/--include-unreviewed/);
  });
});

describe("label download", () => {
  let dir: string;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "tt-label-export-"));
  });
  afterEach(() => rmSync(dir, { recursive: true, force: true }));

  const spec = {
    name: "triage",
    system_prompt: "Classify the email.",
    guidelines: ["Use Category / Priority / Action."],
    constraints: [],
    examples: [],
  };

  it.each(["directory", "file", "created-during-fetch"])("refuses unsafe output paths: %s", async (kind) => {
    const specPath = join(dir, "tunedtensor.json");
    writeFileSync(specPath, JSON.stringify(spec));
    const outside = mkdtempSync(join(tmpdir(), "tt-label-outside-"));
    const target = join(outside, "training.jsonl");
    writeFileSync(target, "keep me");
    const output = join(dir, "datasets", "support-tickets");
    try {
      if (kind === "directory") symlinkSync(outside, join(dir, "datasets"));
      if (kind === "file") {
        mkdirSync(output, { recursive: true });
        symlinkSync(target, join(output, "training.jsonl"));
      }
      vi.mocked(client.get).mockReset();
      vi.mocked(client.get).mockResolvedValueOnce({ data: { ...mockJob, status: "awaiting_review" } })
        .mockImplementationOnce(async () => {
          if (kind === "created-during-fetch") {
            mkdirSync(output, { recursive: true });
            writeFileSync(join(output, "training.jsonl"), "new user data");
          }
          return { data: [labelRow(0, {}), labelRow(1, {})], meta: { total: 2, page: 1, per_page: 100 } };
        });
      await expect(downloadLabelingJob({ jobId: JOB_UUID, specPath, includeUnreviewed: false,
        validationRatio: 0.1, force: kind !== "created-during-fetch" }, {}))
        .rejects.toThrow(/symlinks|already exists/);
      expect(readFileSync(target, "utf8")).toBe("keep me");
      if (kind === "created-during-fetch") expect(readFileSync(join(output, "training.jsonl"), "utf8")).toBe("new user data");
    } finally { rmSync(outside, { recursive: true, force: true }); }
  });

  it("writes chat JSONL under the local spec's system message", async () => {
    const specPath = join(dir, "tunedtensor.json");
    writeFileSync(specPath, JSON.stringify(spec));
    const rows = Array.from({ length: 12 }, (_, index) => labelRow(index, {}));
    vi.mocked(client.get)
      .mockResolvedValueOnce({
        data: { ...mockJob, status: "awaiting_review", row_count: 12, system_message: buildSystemMessage(spec) },
      })
      .mockResolvedValueOnce({ data: rows, meta: { page: 1, per_page: 100, total: 12 } });
    vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    vi.spyOn(console, "error").mockImplementation(() => {});

    await buildProgram().parseAsync(["node", "tt", "label", "export", JOB_UUID, "--spec", specPath]);

    const out = join(dir, "datasets", "support-tickets");
    const training = await examplesFromChatJsonl(join(out, "training.jsonl"));
    const validation = await examplesFromChatJsonl(join(out, "validation.jsonl"));
    expect(training.length + validation.length).toBe(12);
    expect(validation.length).toBeGreaterThan(0);
    const firstRow = JSON.parse(readFileSync(join(out, "training.jsonl"), "utf8").split("\n")[0]!);
    expect(firstRow.messages[0]).toEqual({ role: "system", content: buildSystemMessage(spec) });
    const manifest = JSON.parse(readFileSync(join(out, "export.json"), "utf8"));
    expect(manifest).toMatchObject({ labeling_job_id: JOB_UUID, teacher_instructions_match_spec: true });
    expect(console.error).not.toHaveBeenCalled();
    const printed = vi.mocked(console.log).mock.calls.map((call) => String(call[0])).join("\n");
    expect(printed).toContain('"training": "./datasets/support-tickets/training.jsonl"');

    vi.mocked(client.get).mockClear();
    vi.mocked(client.get)
      .mockResolvedValueOnce({ data: { ...mockJob, status: "awaiting_review", row_count: 12 } });
    await expect(buildProgram().parseAsync(["node", "tt", "label", "download", JOB_UUID, "--spec", specPath]))
      .rejects.toThrow(/already exists.*--force/);
    expect(client.get).not.toHaveBeenCalledWith(`/labeling-jobs/${JOB_UUID}/rows`, expect.anything(), expect.anything());
  });

  it("refuses unfinished jobs and non-adapter specs", async () => {
    const specPath = join(dir, "tunedtensor.json");
    writeFileSync(specPath, JSON.stringify(spec));
    vi.mocked(client.get).mockResolvedValueOnce({ data: { ...mockJob, status: "labeling" } });
    await expect(buildProgram().parseAsync(["node", "tt", "label", "export", JOB_UUID, "--spec", specPath]))
      .rejects.toThrow(/"labeling"/);

    writeFileSync(specPath, JSON.stringify({ ...spec, engine: "decision" }));
    await expect(buildProgram().parseAsync(["node", "tt", "label", "export", JOB_UUID, "--spec", specPath]))
      .rejects.toThrow(/engine "decision"/);
  });
});

describe("teacher instruction matching", () => {
  it("accepts exact instructions and the server's explicit example block only", () => {
    const instructions = "Classify as spam.";
    expect(teacherInstructionsMatch(instructions, instructions)).toBe(true);
    expect(teacherInstructionsMatch(instructions, `${instructions}\n\nLabeled examples. Follow their output format exactly and reply with the output only:\n\nExample 1\nInput:\nx\nOutput:\ny`)).toBe(true);
    expect(teacherInstructionsMatch(instructions, `${instructions} Unless the sender is known.`)).toBe(false);
    expect(teacherInstructionsMatch("", instructions)).toBe(false);
    expect(teacherInstructionsMatch(instructions, undefined)).toBe(false);
    expect(teacherInstructionsMatch("Return a  b", "Return a b")).toBe(false);
  });
});

describe("label watch", () => {
  it("polls the job until awaiting_review without any process calls", async () => {
    vi.useFakeTimers();
    vi.mocked(client.get)
      .mockResolvedValueOnce({
        data: {
          ...mockJob,
          status: "labeling",
          counts: { ...doneCounts, pending: 2, labeled: 1 },
        },
      })
      .mockResolvedValueOnce({
        data: {
          ...mockJob,
          status: "awaiting_review",
          labeled_count: 3,
          actual_cost_cents: 9,
          counts: doneCounts,
        },
      });

    const program = buildProgram();
    const parsePromise = program.parseAsync(["node", "tt", "label", "watch", JOB_UUID]);
    await vi.advanceTimersByTimeAsync(6000);
    await parsePromise;
    vi.useRealTimers();

    expect(client.post).not.toHaveBeenCalled();
    expect(client.get).toHaveBeenCalledTimes(2);
    expect(client.get).toHaveBeenCalledWith(
      `/labeling-jobs/${JOB_UUID}`,
      undefined,
      expect.anything(),
    );
  });

  it("sets a non-zero exit code when labeling fails", async () => {
    vi.mocked(client.get).mockResolvedValue({
      data: {
        ...mockJob,
        status: "failed",
        error: "All rows failed teacher labeling",
        counts: doneCounts,
      },
    });
    vi.spyOn(console, "error").mockImplementation(() => {});

    const program = buildProgram();
    await program.parseAsync(["node", "tt", "label", "watch", JOB_UUID]);

    expect(process.exitCode).toBe(1);
    process.exitCode = 0;
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
