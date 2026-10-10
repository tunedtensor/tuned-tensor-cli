import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../local-runtime/orchestrator.js", async (importOriginal) => ({
  ...await importOriginal<typeof import("../local-runtime/orchestrator.js")>(),
  runLocalPipeline: vi.fn(async (input: { request: { run_id: string } }) => ({
    request: input.request,
    status: "completed",
    outputs: {},
    artifactDir: "/tmp/run",
  })),
}));

const { runLocalPipeline } = await import("../local-runtime/orchestrator.js");
const { createProgram } = await import("../cli.js");
const { setJsonMode } = await import("../output.js");

let root: string;

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), "tt-pipeline-reporting-"));
  setJsonMode(false);
  vi.mocked(runLocalPipeline).mockClear();
  vi.spyOn(console, "log").mockImplementation(() => {});
  vi.spyOn(console, "error").mockImplementation(() => {});
  await writeFile(join(root, "tunedtensor.json"), JSON.stringify({
    id: "22222222-2222-4222-8222-222222222222",
    name: "Reporting",
    system_prompt: "Return a label.",
    guidelines: ["Be brief."],
    base_model: "Qwen/Qwen3.5-2B",
    examples: [{ input: "good", output: "positive" }, { input: "bad", output: "negative" }],
  }));
});

afterEach(async () => {
  vi.restoreAllMocks();
  await rm(root, { recursive: true, force: true });
});

describe("tt pipeline run reporting", () => {
  it("streams stage progress and records the spec file identity", async () => {
    const spec = join(root, "tunedtensor.json");
    await createProgram("test").parseAsync(["pipeline", "run", "--spec", spec], { from: "user" });
    const input = vi.mocked(runLocalPipeline).mock.calls[0]![0];
    expect(input.specFile).toEqual({
      path: spec,
      sha256: createHash("sha256").update(await readFile(spec)).digest("hex"),
    });
    expect(input.reporter?.onEvent).toBeTypeOf("function");
    expect(input.reporter?.verbose).toBe(false);
    expect(vi.mocked(console.error).mock.calls.flat().join("\n")).toMatch(/tt runs audit /);
  });

  it("honors --verbose and --quiet", async () => {
    const spec = join(root, "tunedtensor.json");
    await createProgram("test").parseAsync(["pipeline", "run", "--spec", spec, "--verbose"], { from: "user" });
    expect(vi.mocked(runLocalPipeline).mock.calls[0]![0].reporter?.verbose).toBe(true);
    await createProgram("test").parseAsync(["pipeline", "run", "--spec", spec, "--quiet"], { from: "user" });
    expect(vi.mocked(runLocalPipeline).mock.calls[1]![0].reporter).toBeUndefined();
  });
});
