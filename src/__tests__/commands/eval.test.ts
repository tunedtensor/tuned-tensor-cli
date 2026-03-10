import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { Command } from "commander";
import { writeFileSync, unlinkSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { registerEvalCommand } from "../../commands/eval.js";
import * as runner from "../../eval/runner.js";
import { setJsonMode } from "../../output.js";
import type { EvalSummary } from "../../eval/types.js";

vi.mock("../../eval/runner.js", () => ({
  runEvals: vi.fn(),
}));

vi.mock("ora", () => ({
  default: () => ({
    start: vi.fn().mockReturnThis(),
    stop: vi.fn(),
    succeed: vi.fn(),
    fail: vi.fn(),
    set text(_: string) {},
  }),
}));

const TEST_FILE = resolve("test-eval-spec.json");

const mockSummary: EvalSummary = {
  total: 2,
  passed: 2,
  failed: 0,
  pass_rate: 1,
  model: null,
  results: [
    { input: "Hi", expected: "Hello!", actual: null, passed: true, latency_ms: null, assertions: [] },
    { input: "Bye", expected: "Goodbye!", actual: null, passed: true, latency_ms: null, assertions: [] },
  ],
  spec_validation: {
    valid: true,
    checks: [
      { name: "Has name", passed: true },
      { name: "Has system prompt", passed: true },
      { name: "Has examples", passed: true, message: "2 example(s)" },
    ],
  },
};

function buildProgram() {
  const program = new Command();
  program
    .option("-k, --api-key <key>", "API key")
    .option("-u, --base-url <url>", "Base URL")
    .option("--json", "JSON mode");
  registerEvalCommand(program);
  program.exitOverride();
  return program;
}

beforeEach(() => {
  setJsonMode(false);
  vi.spyOn(console, "log").mockImplementation(() => {});
  vi.mocked(runner.runEvals).mockResolvedValue(mockSummary);

  const spec = {
    name: "Test",
    base_model: "llama3.2",
    system_prompt: "You are helpful.",
    guidelines: ["Be concise"],
    constraints: [],
    examples: [{ input: "Hi", output: "Hello!" }],
  };
  writeFileSync(TEST_FILE, JSON.stringify(spec));
});

afterEach(() => {
  if (existsSync(TEST_FILE)) unlinkSync(TEST_FILE);
});

describe("eval command", () => {
  it("runs offline evals without --model", async () => {
    const program = buildProgram();
    await program.parseAsync([
      "node", "tt", "eval", "--file", "test-eval-spec.json",
    ]);

    expect(runner.runEvals).toHaveBeenCalledWith(
      expect.objectContaining({ name: "Test" }),
      expect.objectContaining({ model: undefined }),
    );
  });

  it("passes model to runner when --model is provided", async () => {
    const program = buildProgram();
    await program.parseAsync([
      "node", "tt", "eval",
      "--file", "test-eval-spec.json",
      "--model", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    ]);

    expect(runner.runEvals).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({ model: "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo" }),
    );
  });

  it("outputs JSON in json mode", async () => {
    setJsonMode(true);
    const spy = vi.spyOn(console, "log");
    const program = buildProgram();
    await program.parseAsync([
      "node", "tt", "eval", "--file", "test-eval-spec.json",
    ]);

    const output = spy.mock.calls[0][0];
    const parsed = JSON.parse(output);
    expect(parsed).toHaveProperty("total");
    expect(parsed).toHaveProperty("pass_rate");
  });
});
