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
    set text(_: string) {},
  }),
}));

const TEST_FILE = resolve("test-eval-spec.json");

const mockSummary: EvalSummary = {
  total: 2,
  passed: 2,
  failed: 0,
  pass_rate: 1,
  model: "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
  results: [
    { input: "Hi", expected: "Hello!", actual: "Hello there!", passed: true, latency_ms: 300, assertions: [] },
    { input: "Bye", expected: "Goodbye!", actual: "See you!", passed: true, latency_ms: 250, assertions: [] },
  ],
  spec_validation: {
    valid: true,
    checks: [{ name: "Has name", passed: true }],
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
  vi.spyOn(console, "error").mockImplementation(() => {});
  vi.mocked(runner.runEvals).mockResolvedValue(mockSummary);

  writeFileSync(TEST_FILE, JSON.stringify({
    name: "Test",
    base_model: "llama3.2",
    system_prompt: "You are helpful.",
    guidelines: [],
    constraints: [],
    examples: [{ input: "Hi", output: "Hello!" }],
  }));
});

afterEach(() => {
  if (existsSync(TEST_FILE)) unlinkSync(TEST_FILE);
});

describe("eval command", () => {
  it("requires --model flag", async () => {
    const program = buildProgram();
    await expect(
      program.parseAsync(["node", "tt", "eval", "--file", "test-eval-spec.json"]),
    ).rejects.toThrow();
  });

  it("calls runEvals with model and spec", async () => {
    const program = buildProgram();
    await program.parseAsync([
      "node", "tt", "eval",
      "--file", "test-eval-spec.json",
      "--model", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    ]);

    expect(runner.runEvals).toHaveBeenCalledWith(
      expect.objectContaining({ name: "Test" }),
      "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
      expect.objectContaining({ clientOpts: expect.anything() }),
    );
  });

  it("outputs JSON in json mode", async () => {
    setJsonMode(true);
    const spy = vi.spyOn(console, "log");
    const program = buildProgram();
    await program.parseAsync([
      "node", "tt", "eval",
      "--file", "test-eval-spec.json",
      "--model", "test-model",
    ]);

    const parsed = JSON.parse(spy.mock.calls[0][0]);
    expect(parsed).toHaveProperty("total");
    expect(parsed).toHaveProperty("model", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo");
  });
});
