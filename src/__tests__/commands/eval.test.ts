import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { Command } from "commander";
import { writeFileSync, unlinkSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { registerEvalCommand } from "../../commands/eval.js";
import * as runner from "../../eval/runner.js";
import * as providers from "../../eval/providers.js";
import { setJsonMode } from "../../output.js";
import type { EvalSummary } from "../../eval/types.js";

vi.mock("../../eval/runner.js", () => ({
  runEvals: vi.fn(),
}));

vi.mock("../../eval/providers.js", () => ({
  checkProviderAvailability: vi.fn(),
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
  passed: 1,
  failed: 1,
  mean_score: 0.75,
  pass_rate: 0.5,
  results: [
    {
      input: "Hi",
      expected: "Hello!",
      actual: null,
      passed: true,
      score: null,
      reasoning: "All checks passed",
      latency_ms: null,
      assertions: [],
    },
    {
      input: "Secret?",
      expected: "No secrets here",
      actual: null,
      passed: false,
      score: null,
      reasoning: "Constraint violation",
      latency_ms: null,
      assertions: [
        { assertion: "constraint: Never mention secrets", passed: false, message: "Contains secrets" },
      ],
    },
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
  program.option("--json", "JSON mode");
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
  it("runs rule-based evals without provider", async () => {
    const program = buildProgram();
    await program.parseAsync([
      "node", "tt", "eval", "--file", "test-eval-spec.json",
    ]);

    expect(runner.runEvals).toHaveBeenCalledWith(
      expect.objectContaining({ name: "Test" }),
      null,
      expect.any(Function),
    );
  });

  it("checks provider availability before running", async () => {
    vi.mocked(providers.checkProviderAvailability).mockResolvedValue({
      available: true,
    });

    const program = buildProgram();
    await program.parseAsync([
      "node", "tt", "eval",
      "--file", "test-eval-spec.json",
      "--provider", "ollama",
      "--model", "llama3.2",
    ]);

    expect(providers.checkProviderAvailability).toHaveBeenCalled();
    expect(runner.runEvals).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({ provider: "ollama", model: "llama3.2" }),
      expect.any(Function),
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
    expect(JSON.parse(output)).toHaveProperty("total");
  });
});
