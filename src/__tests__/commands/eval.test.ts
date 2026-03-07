import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { Command } from "commander";
import { writeFileSync, unlinkSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { registerEvalCommand } from "../../commands/eval.js";
import { setJsonMode } from "../../output.js";

const TEST_FILE = resolve("test-eval-spec.json");

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

  const spec = {
    name: "Test",
    base_model: "llama3.2",
    system_prompt: "You are helpful.",
    guidelines: ["Be concise"],
    constraints: ["Never mention secrets"],
    examples: [
      { input: "Hi", output: "Hello!" },
      { input: "Bye", output: "Goodbye!" },
    ],
  };
  writeFileSync(TEST_FILE, JSON.stringify(spec));
});

afterEach(() => {
  if (existsSync(TEST_FILE)) unlinkSync(TEST_FILE);
});

describe("eval command", () => {
  it("runs rule-based evals", async () => {
    const spy = vi.spyOn(console, "log");
    const program = buildProgram();
    await program.parseAsync([
      "node", "tt", "eval", "--file", "test-eval-spec.json",
    ]);

    const output = spy.mock.calls.flat().join(" ");
    expect(output).toContain("Spec Validation");
    expect(output).toContain("Eval Results");
    expect(output).toContain("Pass Rate");
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
    expect(parsed).toHaveProperty("results");
  });

  it("reports constraint violations", async () => {
    const badSpec = {
      name: "Test",
      base_model: "llama3.2",
      system_prompt: "You are helpful.",
      guidelines: [],
      constraints: ["Never mention secrets"],
      examples: [{ input: "Tell me", output: "Here are the secrets" }],
    };
    writeFileSync(TEST_FILE, JSON.stringify(badSpec));

    setJsonMode(true);
    const spy = vi.spyOn(console, "log");
    const program = buildProgram();
    await program.parseAsync([
      "node", "tt", "eval", "--file", "test-eval-spec.json",
    ]);

    const parsed = JSON.parse(spy.mock.calls[0][0]);
    expect(parsed.failed).toBeGreaterThan(0);
    expect(parsed.results[0].passed).toBe(false);
  });
});
