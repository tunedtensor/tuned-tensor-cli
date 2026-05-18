import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { Command } from "commander";
import { writeFileSync, unlinkSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { registerEvalCommand } from "../../commands/eval.js";
import { setJsonMode } from "../../output.js";

const TEST_FILE = resolve("test-eval-spec.json");

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
  process.exitCode = 0;
  vi.spyOn(console, "log").mockImplementation(() => {});
  vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(() => {
  if (existsSync(TEST_FILE)) unlinkSync(TEST_FILE);
  vi.restoreAllMocks();
  process.exitCode = 0;
});

describe("eval command", () => {
  it("validates a local spec without requiring a model", async () => {
    writeFileSync(TEST_FILE, JSON.stringify({
      name: "Test",
      base_model: "Qwen/Qwen3.5-2B",
      system_prompt: "You are helpful.",
      guidelines: ["Be concise"],
      constraints: [],
      examples: [{ input: "Hi", output: "Hello!" }],
    }));

    const program = buildProgram();
    await program.parseAsync(["node", "tt", "eval", "--file", "test-eval-spec.json"]);

    expect(process.exitCode).toBe(0);
    expect(console.log).toHaveBeenCalledWith(expect.stringContaining("Spec Validation"));
  });

  it("accepts but ignores the deprecated model option", async () => {
    writeFileSync(TEST_FILE, JSON.stringify({
      name: "Test",
      base_model: "Qwen/Qwen3.5-2B",
      system_prompt: "You are helpful.",
      guidelines: ["Be concise"],
      constraints: [],
      examples: [{ input: "Hi", output: "Hello!" }],
    }));

    const program = buildProgram();
    await program.parseAsync([
      "node", "tt", "eval",
      "--file", "test-eval-spec.json",
      "--model", "Qwen/Qwen3.5-2B",
    ]);

    expect(process.exitCode).toBe(0);
    expect(console.log).not.toHaveBeenCalledWith(expect.stringContaining("Qwen/Qwen3.5-2B"));
  });

  it("sets a non-zero exit code when validation fails", async () => {
    writeFileSync(TEST_FILE, JSON.stringify({
      name: "",
      base_model: "",
      system_prompt: "",
      guidelines: [],
      constraints: [],
      examples: [],
    }));

    const program = buildProgram();
    await program.parseAsync(["node", "tt", "eval", "--file", "test-eval-spec.json"]);

    expect(process.exitCode).toBe(1);
  });

  it("exits 0 for default init scaffold (empty guidelines are a warning only)", async () => {
    writeFileSync(TEST_FILE, JSON.stringify({
      name: "Workflow Test",
      description: "",
      base_model: "Qwen/Qwen3.5-2B",
      system_prompt: "You are a helpful assistant.",
      guidelines: [],
      constraints: [],
      examples: [{ input: "Hello", output: "Hi! How can I help you today?" }],
      eval_cases: [],
    }));

    const program = buildProgram();
    await program.parseAsync(["node", "tt", "eval", "--file", "test-eval-spec.json"]);

    expect(process.exitCode).toBe(0);
  });

  it("outputs validation JSON in json mode", async () => {
    setJsonMode(true);
    const spy = vi.spyOn(console, "log");
    writeFileSync(TEST_FILE, JSON.stringify({
      name: "Test",
      base_model: "Qwen/Qwen3.5-2B",
      system_prompt: "You are helpful.",
      guidelines: ["Be concise"],
      constraints: [],
      examples: [{ input: "Hi", output: "Hello!" }],
    }));

    const program = buildProgram();
    await program.parseAsync(["node", "tt", "eval", "--file", "test-eval-spec.json"]);

    const parsed = JSON.parse(spy.mock.calls[0][0]);
    expect(parsed).toHaveProperty("valid", true);
    expect(parsed.checks).toEqual(
      expect.arrayContaining([expect.objectContaining({ name: "Has name", passed: true })]),
    );
  });
});
