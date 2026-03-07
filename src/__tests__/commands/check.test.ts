import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { Command } from "commander";
import { writeFileSync, unlinkSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { registerCheckCommand } from "../../commands/check.js";
import { setJsonMode } from "../../output.js";

const TEST_FILE = resolve("test-check-spec.json");

function buildProgram() {
  const program = new Command();
  program.option("--json", "JSON mode");
  registerCheckCommand(program);
  program.exitOverride();
  return program;
}

beforeEach(() => {
  setJsonMode(false);
  vi.spyOn(console, "log").mockImplementation(() => {});
});

afterEach(() => {
  if (existsSync(TEST_FILE)) unlinkSync(TEST_FILE);
});

describe("check command", () => {
  it("validates a well-formed spec", async () => {
    const spec = {
      name: "Test Bot",
      base_model: "llama3.2",
      system_prompt: "You are helpful.",
      guidelines: ["Be concise"],
      constraints: [],
      examples: [{ input: "Hi", output: "Hello!" }],
    };
    writeFileSync(TEST_FILE, JSON.stringify(spec));

    const spy = vi.spyOn(console, "log");
    const program = buildProgram();
    await program.parseAsync([
      "node", "tt", "check", "--file", "test-check-spec.json",
    ]);

    const output = spy.mock.calls.flat().join(" ");
    expect(output).toContain("Spec Validation");
    expect(output).toContain("valid");
  });

  it("reports issues for an incomplete spec", async () => {
    const spec = {
      name: "",
      base_model: "",
      system_prompt: "",
      guidelines: [],
      constraints: [],
      examples: [],
    };
    writeFileSync(TEST_FILE, JSON.stringify(spec));

    const exitSpy = vi.spyOn(process, "exit").mockImplementation((() => {
      throw new Error("exit");
    }) as never);

    const program = buildProgram();
    await expect(
      program.parseAsync([
        "node", "tt", "check", "--file", "test-check-spec.json",
      ]),
    ).rejects.toThrow("exit");

    expect(exitSpy).toHaveBeenCalledWith(1);
  });

  it("outputs JSON in json mode", async () => {
    setJsonMode(true);
    const spec = {
      name: "Test Bot",
      base_model: "llama3.2",
      system_prompt: "You are helpful.",
      guidelines: ["Be concise"],
      constraints: [],
      examples: [{ input: "Hi", output: "Hello!" }],
    };
    writeFileSync(TEST_FILE, JSON.stringify(spec));

    const spy = vi.spyOn(console, "log");
    const program = buildProgram();
    await program.parseAsync([
      "node", "tt", "check", "--file", "test-check-spec.json",
    ]);

    const output = spy.mock.calls[0][0];
    const parsed = JSON.parse(output);
    expect(parsed).toHaveProperty("valid");
    expect(parsed).toHaveProperty("checks");
  });
});
