import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { Command } from "commander";
import { existsSync, readFileSync, unlinkSync } from "node:fs";
import { resolve } from "node:path";
import { registerInitCommand, loadSpec } from "../../commands/init.js";

const TEST_FILE = resolve("test-init-spec.json");

function buildProgram() {
  const program = new Command();
  program.option("--json", "JSON mode");
  registerInitCommand(program);
  program.exitOverride();
  return program;
}

beforeEach(() => {
  vi.spyOn(console, "log").mockImplementation(() => {});
});

afterEach(() => {
  if (existsSync(TEST_FILE)) unlinkSync(TEST_FILE);
});

describe("init command", () => {
  it("creates a spec file", async () => {
    const program = buildProgram();
    await program.parseAsync([
      "node", "tt", "init", "--file", "test-init-spec.json",
    ]);

    expect(existsSync(TEST_FILE)).toBe(true);
    const content = JSON.parse(readFileSync(TEST_FILE, "utf-8"));
    expect(content.name).toBe("My Agent");
    expect(content.base_model).toBeDefined();
    expect(content.system_prompt).toBeDefined();
    expect(content.examples).toHaveLength(1);
  });

  it("uses custom name and model", async () => {
    const program = buildProgram();
    await program.parseAsync([
      "node", "tt", "init",
      "--file", "test-init-spec.json",
      "--name", "My Bot",
      "--model", "gpt-4o",
    ]);

    const content = JSON.parse(readFileSync(TEST_FILE, "utf-8"));
    expect(content.name).toBe("My Bot");
    expect(content.base_model).toBe("gpt-4o");
  });

  it("warns if file already exists", async () => {
    const program = buildProgram();
    await program.parseAsync([
      "node", "tt", "init", "--file", "test-init-spec.json",
    ]);

    const warnSpy = vi.spyOn(console, "log");
    const program2 = buildProgram();
    await program2.parseAsync([
      "node", "tt", "init", "--file", "test-init-spec.json",
    ]);

    const output = warnSpy.mock.calls.flat().join(" ");
    expect(output).toContain("already exists");
  });
});

describe("loadSpec", () => {
  it("loads a valid spec file", async () => {
    const program = buildProgram();
    await program.parseAsync([
      "node", "tt", "init", "--file", "test-init-spec.json",
    ]);

    const spec = loadSpec("test-init-spec.json");
    expect(spec.name).toBe("My Agent");
    expect(spec.examples).toHaveLength(1);
  });

  it("exits when file does not exist", () => {
    const exitSpy = vi.spyOn(process, "exit").mockImplementation((() => {
      throw new Error("exit");
    }) as never);
    vi.spyOn(console, "error").mockImplementation(() => {});

    expect(() => loadSpec("nonexistent.json")).toThrow("exit");
    expect(exitSpy).toHaveBeenCalledWith(1);
  });
});
