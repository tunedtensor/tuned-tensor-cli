import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { Command } from "commander";
import { existsSync, readFileSync, unlinkSync } from "node:fs";
import { resolve } from "node:path";
import { registerInitCommand, loadSpec } from "../../commands/init.js";
import { setJsonMode } from "../../output.js";

const TEST_FILE = resolve("test-init-spec.json");

function buildProgram() {
  const program = new Command();
  program.option("--json", "JSON mode");
  registerInitCommand(program);
  program.exitOverride();
  return program;
}

beforeEach(() => {
  setJsonMode(false);
  vi.spyOn(console, "log").mockImplementation(() => {});
});

afterEach(() => {
  setJsonMode(false);
  vi.restoreAllMocks();
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
    expect(content.examples).toHaveLength(2);
    expect(content.examples[0].input).not.toBe(content.examples[1].input);
    expect(content).not.toHaveProperty("eval_cases");
  });

  it("uses custom name and model", async () => {
    const program = buildProgram();
    await program.parseAsync([
      "node", "tt", "init",
      "--file", "test-init-spec.json",
      "--name", "My Bot",
      "--model", "qwen/qwen3.5-2b-base",
    ]);

    const content = JSON.parse(readFileSync(TEST_FILE, "utf-8"));
    expect(content.name).toBe("My Bot");
    expect(content.base_model).toBe("Qwen/Qwen3.5-2B");
  });

  it("creates a foundation spec without a base model", async () => {
    const program = buildProgram();
    await program.parseAsync([
      "node", "tt", "init",
      "--file", "test-init-spec.json",
      "--engine", "foundation",
      "--name", "Tiny GPT",
    ]);

    const content = JSON.parse(readFileSync(TEST_FILE, "utf-8"));
    expect(content.engine).toBe("foundation");
    expect(content.name).toBe("Tiny GPT");
    expect(content.base_model).toBeUndefined();
    expect(content.foundation.depth).toBe(2);
    expect(content.examples).toHaveLength(2);
  });

  it("returns one JSON document when creating a project", async () => {
    setJsonMode(true);
    const program = buildProgram();

    await program.parseAsync([
      "node",
      "tt",
      "init",
      "--file",
      "test-init-spec.json",
    ]);

    expect(console.log).toHaveBeenCalledTimes(1);
    expect(JSON.parse(String(vi.mocked(console.log).mock.calls[0]?.[0])))
      .toMatchObject({
        created: true,
        path: TEST_FILE,
        spec: {
          name: "My Agent",
          base_model: "Qwen/Qwen3.5-2B",
        },
      });
  });

  it("canonicalizes Qwen3-VL aliases", async () => {
    const program = buildProgram();
    await program.parseAsync([
      "node", "tt", "init",
      "--file", "test-init-spec.json",
      "--name", "OCR Bot",
      "--model", "qwen/qwen3-vl-2b",
    ]);

    const content = JSON.parse(readFileSync(TEST_FILE, "utf-8"));
    expect(content.base_model).toBe("Qwen/Qwen3-VL-2B-Instruct");
  });

  it("rejects unsupported models", async () => {
    const program = buildProgram();
    await expect(
      program.parseAsync([
        "node", "tt", "init",
        "--file", "test-init-spec.json",
        "--model", "Qwen/Qwen2.5-1.5B-Instruct",
      ]),
    ).rejects.toThrow(/Unsupported base_model/);

    expect(existsSync(TEST_FILE)).toBe(false);
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

  it("reports an existing project as JSON without human warnings", async () => {
    const program = buildProgram();
    await program.parseAsync([
      "node", "tt", "init", "--file", "test-init-spec.json",
    ]);
    vi.mocked(console.log).mockClear();
    setJsonMode(true);

    const program2 = buildProgram();
    await program2.parseAsync([
      "node", "tt", "init", "--file", "test-init-spec.json",
    ]);

    expect(console.log).toHaveBeenCalledTimes(1);
    expect(JSON.parse(String(vi.mocked(console.log).mock.calls[0]?.[0])))
      .toEqual({
        created: false,
        path: TEST_FILE,
      });
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
    expect(spec.examples).toHaveLength(2);
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
