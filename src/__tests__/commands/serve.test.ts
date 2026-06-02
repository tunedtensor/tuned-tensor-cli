import { describe, it, expect, vi, beforeEach } from "vitest";
import { Command } from "commander";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { registerServeCommand } from "../../commands/serve.js";
import * as client from "../../client.js";
import { setJsonMode } from "../../output.js";

vi.mock("../../client.js", async (importOriginal) => {
  const actual = await importOriginal<typeof client>();
  return {
    ...actual,
    get: vi.fn(),
  };
});

const FAKE_KEY = "tt_" + "a".repeat(48);

function buildProgram() {
  const program = new Command();
  program
    .option("-k, --api-key <key>", "API key")
    .option("-u, --base-url <url>", "Base URL")
    .option("--json", "JSON mode");
  registerServeCommand(program);
  program.exitOverride();
  return program;
}

beforeEach(() => {
  vi.restoreAllMocks();
  vi.clearAllMocks();
  setJsonMode(false);
  process.env.TUNED_TENSOR_API_KEY = FAKE_KEY;
});

describe("serve command", () => {
  it("uses the Tuned Tensor reference server with the spec prompt enabled", async () => {
    setJsonMode(true);
    const root = join(tmpdir(), `tt-serve-reference-${process.pid}`);
    const modelDir = join(root, "model");
    const cacheDir = join(root, "cache");
    const specPath = join(root, "tunedtensor.json");
    rmSync(root, { recursive: true, force: true });
    mkdirSync(modelDir, { recursive: true });
    writeFileSync(join(modelDir, "config.json"), "{}");
    writeFileSync(
      specPath,
      JSON.stringify({
        name: "Support Agent",
        base_model: "Qwen/Qwen3.5-2B",
        system_prompt: "You are a careful support agent.",
        guidelines: ["Ask one clarifying question when needed."],
        constraints: ["Do not invent account data."],
        examples: [{ input: "Hi", output: "Hello." }],
      }),
    );
    const spy = vi.spyOn(console, "log").mockImplementation(() => {});

    try {
      const program = buildProgram();
      await program.parseAsync([
        "node",
        "tt",
        "serve",
        modelDir,
        "--spec",
        specPath,
        "--cache-dir",
        cacheDir,
        "--print-command",
      ]);

      const parsed = JSON.parse(spy.mock.calls[0][0]);
      expect(parsed.command).toBe("python3");
      expect(parsed.args[0]).toContain("openai_reference_server.py");
      expect(parsed.env_keys).toContain("TT_MODEL_PATH");
      expect(parsed.env_keys).toContain("TT_SYSTEM_PROMPT");
      expect(parsed.env_keys).toContain("TT_PORT");
      expect(client.get).not.toHaveBeenCalled();
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("uses the same default port as `tt models serve`", async () => {
    setJsonMode(true);
    const root = join(tmpdir(), `tt-serve-port-${process.pid}`);
    const modelDir = join(root, "model");
    const cacheDir = join(root, "cache");
    rmSync(root, { recursive: true, force: true });
    mkdirSync(modelDir, { recursive: true });
    writeFileSync(join(modelDir, "config.json"), "{}");
    const spy = vi.spyOn(console, "log").mockImplementation(() => {});

    try {
      const program = buildProgram();
      await program.parseAsync([
        "node",
        "tt",
        "serve",
        modelDir,
        "--cache-dir",
        cacheDir,
        "--no-spec-prompt",
        "--print-command",
      ]);

      const parsed = JSON.parse(spy.mock.calls[0][0]);
      expect(parsed.env_keys).toContain("TT_PORT");
      expect(parsed.command_line).not.toContain("llama-server");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("passes JSON schema and managed settings through the shared reference-server path", async () => {
    setJsonMode(true);
    const root = join(tmpdir(), `tt-serve-schema-managed-${process.pid}`);
    const modelDir = join(root, "model");
    const cacheDir = join(root, "cache");
    const schemaPath = join(root, "schema.json");
    const logPath = join(root, "serve.jsonl");
    rmSync(root, { recursive: true, force: true });
    mkdirSync(modelDir, { recursive: true });
    writeFileSync(join(modelDir, "config.json"), "{}");
    writeFileSync(
      schemaPath,
      JSON.stringify({
        type: "object",
        required: ["should_process"],
        properties: { should_process: { type: "boolean" } },
      }),
    );
    const spy = vi.spyOn(console, "log").mockImplementation(() => {});

    try {
      const program = buildProgram();
      await program.parseAsync([
        "node",
        "tt",
        "serve",
        modelDir,
        "--cache-dir",
        cacheDir,
        "--json-schema",
        schemaPath,
        "--json-repair-attempts",
        "2",
        "--managed",
        "--idle-timeout",
        "120",
        "--restart-after-requests",
        "7",
        "--gate-field",
        "decision.allowed",
        "--log-file",
        logPath,
        "--print-command",
      ]);

      const parsed = JSON.parse(spy.mock.calls[0][0]);
      expect(parsed.env_keys).toContain("TT_JSON_SCHEMA");
      expect(parsed.env_keys).toContain("TT_JSON_REPAIR_ATTEMPTS");
      expect(parsed.managed).toMatchObject({
        enabled: true,
        host: "127.0.0.1",
        port: 8000,
        idle_timeout_seconds: 120,
        restart_after_requests: 7,
        gate_field: "decision.allowed",
        log_file: logPath,
      });
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});
