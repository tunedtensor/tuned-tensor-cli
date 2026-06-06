import { describe, it, expect, vi, beforeEach } from "vitest";
import { Command } from "commander";
import { existsSync, mkdirSync, rmSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { registerModelsCommands } from "../../commands/models.js";
import * as client from "../../client.js";
import { setJsonMode } from "../../output.js";

vi.mock("../../client.js", async (importOriginal) => {
  const actual = await importOriginal<typeof client>();
  return {
    ...actual,
    get: vi.fn(),
    del: vi.fn(),
  };
});

const FAKE_KEY = "tt_" + "a".repeat(48);

function buildProgram() {
  const program = new Command();
  program
    .option("-k, --api-key <key>", "API key")
    .option("-u, --base-url <url>", "Base URL")
    .option("--json", "JSON mode");
  registerModelsCommands(program);
  program.exitOverride();
  return program;
}

const mockModel = {
  id: "model-12345678-abcd",
  name: "my-fine-tuned-model",
  provider: "hosted",
  provider_model_id: "model-12345678-abcd",
  base_model: "meta-llama/Llama-3.2-3B-Instruct",
  description: null,
  created_at: "2024-01-01T00:00:00Z",
};

beforeEach(() => {
  vi.clearAllMocks();
  setJsonMode(false);
  process.env.TUNED_TENSOR_API_KEY = FAKE_KEY;
});

describe("models commands", () => {
  describe("models base", () => {
    it("lists supported base models without an API request", async () => {
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "models", "base"]);

      const allOutput = spy.mock.calls.map((c) => String(c[0])).join("\n");
      expect(allOutput).toContain("Qwen/Qwen3.5-2B");
      expect(allOutput).toContain("google/gemma-4-E2B-it");
      expect(client.get).not.toHaveBeenCalled();
    });

    it("outputs JSON when json mode is on", async () => {
      setJsonMode(true);
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "models", "base"]);

      const parsed = JSON.parse(spy.mock.calls[0][0]);
      expect(parsed.data).toContainEqual({
        id: "Qwen/Qwen3.5-2B",
        name: "Qwen3.5-2B",
        type: "base",
      });
      expect(client.get).not.toHaveBeenCalled();
    });
  });

  describe("models list", () => {
    it("fetches and displays models", async () => {
      vi.mocked(client.get).mockResolvedValue({
        data: [mockModel],
        meta: { page: 1, per_page: 20, total: 1 },
      });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "models", "list"]);
      expect(client.get).toHaveBeenCalledWith(
        "/models",
        { page: "1", per_page: "20" },
        expect.anything(),
      );
    });

    it("outputs JSON when json mode is on", async () => {
      setJsonMode(true);
      vi.mocked(client.get).mockResolvedValue({
        data: [mockModel],
        meta: { page: 1, per_page: 20, total: 1 },
      });
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "models", "list"]);
      expect(JSON.parse(spy.mock.calls[0][0])).toHaveProperty("data");
    });
  });

  describe("models get", () => {
    it("fetches model details", async () => {
      vi.mocked(client.get)
        .mockResolvedValueOnce({
          data: [mockModel],
          meta: { page: 1, per_page: 100, total: 1 },
        })
        .mockResolvedValueOnce({ data: mockModel });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "models", "get", "model-1234"]);
      expect(client.get).toHaveBeenNthCalledWith(
        1,
        "/models",
        { page: 1, per_page: 100 },
        expect.anything(),
      );
      expect(client.get).toHaveBeenNthCalledWith(
        2,
        "/models/model-12345678-abcd",
        undefined,
        expect.anything(),
      );
    });

    it("displays all fields", async () => {
      vi.mocked(client.get)
        .mockResolvedValueOnce({
          data: [mockModel],
          meta: { page: 1, per_page: 100, total: 1 },
        })
        .mockResolvedValueOnce({ data: mockModel });
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "models", "get", "model-1234"]);
      const allOutput = spy.mock.calls.map((c) => String(c[0])).join("\n");
      expect(allOutput).toContain("my-fine-tuned-model");
      expect(allOutput).toContain("hosted");
      expect(allOutput).not.toMatch(/sagemaker|s3:\/\//i);
    });
  });

  describe("models delete", () => {
    it("deletes a model", async () => {
      vi.mocked(client.get).mockResolvedValue({
        data: [mockModel],
        meta: { page: 1, per_page: 100, total: 1 },
      });
      vi.mocked(client.del).mockResolvedValue({ data: null as never });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "models", "delete", "model-1234"]);
      expect(client.get).toHaveBeenCalledWith(
        "/models",
        { page: 1, per_page: 100 },
        expect.anything(),
      );
      expect(client.del).toHaveBeenCalledWith(
        "/models/model-12345678-abcd",
        expect.anything(),
      );
    });
  });

  describe("models download", () => {
    it("downloads a model artifact to the requested path", async () => {
      const outputPath = join(tmpdir(), `tt-model-download-${process.pid}.tar.gz`);
      if (existsSync(outputPath)) rmSync(outputPath);

      vi.mocked(client.get)
        .mockResolvedValueOnce({
          data: [mockModel],
          meta: { page: 1, per_page: 100, total: 1 },
        })
        .mockResolvedValueOnce({
          data: {
            url: "https://signed.example/model.tar.gz",
            filename: "model.tar.gz",
            expires_at: "2026-01-01T00:10:00Z",
          },
        });
      vi.spyOn(globalThis, "fetch").mockResolvedValue(
        new Response("artifact-bytes", {
          status: 200,
          headers: { "content-length": "14" },
        }),
      );
      vi.spyOn(console, "log").mockImplementation(() => {});

      const program = buildProgram();
      await program.parseAsync([
        "node",
        "tt",
        "models",
        "download",
        "model-1234",
        "--output",
        outputPath,
      ]);

      expect(client.get).toHaveBeenNthCalledWith(
        1,
        "/models",
        { page: 1, per_page: 100 },
        expect.anything(),
      );
      expect(client.get).toHaveBeenNthCalledWith(
        2,
        "/models/model-12345678-abcd/download",
        undefined,
        expect.anything(),
      );
      expect(globalThis.fetch).toHaveBeenCalledWith("https://signed.example/model.tar.gz");
      expect(readFileSync(outputPath, "utf8")).toBe("artifact-bytes");

      rmSync(outputPath);
    });

    it("renders an interactive progress bar with ETA", async () => {
      const outputPath = join(tmpdir(), `tt-model-download-progress-${process.pid}.tar.gz`);
      rmSync(outputPath, { force: true });
      const originalIsTTY = process.stderr.isTTY;
      const originalColumns = process.stderr.columns;
      Object.defineProperty(process.stderr, "isTTY", { value: true, configurable: true });
      Object.defineProperty(process.stderr, "columns", { value: 160, configurable: true });
      const writeSpy = vi.spyOn(process.stderr, "write").mockImplementation(() => true);

      try {
        vi.mocked(client.get)
          .mockResolvedValueOnce({
            data: [mockModel],
            meta: { page: 1, per_page: 100, total: 1 },
          })
          .mockResolvedValueOnce({
            data: {
              url: "https://signed.example/model.tar.gz",
              filename: "model.tar.gz",
              expires_at: "2026-01-01T00:10:00Z",
            },
          });
        vi.spyOn(globalThis, "fetch").mockResolvedValue(
          new Response("artifact-bytes", {
            status: 200,
            headers: { "content-length": "14" },
          }),
        );
        vi.spyOn(console, "log").mockImplementation(() => {});

        const program = buildProgram();
        await program.parseAsync([
          "node",
          "tt",
          "models",
          "download",
          "model-1234",
          "--output",
          outputPath,
        ]);

        const progressOutput = writeSpy.mock.calls.map((call) => String(call[0])).join("");
        expect(progressOutput).toContain("Downloading [");
        expect(progressOutput).toContain("100.0%");
        expect(progressOutput).toContain("ETA");
      } finally {
        writeSpy.mockRestore();
        Object.defineProperty(process.stderr, "isTTY", {
          value: originalIsTTY,
          configurable: true,
        });
        Object.defineProperty(process.stderr, "columns", {
          value: originalColumns,
          configurable: true,
        });
        rmSync(outputPath, { force: true });
      }
    });

    it("does not render progress in JSON mode", async () => {
      const outputPath = join(tmpdir(), `tt-model-download-json-${process.pid}.tar.gz`);
      rmSync(outputPath, { force: true });
      const originalIsTTY = process.stderr.isTTY;
      Object.defineProperty(process.stderr, "isTTY", { value: true, configurable: true });
      const writeSpy = vi.spyOn(process.stderr, "write").mockImplementation(() => true);

      try {
        setJsonMode(true);
        vi.mocked(client.get)
          .mockResolvedValueOnce({
            data: [mockModel],
            meta: { page: 1, per_page: 100, total: 1 },
          })
          .mockResolvedValueOnce({
            data: {
              url: "https://signed.example/model.tar.gz",
              filename: "model.tar.gz",
              expires_at: "2026-01-01T00:10:00Z",
            },
          });
        vi.spyOn(globalThis, "fetch").mockResolvedValue(
          new Response("artifact-bytes", {
            status: 200,
            headers: { "content-length": "14" },
          }),
        );
        vi.spyOn(console, "log").mockImplementation(() => {});

        const program = buildProgram();
        await program.parseAsync([
          "node",
          "tt",
          "models",
          "download",
          "model-1234",
          "--output",
          outputPath,
        ]);

        expect(writeSpy).not.toHaveBeenCalled();
      } finally {
        writeSpy.mockRestore();
        Object.defineProperty(process.stderr, "isTTY", {
          value: originalIsTTY,
          configurable: true,
        });
        rmSync(outputPath, { force: true });
      }
    });

    it("refuses to overwrite without --force", async () => {
      const outputPath = join(tmpdir(), `tt-model-download-existing-${process.pid}.tar.gz`);
      rmSync(outputPath, { force: true });
      writeFileSync(outputPath, "existing");

      vi.mocked(client.get)
        .mockResolvedValueOnce({
          data: [mockModel],
          meta: { page: 1, per_page: 100, total: 1 },
        })
        .mockResolvedValueOnce({
          data: {
            url: "https://signed.example/model.tar.gz",
            filename: "model.tar.gz",
            expires_at: "2026-01-01T00:10:00Z",
          },
        });

      const program = buildProgram();
      await expect(
        program.parseAsync([
          "node",
          "tt",
          "models",
          "download",
          "model-1234",
          "--output",
          outputPath,
        ]),
      ).rejects.toThrow("Output file already exists");

      rmSync(outputPath, { force: true });
    });
  });

  describe("models serve", () => {
    it("prints a local reference server command with the spec prompt enabled", async () => {
      setJsonMode(true);
      const root = join(tmpdir(), `tt-model-serve-${process.pid}`);
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
          "models",
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
        expect(client.get).not.toHaveBeenCalled();
      } finally {
        rmSync(root, { recursive: true, force: true });
      }
    });

    it("can disable the auto-applied spec prompt", async () => {
      setJsonMode(true);
      const root = join(tmpdir(), `tt-model-serve-no-spec-${process.pid}`);
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
          "models",
          "serve",
          modelDir,
          "--no-spec-prompt",
          "--cache-dir",
          cacheDir,
          "--print-command",
        ]);

        const parsed = JSON.parse(spy.mock.calls[0][0]);
        expect(parsed.env_keys).toContain("TT_MODEL_PATH");
        expect(parsed.env_keys).not.toContain("TT_SYSTEM_PROMPT");
      } finally {
        rmSync(root, { recursive: true, force: true });
      }
    });

    it("passes default JSON schema enforcement settings to the reference server", async () => {
      setJsonMode(true);
      const root = join(tmpdir(), `tt-model-serve-schema-${process.pid}`);
      const modelDir = join(root, "model");
      const cacheDir = join(root, "cache");
      const schemaPath = join(root, "schema.json");
      rmSync(root, { recursive: true, force: true });
      mkdirSync(modelDir, { recursive: true });
      writeFileSync(join(modelDir, "config.json"), "{}");
      writeFileSync(
        schemaPath,
        JSON.stringify({
          type: "object",
          required: ["triage", "risk", "should_process"],
          properties: {
            triage: { type: "string", enum: ["reply", "archive", "escalate", "ignore", "review"] },
            risk: { type: "string", enum: ["none", "spam", "phishing", "prompt_attack", "suspicious"] },
            should_process: { type: "boolean" },
          },
          additionalProperties: true,
        }),
      );

      const spy = vi.spyOn(console, "log").mockImplementation(() => {});

      try {
        const program = buildProgram();
        await program.parseAsync([
          "node",
          "tt",
          "models",
          "serve",
          modelDir,
          "--cache-dir",
          cacheDir,
          "--json-schema",
          schemaPath,
          "--json-repair-attempts",
          "2",
          "--print-command",
        ]);

        const parsed = JSON.parse(spy.mock.calls[0][0]);
        expect(parsed.env_keys).toContain("TT_JSON_SCHEMA");
        expect(parsed.env_keys).toContain("TT_JSON_REPAIR_ATTEMPTS");
      } finally {
        rmSync(root, { recursive: true, force: true });
      }
    });

    it("prints managed serving settings without starting the wrapper", async () => {
      setJsonMode(true);
      const root = join(tmpdir(), `tt-model-serve-managed-${process.pid}`);
      const modelDir = join(root, "model");
      const cacheDir = join(root, "cache");
      const logPath = join(root, "managed.jsonl");
      rmSync(root, { recursive: true, force: true });
      mkdirSync(modelDir, { recursive: true });
      writeFileSync(join(modelDir, "config.json"), "{}");

      const spy = vi.spyOn(console, "log").mockImplementation(() => {});

      try {
        const program = buildProgram();
        await program.parseAsync([
          "node",
          "tt",
          "models",
          "serve",
          modelDir,
          "--cache-dir",
          cacheDir,
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

  describe("models export", () => {
    function makeModelDir(label: string) {
      const root = join(tmpdir(), `tt-model-export-${label}-${process.pid}`);
      const modelDir = join(root, "model");
      const cacheDir = join(root, "cache");
      rmSync(root, { recursive: true, force: true });
      mkdirSync(modelDir, { recursive: true });
      writeFileSync(join(modelDir, "config.json"), "{}");
      return { root, modelDir, cacheDir };
    }

    it("plans a two-step convert + quantize for a k-quant", async () => {
      setJsonMode(true);
      const { root, modelDir, cacheDir } = makeModelDir("kquant");
      const outputDir = join(root, "out");
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});

      try {
        const program = buildProgram();
        await program.parseAsync([
          "node",
          "tt",
          "models",
          "export",
          modelDir,
          "--quant",
          "q4_k_m",
          "--output",
          outputDir,
          "--cache-dir",
          cacheDir,
          "--convert-script",
          "/opt/llama.cpp/convert_hf_to_gguf.py",
          "--quantize-bin",
          "/opt/llama.cpp/llama-quantize",
          "--print-command",
        ]);

        const plan = JSON.parse(spy.mock.calls[0][0]);
        expect(plan.format).toBe("gguf");
        expect(plan.quant).toBe("q4_k_m");
        expect(plan.steps).toHaveLength(2);
        expect(plan.steps[0].name).toBe("convert");
        expect(plan.steps[0].command_line).toContain("convert_hf_to_gguf.py");
        expect(plan.steps[0].command_line).toContain("--outtype f16");
        expect(plan.steps[1].name).toBe("quantize");
        expect(plan.steps[1].command_line).toContain("llama-quantize");
        expect(plan.steps[1].command_line).toContain("Q4_K_M");
        expect(plan.gguf_path.endsWith("q4_k_m.gguf")).toBe(true);
        expect(plan.ollama).toBeUndefined();
        expect(client.get).not.toHaveBeenCalled();
      } finally {
        rmSync(root, { recursive: true, force: true });
      }
    });

    it("uses a single convert step for convert-native outtypes like f16", async () => {
      setJsonMode(true);
      const { root, modelDir, cacheDir } = makeModelDir("f16");
      const outputDir = join(root, "out");
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});

      try {
        const program = buildProgram();
        await program.parseAsync([
          "node",
          "tt",
          "models",
          "export",
          modelDir,
          "--quant",
          "f16",
          "--output",
          outputDir,
          "--cache-dir",
          cacheDir,
          "--convert-script",
          "/opt/llama.cpp/convert_hf_to_gguf.py",
          "--print-command",
        ]);

        const plan = JSON.parse(spy.mock.calls[0][0]);
        expect(plan.steps).toHaveLength(1);
        expect(plan.steps[0].name).toBe("convert");
        expect(plan.steps[0].command_line).toContain("--outtype f16");
        expect(plan.intermediate_path).toBeUndefined();
        expect(plan.gguf_path.endsWith("f16.gguf")).toBe(true);
      } finally {
        rmSync(root, { recursive: true, force: true });
      }
    });

    it("writes a Modelfile with the spec prompt and an ollama create step with --ollama", async () => {
      setJsonMode(true);
      const root = join(tmpdir(), `tt-model-export-ollama-${process.pid}`);
      const modelDir = join(root, "PR Circuit Breaker");
      const cacheDir = join(root, "cache");
      rmSync(root, { recursive: true, force: true });
      mkdirSync(modelDir, { recursive: true });
      writeFileSync(join(modelDir, "config.json"), "{}");
      const outputDir = join(root, "out");
      const specPath = join(root, "tunedtensor.json");
      writeFileSync(
        specPath,
        JSON.stringify({
          name: "PR Circuit Breaker",
          base_model: "Qwen/Qwen3.5-2B",
          system_prompt: "You triage PRs.",
          guidelines: ["Be terse."],
          constraints: ["Never approve risky diffs."],
          examples: [{ input: "Hi", output: "Hello." }],
        }),
      );
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});

      try {
        const program = buildProgram();
        await program.parseAsync([
          "node",
          "tt",
          "models",
          "export",
          modelDir,
          "--quant",
          "q4_k_m",
          "--output",
          outputDir,
          "--cache-dir",
          cacheDir,
          "--ollama",
          "--spec",
          specPath,
          "--convert-script",
          "/opt/llama.cpp/convert_hf_to_gguf.py",
          "--quantize-bin",
          "/opt/llama.cpp/llama-quantize",
          "--print-command",
        ]);

        const plan = JSON.parse(spy.mock.calls[0][0]);
        expect(plan.ollama).toBeDefined();
        expect(plan.ollama.name).toBe("tt-pr-circuit-breaker");
        expect(plan.ollama.create).toBe(true);
        expect(plan.ollama.modelfile).toContain("FROM ./");
        expect(plan.ollama.modelfile).toContain("SYSTEM");
        expect(plan.ollama.modelfile).toContain("You triage PRs.");
        expect(plan.ollama.modelfile).toContain("Never approve risky diffs.");
        const createStep = plan.steps.find((s: { name: string }) => s.name === "ollama-create");
        expect(createStep.command_line).toContain("ollama create tt-pr-circuit-breaker");
      } finally {
        rmSync(root, { recursive: true, force: true });
      }
    });

    it("honors a custom --ollama-name and --no-ollama-create", async () => {
      setJsonMode(true);
      const { root, modelDir, cacheDir } = makeModelDir("ollama-name");
      const outputDir = join(root, "out");
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});

      try {
        const program = buildProgram();
        await program.parseAsync([
          "node",
          "tt",
          "models",
          "export",
          modelDir,
          "--output",
          outputDir,
          "--cache-dir",
          cacheDir,
          "--ollama",
          "--ollama-name",
          "tt-custom",
          "--no-ollama-create",
          "--no-spec-prompt",
          "--convert-script",
          "/opt/llama.cpp/convert_hf_to_gguf.py",
          "--quantize-bin",
          "/opt/llama.cpp/llama-quantize",
          "--print-command",
        ]);

        const plan = JSON.parse(spy.mock.calls[0][0]);
        expect(plan.ollama.name).toBe("tt-custom");
        expect(plan.ollama.create).toBe(false);
        expect(plan.ollama.modelfile).not.toContain("SYSTEM");
        expect(plan.steps.find((s: { name: string }) => s.name === "ollama-create")).toBeUndefined();
      } finally {
        rmSync(root, { recursive: true, force: true });
      }
    });

    it("rejects unsupported quant types", async () => {
      setJsonMode(true);
      const { root, modelDir, cacheDir } = makeModelDir("badquant");

      try {
        const program = buildProgram();
        await expect(
          program.parseAsync([
            "node",
            "tt",
            "models",
            "export",
            modelDir,
            "--quant",
            "q9_bogus",
            "--cache-dir",
            cacheDir,
            "--print-command",
          ]),
        ).rejects.toThrow(/Unsupported --quant/);
      } finally {
        rmSync(root, { recursive: true, force: true });
      }
    });

    it("rejects unsupported export formats", async () => {
      setJsonMode(true);
      const { root, modelDir, cacheDir } = makeModelDir("badformat");

      try {
        const program = buildProgram();
        await expect(
          program.parseAsync([
            "node",
            "tt",
            "models",
            "export",
            modelDir,
            "--format",
            "onnx",
            "--cache-dir",
            cacheDir,
            "--print-command",
          ]),
        ).rejects.toThrow(/Unsupported --format/);
      } finally {
        rmSync(root, { recursive: true, force: true });
      }
    });

  });
});
