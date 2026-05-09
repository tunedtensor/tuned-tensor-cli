import { describe, it, expect, vi, beforeEach } from "vitest";
import { Command } from "commander";
import { writeFileSync, mkdirSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { registerSpecsCommands } from "../../commands/specs.js";
import * as client from "../../client.js";
import { setJsonMode } from "../../output.js";

vi.mock("../../client.js", async (importOriginal) => {
  const actual = await importOriginal<typeof client>();
  return {
    ...actual,
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
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
  registerSpecsCommands(program);
  program.exitOverride();
  return program;
}

const SPEC_UUID = "11111111-1111-4111-8111-111111111111";
const SPEC_UUID_2 = "22222222-2222-4222-8222-222222222222";

const mockSpec = {
  id: SPEC_UUID,
  name: "Test Spec",
  description: "A test spec",
  system_prompt: "You are helpful.",
  guidelines: ["Be nice"],
  examples: [{ input: "hi", output: "hello" }],
  constraints: ["No bad words"],
  base_model: "Qwen/Qwen3.5-2B",
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-01-01T00:00:00Z",
  _run_count: 2,
  _latest_run_status: "completed",
};

beforeEach(() => {
  setJsonMode(false);
  process.env.TUNED_TENSOR_API_KEY = FAKE_KEY;
  vi.mocked(client.get).mockReset();
  vi.mocked(client.post).mockReset();
  vi.mocked(client.put).mockReset();
  vi.mocked(client.del).mockReset();
});

describe("specs commands", () => {
  describe("specs list", () => {
    it("fetches and displays specs", async () => {
      vi.mocked(client.get).mockResolvedValue({
        data: [mockSpec],
        meta: { page: 1, per_page: 20, total: 1 },
      });
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "specs", "list"]);
      expect(client.get).toHaveBeenCalledWith(
        "/behavior-specs",
        { page: "1", per_page: "20" },
        expect.anything(),
      );
    });

    it("outputs JSON when --json is set", async () => {
      setJsonMode(true);
      vi.mocked(client.get).mockResolvedValue({
        data: [mockSpec],
        meta: { page: 1, per_page: 20, total: 1 },
      });
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "specs", "list"]);
      const output = spy.mock.calls[0][0];
      expect(JSON.parse(output)).toHaveProperty("data");
    });
  });

  describe("specs get", () => {
    it("fetches and displays a single spec", async () => {
      vi.mocked(client.get).mockResolvedValue({ data: mockSpec });
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "specs", "get", SPEC_UUID]);
      expect(client.get).toHaveBeenCalledWith(
        `/behavior-specs/${SPEC_UUID}`,
        undefined,
        expect.anything(),
      );
    });
  });

  describe("specs create", () => {
    it("creates a spec with --name", async () => {
      vi.mocked(client.post).mockResolvedValue({ data: mockSpec });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync([
        "node", "tt", "specs", "create", "--name", "My Spec",
      ]);
      expect(client.post).toHaveBeenCalledWith(
        "/behavior-specs",
        { name: "My Spec" },
        expect.anything(),
      );
    });

    it("creates a spec with --name and --model", async () => {
      vi.mocked(client.post).mockResolvedValue({ data: mockSpec });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync([
        "node", "tt", "specs", "create",
        "--name", "My Spec",
        "--model", "Qwen/Qwen3.5-2B",
      ]);
      expect(client.post).toHaveBeenCalledWith(
        "/behavior-specs",
        { name: "My Spec", base_model: "Qwen/Qwen3.5-2B" },
        expect.anything(),
      );
    });

    it("canonicalizes supported model aliases before posting", async () => {
      vi.mocked(client.post).mockResolvedValue({ data: mockSpec });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync([
        "node", "tt", "specs", "create",
        "--name", "My Spec",
        "--model", "qwen/qwen3.5-2b-base",
      ]);
      expect(client.post).toHaveBeenCalledWith(
        "/behavior-specs",
        { name: "My Spec", base_model: "Qwen/Qwen3.5-2B" },
        expect.anything(),
      );
    });

    it("rejects unsupported inline model values before posting", async () => {
      const program = buildProgram();
      await expect(
        program.parseAsync([
          "node", "tt", "specs", "create",
          "--name", "My Spec",
          "--model", "Qwen/Qwen2.5-1.5B-Instruct",
        ]),
      ).rejects.toThrow(/Unsupported base_model.*Qwen\/Qwen3\.5-2B/);

      expect(client.post).not.toHaveBeenCalled();
    });

    describe("--file validation", () => {
      const fixtureDir = join(tmpdir(), `tt-test-specs-${process.pid}`);
      const writeFixture = (name: string, body: unknown) => {
        const path = join(fixtureDir, name);
        writeFileSync(path, JSON.stringify(body));
        return path;
      };

      beforeEach(() => {
        rmSync(fixtureDir, { recursive: true, force: true });
        mkdirSync(fixtureDir, { recursive: true });
      });

      it("posts the file body when valid", async () => {
        const path = writeFixture("spec.json", {
          name: "From File",
          base_model: "google/gemma-4-e2b",
          examples: [],
        });
        vi.mocked(client.post).mockResolvedValue({ data: mockSpec });
        vi.spyOn(console, "log").mockImplementation(() => {});

        const program = buildProgram();
        await program.parseAsync([
          "node", "tt", "specs", "create", "--file", path,
        ]);

        expect(client.post).toHaveBeenCalledWith(
          "/behavior-specs",
          { name: "From File", base_model: "google/gemma-4-E2B-it", examples: [] },
          expect.anything(),
        );
      });

      it("rejects a run-input shape with a helpful hint", async () => {
        const path = writeFixture("run-input.json", {
          run_id: "r1",
          behavior_spec_id: "s1",
          run_number: 1,
          spec_snapshot: { name: "Nested Spec" },
          hyperparameters: {},
        });

        const program = buildProgram();
        await expect(
          program.parseAsync(["node", "tt", "specs", "create", "--file", path]),
        ).rejects.toThrow(/run-input payload.*spec_snapshot/s);

        expect(client.post).not.toHaveBeenCalled();
      });

      it("rejects when name is missing on create", async () => {
        const path = writeFixture("nameless.json", { description: "hi" });

        const program = buildProgram();
        await expect(
          program.parseAsync(["node", "tt", "specs", "create", "--file", path]),
        ).rejects.toThrow(/missing required field "name"/);

        expect(client.post).not.toHaveBeenCalled();
      });

      it("warns on unknown top-level keys but still posts", async () => {
        const path = writeFixture("extra.json", {
          name: "Has Extras",
          base_model: "Qwen/Qwen3.5-2B",
          totally_unknown_field: 42,
        });
        vi.mocked(client.post).mockResolvedValue({ data: mockSpec });
        const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});

        const program = buildProgram();
        await program.parseAsync([
          "node", "tt", "specs", "create", "--file", path,
        ]);

        const warnedAt = logSpy.mock.calls.findIndex((c) =>
          String(c[0]).includes("totally_unknown_field"),
        );
        expect(warnedAt).toBeGreaterThanOrEqual(0);
        expect(client.post).toHaveBeenCalledWith(
          "/behavior-specs",
          expect.objectContaining({ totally_unknown_field: 42 }),
          expect.anything(),
        );
      });

      it("rejects unsupported file model values before posting", async () => {
        const path = writeFixture("bad-model.json", {
          name: "Bad Model",
          base_model: "Qwen/Qwen2.5-1.5B-Instruct",
          examples: [],
        });

        const program = buildProgram();
        await expect(
          program.parseAsync(["node", "tt", "specs", "create", "--file", path]),
        ).rejects.toThrow(/Unsupported base_model/);

        expect(client.post).not.toHaveBeenCalled();
      });
    });
  });

  describe("specs update", () => {
    it("updates a spec with --name", async () => {
      vi.mocked(client.put).mockResolvedValue({ data: mockSpec });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync([
        "node", "tt", "specs", "update", SPEC_UUID, "--name", "New Name",
      ]);
      expect(client.put).toHaveBeenCalledWith(
        `/behavior-specs/${SPEC_UUID}`,
        { name: "New Name" },
        expect.anything(),
      );
    });
  });

  describe("specs delete", () => {
    it("deletes a spec", async () => {
      vi.mocked(client.del).mockResolvedValue({ data: null as never });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "specs", "delete", SPEC_UUID]);
      expect(client.del).toHaveBeenCalledWith(
        `/behavior-specs/${SPEC_UUID}`,
        expect.anything(),
      );
    });

    it("resolves an 8-char prefix via the list endpoint", async () => {
      const otherSpec = { ...mockSpec, id: SPEC_UUID_2, name: "Other" };
      vi.mocked(client.get).mockResolvedValue({
        data: [mockSpec, otherSpec],
        meta: { page: 1, per_page: 100, total: 2 },
      });
      vi.mocked(client.del).mockResolvedValue({ data: null as never });
      vi.spyOn(console, "log").mockImplementation(() => {});

      const program = buildProgram();
      await program.parseAsync([
        "node", "tt", "specs", "delete", SPEC_UUID.slice(0, 8),
      ]);

      expect(client.get).toHaveBeenCalledWith(
        "/behavior-specs",
        { page: 1, per_page: 100 },
        expect.anything(),
      );
      expect(client.del).toHaveBeenCalledWith(
        `/behavior-specs/${SPEC_UUID}`,
        expect.anything(),
      );
    });
  });
});
