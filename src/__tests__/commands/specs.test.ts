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

const mockSpec = {
  id: "spec-12345678-abcd",
  name: "Test Spec",
  description: "A test spec",
  system_prompt: "You are helpful.",
  guidelines: ["Be nice"],
  examples: [{ input: "hi", output: "hello" }],
  constraints: ["No bad words"],
  base_model: "meta-llama/Llama-3.2-3B-Instruct",
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
      await program.parseAsync(["node", "tt", "specs", "get", "spec-1234"]);
      expect(client.get).toHaveBeenCalledWith(
        "/behavior-specs/spec-1234",
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
        "--model", "meta-llama/Llama-3.2-3B-Instruct",
      ]);
      expect(client.post).toHaveBeenCalledWith(
        "/behavior-specs",
        { name: "My Spec", base_model: "meta-llama/Llama-3.2-3B-Instruct" },
        expect.anything(),
      );
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
          base_model: "x/y",
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
          { name: "From File", base_model: "x/y", examples: [] },
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
          base_model: "x/y",
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
    });
  });

  describe("specs update", () => {
    it("updates a spec with --name", async () => {
      vi.mocked(client.put).mockResolvedValue({ data: mockSpec });
      vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync([
        "node", "tt", "specs", "update", "spec-1234", "--name", "New Name",
      ]);
      expect(client.put).toHaveBeenCalledWith(
        "/behavior-specs/spec-1234",
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
      await program.parseAsync(["node", "tt", "specs", "delete", "spec-1234"]);
      expect(client.del).toHaveBeenCalledWith(
        "/behavior-specs/spec-1234",
        expect.anything(),
      );
    });
  });
});
