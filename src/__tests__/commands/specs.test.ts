import { describe, it, expect, vi, beforeEach } from "vitest";
import { Command } from "commander";
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
