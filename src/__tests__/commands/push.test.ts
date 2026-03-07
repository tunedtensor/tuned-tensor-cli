import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { Command } from "commander";
import { writeFileSync, readFileSync, unlinkSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { registerPushCommand } from "../../commands/push.js";
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
const TEST_FILE = resolve("test-push-spec.json");

function buildProgram() {
  const program = new Command();
  program
    .option("-k, --api-key <key>", "API key")
    .option("-u, --base-url <url>", "Base URL")
    .option("--json", "JSON mode");
  registerPushCommand(program);
  program.exitOverride();
  return program;
}

beforeEach(() => {
  setJsonMode(false);
  process.env.TUNED_TENSOR_API_KEY = FAKE_KEY;
  vi.spyOn(console, "log").mockImplementation(() => {});
});

afterEach(() => {
  if (existsSync(TEST_FILE)) unlinkSync(TEST_FILE);
});

describe("push command", () => {
  it("creates a new spec when no id is present", async () => {
    const spec = {
      name: "Test Bot",
      base_model: "llama3.2",
      system_prompt: "You are helpful.",
      guidelines: [],
      constraints: [],
      examples: [{ input: "Hi", output: "Hello!" }],
    };
    writeFileSync(TEST_FILE, JSON.stringify(spec));

    vi.mocked(client.post).mockResolvedValue({
      data: { id: "spec-new-id-12345678", name: "Test Bot" },
    });

    const program = buildProgram();
    await program.parseAsync([
      "node", "tt", "push", "--file", "test-push-spec.json",
    ]);

    expect(client.post).toHaveBeenCalledWith(
      "/behavior-specs",
      expect.objectContaining({ name: "Test Bot" }),
      expect.anything(),
    );

    const updated = JSON.parse(readFileSync(TEST_FILE, "utf-8"));
    expect(updated.id).toBe("spec-new-id-12345678");
  });

  it("updates an existing spec when id is present", async () => {
    const spec = {
      id: "spec-existing-id",
      name: "Test Bot",
      base_model: "llama3.2",
      system_prompt: "Updated prompt",
      guidelines: [],
      constraints: [],
      examples: [{ input: "Hi", output: "Hello!" }],
    };
    writeFileSync(TEST_FILE, JSON.stringify(spec));

    vi.mocked(client.put).mockResolvedValue({
      data: { id: "spec-existing-id", name: "Test Bot" },
    });

    const program = buildProgram();
    await program.parseAsync([
      "node", "tt", "push", "--file", "test-push-spec.json",
    ]);

    expect(client.put).toHaveBeenCalledWith(
      "/behavior-specs/spec-existing-id",
      expect.objectContaining({ name: "Test Bot" }),
      expect.anything(),
    );
  });

  it("strips eval_cases before pushing", async () => {
    const spec = {
      name: "Test Bot",
      base_model: "llama3.2",
      system_prompt: "Helpful",
      guidelines: [],
      constraints: [],
      examples: [{ input: "Hi", output: "Hello!" }],
      eval_cases: [{ input: "Secret?", assert: ["not-contains:secret"] }],
    };
    writeFileSync(TEST_FILE, JSON.stringify(spec));

    vi.mocked(client.post).mockResolvedValue({
      data: { id: "spec-123", name: "Test Bot" },
    });

    const program = buildProgram();
    await program.parseAsync([
      "node", "tt", "push", "--file", "test-push-spec.json",
    ]);

    const body = vi.mocked(client.post).mock.calls[0][1] as Record<string, unknown>;
    expect(body.eval_cases).toBeUndefined();
  });
});
