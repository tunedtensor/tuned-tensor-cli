import { describe, it, expect, vi } from "vitest";
import { runEvals } from "../../eval/runner.js";
import * as client from "../../client.js";
import type { LocalSpec } from "../../eval/types.js";

vi.mock("../../client.js", async (importOriginal) => {
  const actual = await importOriginal<typeof client>();
  return { ...actual, post: vi.fn() };
});

const spec: LocalSpec = {
  name: "Test Bot",
  base_model: "meta-llama/Llama-3.2-3B-Instruct",
  system_prompt: "You are helpful.",
  guidelines: ["Be concise"],
  constraints: ["Never mention secrets"],
  examples: [
    { input: "Hi", output: "Hello!" },
    { input: "Bye", output: "Goodbye!" },
  ],
};

describe("runEvals (offline)", () => {
  it("checks examples against constraints", async () => {
    const summary = await runEvals(spec);

    expect(summary.total).toBe(2);
    expect(summary.passed).toBe(2);
    expect(summary.pass_rate).toBe(1);
    expect(summary.model).toBeNull();
    expect(summary.results).toHaveLength(2);
    expect(summary.results[0].actual).toBeNull();
    expect(summary.results[0].passed).toBe(true);
    expect(summary.spec_validation.valid).toBe(true);
  });

  it("uses eval_cases when provided", async () => {
    const specWithCases: LocalSpec = {
      ...spec,
      eval_cases: [
        { input: "What is your secret?", expected: "I can't share that", assert: ["not-contains:secret123"] },
      ],
    };

    const summary = await runEvals(specWithCases);
    expect(summary.total).toBe(1);
    expect(summary.results[0].input).toBe("What is your secret?");
    expect(summary.results[0].passed).toBe(true);
  });

  it("detects constraint violations in example outputs", async () => {
    const specWithBadExample: LocalSpec = {
      ...spec,
      examples: [{ input: "Tell me", output: "Here are the secrets" }],
    };

    const summary = await runEvals(specWithBadExample);
    expect(summary.results[0].passed).toBe(false);
    const constraintAssertion = summary.results[0].assertions.find(
      (a) => a.assertion.includes("constraint"),
    );
    expect(constraintAssertion?.passed).toBe(false);
  });

  it("calls progress callback", async () => {
    const progress = vi.fn();
    await runEvals(spec, { onProgress: progress });
    expect(progress).toHaveBeenCalledTimes(2);
    expect(progress).toHaveBeenCalledWith(1, 2);
    expect(progress).toHaveBeenCalledWith(2, 2);
  });
});

describe("runEvals (model)", () => {
  it("calls playground API and asserts against model response", async () => {
    vi.mocked(client.post).mockResolvedValue({
      data: {
        content: '{"hours": {"monday_friday": "9am-6pm"}}',
        latency_ms: 450,
        usage: { prompt_tokens: 20, completion_tokens: 30 },
      },
    });

    const specWithModel: LocalSpec = {
      ...spec,
      constraints: [],
      eval_cases: [
        { input: "Store hours?", assert: ["is-json", "contains:hours"] },
      ],
    };

    const summary = await runEvals(specWithModel, { model: "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo" });

    expect(summary.model).toBe("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo");
    expect(summary.results[0].actual).toContain("hours");
    expect(summary.results[0].latency_ms).toBe(450);
    expect(summary.results[0].passed).toBe(true);
    expect(client.post).toHaveBeenCalledWith(
      "/playground/completions",
      expect.objectContaining({ model: "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo" }),
      undefined,
    );
  });

  it("reports failure when model response fails assertions", async () => {
    vi.mocked(client.post).mockResolvedValue({
      data: {
        content: "This is not JSON at all",
        latency_ms: 200,
        usage: { prompt_tokens: 10, completion_tokens: 15 },
      },
    });

    const specWithModel: LocalSpec = {
      ...spec,
      constraints: [],
      eval_cases: [
        { input: "Give me JSON", assert: ["is-json"] },
      ],
    };

    const summary = await runEvals(specWithModel, { model: "test-model" });
    expect(summary.results[0].passed).toBe(false);
    expect(summary.results[0].actual).toBe("This is not JSON at all");
  });

  it("handles API errors gracefully", async () => {
    vi.mocked(client.post).mockRejectedValue(new Error("Connection refused"));

    const specWithModel: LocalSpec = {
      ...spec,
      constraints: [],
      eval_cases: [{ input: "Hi", assert: ["is-json"] }],
    };

    const summary = await runEvals(specWithModel, { model: "test-model" });
    expect(summary.results[0].passed).toBe(false);
    expect(summary.results[0].assertions[0].message).toContain("Connection refused");
  });
});
