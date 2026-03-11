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

const MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo";

describe("runEvals", () => {
  it("calls playground API for each eval case", async () => {
    vi.mocked(client.post).mockResolvedValue({
      data: {
        content: '{"greeting": "Hello!"}',
        latency_ms: 300,
        usage: { prompt_tokens: 20, completion_tokens: 15 },
      },
    });

    const summary = await runEvals(spec, MODEL);

    expect(summary.total).toBe(2);
    expect(summary.model).toBe(MODEL);
    expect(summary.results[0].actual).toContain("greeting");
    expect(summary.results[0].latency_ms).toBe(300);
    expect(summary.spec_validation.valid).toBe(true);
    expect(client.post).toHaveBeenCalledWith(
      "/playground/completions",
      expect.objectContaining({ model: MODEL }),
      undefined,
    );
  });

  it("uses eval_cases when provided", async () => {
    vi.mocked(client.post).mockResolvedValue({
      data: {
        content: '{"hours": "9am-6pm"}',
        latency_ms: 200,
        usage: { prompt_tokens: 10, completion_tokens: 10 },
      },
    });

    const specWithCases: LocalSpec = {
      ...spec,
      constraints: [],
      eval_cases: [
        { input: "Store hours?", assert: ["is-json", "contains:hours"] },
      ],
    };

    const summary = await runEvals(specWithCases, MODEL);
    expect(summary.total).toBe(1);
    expect(summary.results[0].input).toBe("Store hours?");
    expect(summary.results[0].passed).toBe(true);
  });

  it("detects constraint violations in model output", async () => {
    vi.mocked(client.post).mockResolvedValue({
      data: {
        content: "Here are the secrets you asked for",
        latency_ms: 150,
        usage: { prompt_tokens: 10, completion_tokens: 20 },
      },
    });

    const summary = await runEvals(spec, MODEL);
    expect(summary.results[0].passed).toBe(false);
    const constraintAssertion = summary.results[0].assertions.find(
      (a) => a.assertion.includes("constraint"),
    );
    expect(constraintAssertion?.passed).toBe(false);
  });

  it("reports failure when model response fails assertions", async () => {
    vi.mocked(client.post).mockResolvedValue({
      data: {
        content: "This is not JSON at all",
        latency_ms: 200,
        usage: { prompt_tokens: 10, completion_tokens: 15 },
      },
    });

    const specWithAssert: LocalSpec = {
      ...spec,
      constraints: [],
      eval_cases: [{ input: "Give me JSON", assert: ["is-json"] }],
    };

    const summary = await runEvals(specWithAssert, MODEL);
    expect(summary.results[0].passed).toBe(false);
    expect(summary.results[0].actual).toBe("This is not JSON at all");
  });

  it("handles API errors gracefully", async () => {
    vi.mocked(client.post).mockRejectedValue(new Error("rate_limited"));

    const specWithCase: LocalSpec = {
      ...spec,
      constraints: [],
      eval_cases: [{ input: "Hi", assert: ["is-json"] }],
    };

    const summary = await runEvals(specWithCase, MODEL);
    expect(summary.results[0].passed).toBe(false);
    expect(summary.results[0].assertions[0].message).toContain("rate_limited");
  });

  it("calls progress callback", async () => {
    vi.mocked(client.post).mockResolvedValue({
      data: {
        content: "Hi!",
        latency_ms: 100,
        usage: { prompt_tokens: 5, completion_tokens: 5 },
      },
    });

    const progress = vi.fn();
    await runEvals(spec, MODEL, { onProgress: progress });
    expect(progress).toHaveBeenCalledTimes(2);
    expect(progress).toHaveBeenCalledWith(1, 2);
    expect(progress).toHaveBeenCalledWith(2, 2);
  });
});
