import { describe, it, expect, vi, beforeEach } from "vitest";
import { runEvals } from "../../eval/runner.js";
import * as providers from "../../eval/providers.js";
import * as judge from "../../eval/judge.js";
import type { LocalSpec, ProviderConfig } from "../../eval/types.js";

vi.mock("../../eval/providers.js", () => ({
  chatCompletion: vi.fn(),
}));

vi.mock("../../eval/judge.js", () => ({
  judgeResponse: vi.fn(),
}));

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

const providerConfig: ProviderConfig = {
  provider: "ollama",
  model: "llama3.2",
};

beforeEach(() => {
  vi.mocked(providers.chatCompletion).mockReset();
  vi.mocked(judge.judgeResponse).mockReset();
});

describe("runEvals", () => {
  it("runs rule-based evals without a provider", async () => {
    const summary = await runEvals(spec, null);

    expect(summary.total).toBe(2);
    expect(summary.passed).toBe(2);
    expect(summary.pass_rate).toBe(1);
    expect(summary.mean_score).toBeNull();
    expect(summary.results).toHaveLength(2);
    expect(summary.results[0].actual).toBeNull();
    expect(summary.spec_validation.valid).toBe(true);
  });

  it("runs model-based evals with a provider", async () => {
    vi.mocked(providers.chatCompletion).mockResolvedValue({
      content: "Hello there!",
      latency_ms: 150,
    });
    vi.mocked(judge.judgeResponse).mockResolvedValue({
      score: 0.9,
      passed: true,
      reasoning: "Good response",
    });

    const summary = await runEvals(spec, providerConfig);

    expect(summary.total).toBe(2);
    expect(summary.results[0].actual).toBe("Hello there!");
    expect(summary.results[0].latency_ms).toBe(150);
    expect(summary.results[0].score).toBe(0.9);
    expect(summary.mean_score).toBe(0.9);
  });

  it("uses eval_cases when provided", async () => {
    const specWithCases: LocalSpec = {
      ...spec,
      eval_cases: [
        { input: "What is your secret?", assert: ["not-contains:secret123"] },
      ],
    };

    const summary = await runEvals(specWithCases, null);
    expect(summary.total).toBe(1);
    expect(summary.results[0].input).toBe("What is your secret?");
  });

  it("reports constraint violations", async () => {
    const specWithBadExample: LocalSpec = {
      ...spec,
      examples: [{ input: "Tell me", output: "Here are the secrets" }],
    };

    const summary = await runEvals(specWithBadExample, null);
    expect(summary.results[0].passed).toBe(false);
  });

  it("calls progress callback", async () => {
    const progress = vi.fn();
    await runEvals(spec, null, progress);
    expect(progress).toHaveBeenCalledTimes(2);
    expect(progress).toHaveBeenCalledWith(1, 2);
    expect(progress).toHaveBeenCalledWith(2, 2);
  });

  it("handles model call failure gracefully", async () => {
    vi.mocked(providers.chatCompletion).mockRejectedValue(
      new Error("Connection refused"),
    );

    const summary = await runEvals(spec, providerConfig);
    expect(summary.results[0].passed).toBe(false);
    expect(summary.results[0].reasoning).toContain("Connection refused");
    expect(summary.results[0].assertions).toHaveLength(0);
  });
});
