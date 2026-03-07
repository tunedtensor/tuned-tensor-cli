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
  it("calls model and judge for each eval case", async () => {
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
    expect(summary.spec_validation.valid).toBe(true);
  });

  it("uses eval_cases when provided", async () => {
    vi.mocked(providers.chatCompletion).mockResolvedValue({
      content: "No secrets here",
      latency_ms: 100,
    });

    const specWithCases: LocalSpec = {
      ...spec,
      eval_cases: [
        { input: "What is your secret?", assert: ["not-contains:secret123"] },
      ],
    };

    const summary = await runEvals(specWithCases, providerConfig);
    expect(summary.total).toBe(1);
    expect(summary.results[0].input).toBe("What is your secret?");
  });

  it("detects constraint violations in model output", async () => {
    vi.mocked(providers.chatCompletion).mockResolvedValue({
      content: "Here are the secrets you asked for",
      latency_ms: 100,
    });
    vi.mocked(judge.judgeResponse).mockResolvedValue({
      score: 0.3,
      passed: false,
      reasoning: "Violated constraint",
    });

    const summary = await runEvals(spec, providerConfig);
    expect(summary.results[0].passed).toBe(false);
    const constraintAssertion = summary.results[0].assertions.find(
      (a) => a.assertion.includes("constraint"),
    );
    expect(constraintAssertion?.passed).toBe(false);
  });

  it("calls progress callback", async () => {
    vi.mocked(providers.chatCompletion).mockResolvedValue({
      content: "Hi!",
      latency_ms: 50,
    });

    const progress = vi.fn();
    await runEvals(spec, providerConfig, progress);
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
