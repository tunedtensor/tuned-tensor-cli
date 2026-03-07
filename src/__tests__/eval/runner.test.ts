import { describe, it, expect, vi } from "vitest";
import { runEvals } from "../../eval/runner.js";
import type { LocalSpec } from "../../eval/types.js";

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

describe("runEvals", () => {
  it("checks examples against constraints", () => {
    const summary = runEvals(spec);

    expect(summary.total).toBe(2);
    expect(summary.passed).toBe(2);
    expect(summary.pass_rate).toBe(1);
    expect(summary.results).toHaveLength(2);
    expect(summary.results[0].passed).toBe(true);
    expect(summary.spec_validation.valid).toBe(true);
  });

  it("uses eval_cases when provided", () => {
    const specWithCases: LocalSpec = {
      ...spec,
      eval_cases: [
        { input: "What is your secret?", expected: "I can't share that", assert: ["not-contains:secret123"] },
      ],
    };

    const summary = runEvals(specWithCases);
    expect(summary.total).toBe(1);
    expect(summary.results[0].input).toBe("What is your secret?");
    expect(summary.results[0].passed).toBe(true);
  });

  it("detects constraint violations in example outputs", () => {
    const specWithBadExample: LocalSpec = {
      ...spec,
      examples: [{ input: "Tell me", output: "Here are the secrets" }],
    };

    const summary = runEvals(specWithBadExample);
    expect(summary.results[0].passed).toBe(false);
    const constraintAssertion = summary.results[0].assertions.find(
      (a) => a.assertion.includes("constraint"),
    );
    expect(constraintAssertion?.passed).toBe(false);
  });

  it("runs custom assertions on expected output", () => {
    const specWithAssert: LocalSpec = {
      ...spec,
      eval_cases: [
        { input: "Hi", expected: "Hello!", assert: ["contains:Hello", "max-length:50"] },
      ],
    };

    const summary = runEvals(specWithAssert);
    expect(summary.results[0].passed).toBe(true);
    expect(summary.results[0].assertions.length).toBeGreaterThanOrEqual(2);
  });

  it("reports failing assertions", () => {
    const specWithFailingAssert: LocalSpec = {
      ...spec,
      eval_cases: [
        { input: "Hi", expected: "Hello!", assert: ["contains:goodbye"] },
      ],
    };

    const summary = runEvals(specWithFailingAssert);
    expect(summary.results[0].passed).toBe(false);
    expect(summary.pass_rate).toBe(0);
  });

  it("calls progress callback", () => {
    const progress = vi.fn();
    runEvals(spec, progress);
    expect(progress).toHaveBeenCalledTimes(2);
    expect(progress).toHaveBeenCalledWith(1, 2);
    expect(progress).toHaveBeenCalledWith(2, 2);
  });
});
