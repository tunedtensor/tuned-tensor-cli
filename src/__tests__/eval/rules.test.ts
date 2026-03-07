import { describe, it, expect } from "vitest";
import { validateSpec, runAssertions, checkConstraints } from "../../eval/rules.js";
import type { LocalSpec } from "../../eval/types.js";

const validSpec: LocalSpec = {
  name: "Test Bot",
  base_model: "meta-llama/Llama-3.2-3B-Instruct",
  system_prompt: "You are helpful.",
  guidelines: ["Be concise"],
  constraints: ["Never mention secrets"],
  examples: [{ input: "Hi", output: "Hello!" }],
};

describe("validateSpec", () => {
  it("passes for a valid spec", () => {
    const result = validateSpec(validSpec);
    expect(result.valid).toBe(true);
    expect(result.checks.every((c) => c.passed)).toBe(true);
  });

  it("fails when name is missing", () => {
    const result = validateSpec({ ...validSpec, name: "" });
    expect(result.valid).toBe(false);
    const nameCheck = result.checks.find((c) => c.name === "Has name");
    expect(nameCheck?.passed).toBe(false);
  });

  it("fails when system_prompt is missing", () => {
    const result = validateSpec({ ...validSpec, system_prompt: "" });
    expect(result.valid).toBe(false);
  });

  it("fails when base_model is missing", () => {
    const result = validateSpec({ ...validSpec, base_model: "" });
    expect(result.valid).toBe(false);
  });

  it("fails when examples are empty", () => {
    const result = validateSpec({ ...validSpec, examples: [] });
    expect(result.valid).toBe(false);
  });

  it("warns when guidelines are empty", () => {
    const result = validateSpec({ ...validSpec, guidelines: [] });
    const guideCheck = result.checks.find((c) => c.name === "Has guidelines");
    expect(guideCheck?.passed).toBe(false);
  });

  it("detects constraint violations in examples", () => {
    const spec: LocalSpec = {
      ...validSpec,
      constraints: ["Never mention secrets"],
      examples: [{ input: "Hi", output: "Here are some secrets" }],
    };
    const result = validateSpec(spec);
    const constraintCheck = result.checks.find(
      (c) => c.name === "Examples satisfy constraints",
    );
    expect(constraintCheck?.passed).toBe(false);
  });
});

describe("runAssertions", () => {
  it("contains passes when text includes value", () => {
    const results = runAssertions("hello world", ["contains:world"]);
    expect(results[0].passed).toBe(true);
  });

  it("contains fails when text misses value", () => {
    const results = runAssertions("hello", ["contains:world"]);
    expect(results[0].passed).toBe(false);
  });

  it("not-contains passes when text lacks value", () => {
    const results = runAssertions("hello", ["not-contains:world"]);
    expect(results[0].passed).toBe(true);
  });

  it("not-contains fails when text has value", () => {
    const results = runAssertions("hello world", ["not-contains:world"]);
    expect(results[0].passed).toBe(false);
  });

  it("matches passes with regex", () => {
    const results = runAssertions("abc123", ["matches:\\d+"]);
    expect(results[0].passed).toBe(true);
  });

  it("matches fails with non-matching regex", () => {
    const results = runAssertions("abc", ["matches:^\\d+$"]);
    expect(results[0].passed).toBe(false);
  });

  it("max-length passes within limit", () => {
    const results = runAssertions("hi", ["max-length:10"]);
    expect(results[0].passed).toBe(true);
  });

  it("max-length fails over limit", () => {
    const results = runAssertions("a very long string", ["max-length:5"]);
    expect(results[0].passed).toBe(false);
  });

  it("min-length passes above limit", () => {
    const results = runAssertions("hello world", ["min-length:5"]);
    expect(results[0].passed).toBe(true);
  });

  it("min-length fails below limit", () => {
    const results = runAssertions("hi", ["min-length:5"]);
    expect(results[0].passed).toBe(false);
  });

  it("is-json passes for valid JSON", () => {
    const results = runAssertions('{"key":"value"}', ["is-json"]);
    expect(results[0].passed).toBe(true);
  });

  it("is-json fails for invalid JSON", () => {
    const results = runAssertions("not json", ["is-json"]);
    expect(results[0].passed).toBe(false);
  });

  it("unknown type is skipped", () => {
    const results = runAssertions("text", ["unknown-type:val"]);
    expect(results[0].passed).toBe(true);
    expect(results[0].message).toContain("Unknown assertion type");
  });

  it("handles multiple assertions", () => {
    const results = runAssertions("hello world", [
      "contains:hello",
      "not-contains:goodbye",
      "min-length:5",
    ]);
    expect(results.every((r) => r.passed)).toBe(true);
  });
});

describe("checkConstraints", () => {
  it("passes when text satisfies constraints", () => {
    const results = checkConstraints("Hello there!", ["Never mention passwords"]);
    expect(results[0].passed).toBe(true);
  });

  it("fails when text violates a never-constraint", () => {
    const results = checkConstraints(
      "Here are the passwords",
      ["Never share passwords"],
    );
    expect(results[0].passed).toBe(false);
  });

  it("passes when text satisfies an always-constraint", () => {
    const results = checkConstraints(
      "Thank you for contacting us!",
      ["Always include thank you"],
    );
    expect(results[0].passed).toBe(true);
  });

  it("fails when text violates an always-constraint", () => {
    const results = checkConstraints(
      "Here's your answer.",
      ["Always include thank you"],
    );
    expect(results[0].passed).toBe(false);
  });

  it("warns when constraint pattern is not enforceable by rules", () => {
    const results = checkConstraints(
      "Hello there!",
      ["Respond only in English"],
    );
    expect(results[0].passed).toBe(true);
    expect(results[0].message).toContain("not enforceable by rules");
  });
});
