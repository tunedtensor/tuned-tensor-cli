import { describe, it, expect, vi, beforeEach } from "vitest";
import { judgeResponse } from "../../eval/judge.js";
import * as providers from "../../eval/providers.js";
import type { ProviderConfig } from "../../eval/types.js";

vi.mock("../../eval/providers.js", () => ({
  chatCompletion: vi.fn(),
}));

const config: ProviderConfig = {
  provider: "ollama",
  model: "llama3.2",
};

beforeEach(() => {
  vi.mocked(providers.chatCompletion).mockReset();
});

describe("judgeResponse", () => {
  it("parses a valid judge response", async () => {
    vi.mocked(providers.chatCompletion).mockResolvedValue({
      content: '{"score": 0.85, "passed": true, "reasoning": "Good response"}',
      latency_ms: 100,
    });

    const result = await judgeResponse({
      input: "How do I reset my password?",
      output: "Go to Settings > Security > Reset Password.",
      expected: "Go to Settings -> Security -> Reset Password.",
      systemPrompt: "You are a helpful assistant.",
      guidelines: ["Be concise"],
      constraints: [],
      config,
    });

    expect(result.score).toBe(0.85);
    expect(result.passed).toBe(true);
    expect(result.reasoning).toBe("Good response");
  });

  it("handles JSON embedded in other text", async () => {
    vi.mocked(providers.chatCompletion).mockResolvedValue({
      content: 'Here is my evaluation:\n{"score": 0.6, "passed": false, "reasoning": "Too vague"}\nDone.',
      latency_ms: 100,
    });

    const result = await judgeResponse({
      input: "Help me",
      output: "Sure",
      expected: null,
      systemPrompt: "Be helpful",
      guidelines: ["Be detailed"],
      constraints: [],
      config,
    });

    expect(result.score).toBe(0.6);
    expect(result.passed).toBe(false);
  });

  it("clamps score to 0-1 range", async () => {
    vi.mocked(providers.chatCompletion).mockResolvedValue({
      content: '{"score": 1.5, "passed": true, "reasoning": "Great"}',
      latency_ms: 100,
    });

    const result = await judgeResponse({
      input: "Hi",
      output: "Hello!",
      expected: null,
      systemPrompt: "Be helpful",
      guidelines: [],
      constraints: [],
      config,
    });

    expect(result.score).toBe(1.0);
  });

  it("returns failure for invalid judge output", async () => {
    vi.mocked(providers.chatCompletion).mockResolvedValue({
      content: "I cannot evaluate this.",
      latency_ms: 100,
    });

    const result = await judgeResponse({
      input: "Hi",
      output: "Hello!",
      expected: null,
      systemPrompt: "Be helpful",
      guidelines: [],
      constraints: [],
      config,
    });

    expect(result.score).toBe(0);
    expect(result.passed).toBe(false);
    expect(result.reasoning).toContain("invalid format");
  });

  it("includes guidelines and constraints in the judge prompt", async () => {
    vi.mocked(providers.chatCompletion).mockResolvedValue({
      content: '{"score": 0.9, "passed": true, "reasoning": "OK"}',
      latency_ms: 100,
    });

    await judgeResponse({
      input: "Hi",
      output: "Hello!",
      expected: "Hello there!",
      systemPrompt: "Be helpful",
      guidelines: ["Be friendly", "Be concise"],
      constraints: ["Never be rude"],
      config,
    });

    const callArgs = vi.mocked(providers.chatCompletion).mock.calls[0];
    const prompt = callArgs[0][0].content;
    expect(prompt).toContain("Be friendly");
    expect(prompt).toContain("Be concise");
    expect(prompt).toContain("Never be rude");
    expect(prompt).toContain("Hello there!");
  });
});
