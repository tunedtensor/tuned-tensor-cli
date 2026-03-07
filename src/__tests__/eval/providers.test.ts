import { describe, it, expect, vi, beforeEach } from "vitest";
import { chatCompletion, checkProviderAvailability } from "../../eval/providers.js";
import type { ProviderConfig } from "../../eval/types.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

beforeEach(() => {
  mockFetch.mockReset();
});

describe("chatCompletion", () => {
  const config: ProviderConfig = {
    provider: "ollama",
    model: "llama3.2",
  };

  it("calls the OpenAI-compatible endpoint and returns content", async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        choices: [{ message: { content: "Hello!" } }],
      }),
    });

    const result = await chatCompletion(
      [
        { role: "system", content: "You are helpful." },
        { role: "user", content: "Hi" },
      ],
      config,
    );

    expect(result.content).toBe("Hello!");
    expect(result.latency_ms).toBeGreaterThanOrEqual(0);
    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:11434/v1/chat/completions",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("uses openai base URL for openai provider", async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        choices: [{ message: { content: "Hi" } }],
      }),
    });

    const openaiConfig: ProviderConfig = {
      provider: "openai",
      model: "gpt-4o-mini",
      apiKey: "sk-test",
    };

    await chatCompletion([{ role: "user", content: "Hi" }], openaiConfig);
    expect(mockFetch).toHaveBeenCalledWith(
      "https://api.openai.com/v1/chat/completions",
      expect.anything(),
    );
  });

  it("includes auth header when API key is provided", async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        choices: [{ message: { content: "Hi" } }],
      }),
    });

    await chatCompletion(
      [{ role: "user", content: "Hi" }],
      { ...config, apiKey: "test-key" },
    );

    const callArgs = mockFetch.mock.calls[0][1];
    expect(callArgs.headers.Authorization).toBe("Bearer test-key");
  });

  it("throws on API error", async () => {
    mockFetch.mockResolvedValue({
      ok: false,
      status: 500,
      statusText: "Internal Server Error",
      text: async () => "Server Error",
    });

    await expect(
      chatCompletion([{ role: "user", content: "Hi" }], config),
    ).rejects.toThrow("ollama API error [500]");
  });

  it("uses custom base URL when provided", async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        choices: [{ message: { content: "Hi" } }],
      }),
    });

    const customConfig: ProviderConfig = {
      provider: "custom",
      model: "my-model",
      baseUrl: "https://my-api.example.com",
    };

    await chatCompletion([{ role: "user", content: "Hi" }], customConfig);
    expect(mockFetch).toHaveBeenCalledWith(
      "https://my-api.example.com/v1/chat/completions",
      expect.anything(),
    );
  });
});

describe("checkProviderAvailability", () => {
  it("returns available for healthy ollama with matching model", async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        models: [{ name: "llama3.2:latest" }],
      }),
    });

    const result = await checkProviderAvailability({
      provider: "ollama",
      model: "llama3.2",
    });
    expect(result.available).toBe(true);
  });

  it("returns unavailable when ollama model is missing", async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ models: [{ name: "mistral:latest" }] }),
    });

    const result = await checkProviderAvailability({
      provider: "ollama",
      model: "llama3.2",
    });
    expect(result.available).toBe(false);
    expect(result.error).toContain("not found");
  });

  it("handles connection refused", async () => {
    mockFetch.mockRejectedValue(new Error("fetch failed"));

    const result = await checkProviderAvailability({
      provider: "ollama",
      model: "llama3.2",
    });
    expect(result.available).toBe(false);
  });
});
