import { afterEach, describe, expect, it, vi } from "vitest";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  createPiModelRuntime,
  resolveAgentModel,
  type AgentModelRuntime,
} from "../agent-model.js";

const originalAgentDir = process.env.PI_CODING_AGENT_DIR;

afterEach(() => {
  if (originalAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
  else process.env.PI_CODING_AGENT_DIR = originalAgentDir;
});

const model = {
  id: "claude-sonnet-4-5",
  provider: "anthropic",
  name: "Claude Sonnet 4.5",
  reasoning: true,
};

function runtime(overrides: Partial<AgentModelRuntime> = {}): AgentModelRuntime {
  return {
    getProviders: vi.fn(() => [{ id: "anthropic", name: "Anthropic" }]),
    getModels: vi.fn(() => [model]),
    getModel: vi.fn(() => model),
    hasConfiguredAuth: vi.fn(() => true),
    ...overrides,
  };
}

describe("local agent model resolution", () => {
  it("selects a configured Pi model without accepting or returning credentials", () => {
    const result = resolveAgentModel(runtime(), {
      provider: "anthropic",
      model: "claude-sonnet-4-5",
      thinking: "high",
    });

    expect(result).toEqual({ model, thinking: "high" });
    expect(result).not.toHaveProperty("apiKey");
  });

  it("gives an actionable error when provider auth is missing", () => {
    expect(() => resolveAgentModel(runtime({
      hasConfiguredAuth: vi.fn(() => false),
    }), {
      provider: "anthropic",
      model: "claude-sonnet-4-5",
      thinking: "medium",
    })).toThrow(/authenticate.*Pi.*anthropic/i);
  });

  it("rejects a thinking level unsupported by the selected model", () => {
    expect(() => resolveAgentModel(runtime({
      getModel: vi.fn(() => ({ ...model, reasoning: false })),
    }), {
      provider: "anthropic",
      model: "claude-sonnet-4-5",
      thinking: "high",
    })).toThrow(/does not support thinking/i);
  });

  it("loads Pi's production provider catalog without credentials or network", async () => {
    const agentDir = mkdtempSync(join(tmpdir(), "tt-pi-runtime-"));
    process.env.PI_CODING_AGENT_DIR = agentDir;
    try {
      const fetchSpy = vi.spyOn(globalThis, "fetch");
      const productionRuntime = await createPiModelRuntime();
      expect(productionRuntime.getProviders().length).toBeGreaterThan(0);
      expect(productionRuntime.getModels().length).toBeGreaterThan(0);
      expect(fetchSpy).not.toHaveBeenCalled();
    } finally {
      rmSync(agentDir, { recursive: true, force: true });
    }
  });

  it("adds OpenRouter app attribution headers to models.json", async () => {
    const agentDir = mkdtempSync(join(tmpdir(), "tt-pi-runtime-"));
    process.env.PI_CODING_AGENT_DIR = agentDir;
    try {
      await createPiModelRuntime();
      const modelsJson = JSON.parse(
        readFileSync(join(agentDir, "models.json"), "utf-8"),
      );
      expect(modelsJson.providers.openrouter.headers).toEqual({
        "HTTP-Referer": "https://tunedtensor.com",
        "X-Title": "Tuned Tensor",
      });
    } finally {
      rmSync(agentDir, { recursive: true, force: true });
    }
  });

  it("preserves existing models.json config when adding OpenRouter headers", async () => {
    const agentDir = mkdtempSync(join(tmpdir(), "tt-pi-runtime-"));
    process.env.PI_CODING_AGENT_DIR = agentDir;
    try {
      writeFileSync(
        join(agentDir, "models.json"),
        JSON.stringify({
          providers: {
            openrouter: { headers: { "X-Custom": "keep-me" } },
            ollama: { baseUrl: "http://localhost:11434/v1" },
          },
        }),
      );
      await createPiModelRuntime();
      const modelsJson = JSON.parse(
        readFileSync(join(agentDir, "models.json"), "utf-8"),
      );
      expect(modelsJson.providers.openrouter.headers).toEqual({
        "X-Custom": "keep-me",
        "HTTP-Referer": "https://tunedtensor.com",
        "X-Title": "Tuned Tensor",
      });
      expect(modelsJson.providers.ollama.baseUrl).toBe(
        "http://localhost:11434/v1",
      );
    } finally {
      rmSync(agentDir, { recursive: true, force: true });
    }
  });

  it("leaves an unparseable models.json untouched", async () => {
    const agentDir = mkdtempSync(join(tmpdir(), "tt-pi-runtime-"));
    process.env.PI_CODING_AGENT_DIR = agentDir;
    try {
      writeFileSync(join(agentDir, "models.json"), "{ not json");
      await createPiModelRuntime();
      expect(readFileSync(join(agentDir, "models.json"), "utf-8")).toBe(
        "{ not json",
      );
    } finally {
      rmSync(agentDir, { recursive: true, force: true });
    }
  });
});
