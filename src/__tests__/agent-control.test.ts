import { afterEach, describe, expect, it } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  describeAgentModel,
  findAgentProvider,
  listAgentModels,
  listAgentProviders,
  loginAgentProvider,
  setAgentModel,
} from "../agent-control.js";
import type { AgentModelRuntime } from "../agent-model.js";

const models = [
  { id: "claude-sonnet-4-5", provider: "anthropic", name: "Claude Sonnet 4.5", reasoning: true },
  { id: "claude-haiku-4-5", provider: "anthropic", name: "Claude Haiku 4.5", reasoning: false },
  { id: "gpt-5.2", provider: "openai", name: "GPT 5.2", reasoning: true },
  { id: "meta-llama/llama-3.3-70b", provider: "openrouter", name: "Llama 3.3 70B", reasoning: true },
];

function makeRuntime(overrides: Partial<AgentModelRuntime> = {}): AgentModelRuntime {
  return {
    getProviders: () => [
      { id: "anthropic", name: "Anthropic" },
      { id: "openai", name: "OpenAI" },
      { id: "openrouter", name: "OpenRouter" },
    ],
    getModels: () => models,
    getModel: (provider, model) =>
      models.find((candidate) => candidate.provider === provider && candidate.id === model),
    hasConfiguredAuth: () => true,
    ...overrides,
  };
}

const roots: string[] = [];

function temporaryConfigRoot(): string {
  const root = mkdtempSync(join(tmpdir(), "tt-agent-control-"));
  roots.push(root);
  process.env.XDG_CONFIG_HOME = root;
  return root;
}

afterEach(() => {
  delete process.env.XDG_CONFIG_HOME;
  for (const root of roots.splice(0)) {
    rmSync(root, { recursive: true, force: true });
  }
});

describe("listAgentModels", () => {
  it("ranks search matches by relevance and caps with limit", () => {
    const runtime = makeRuntime();
    const results = listAgentModels(runtime, { query: "sonnet", limit: 1 });
    expect(results).toEqual([
      { provider: "anthropic", id: "claude-sonnet-4-5", name: "Claude Sonnet 4.5", authenticated: true, thinking: true },
    ]);

    const allMatches = listAgentModels(runtime, { query: "claude" });
    expect(allMatches.map((model) => model.id)).toEqual([
      "claude-haiku-4-5",
      "claude-sonnet-4-5",
    ]);
  });

  it("finds a bare id even when it contains a slash", () => {
    const runtime = makeRuntime();
    const results = listAgentModels(runtime, { query: "llama-3.3-70b" });
    expect(results.map((model) => `${model.provider}/${model.id}`)).toEqual([
      "openrouter/meta-llama/llama-3.3-70b",
    ]);
  });

  it("filters out providers without configured auth by default", () => {
    const runtime = makeRuntime({
      hasConfiguredAuth: (provider) => provider !== "openrouter",
    });
    const results = listAgentModels(runtime);
    expect(results.some((model) => model.provider === "openrouter")).toBe(false);

    const all = listAgentModels(runtime, { includeUnauthenticated: true });
    expect(all.some((model) => model.provider === "openrouter")).toBe(true);
  });
});

describe("listAgentProviders", () => {
  it("lists providers with auth state and matches an exact provider id", () => {
    const runtime = makeRuntime({
      hasConfiguredAuth: (provider) => provider === "anthropic",
    });
    expect(listAgentProviders(runtime)).toEqual([
      { id: "anthropic", name: "Anthropic", authenticated: true },
      { id: "openai", name: "OpenAI", authenticated: false },
      { id: "openrouter", name: "OpenRouter", authenticated: false },
    ]);
    expect(findAgentProvider(runtime, "Anthropic")?.id).toBe("anthropic");
    expect(findAgentProvider(runtime, "sonnet")).toBeUndefined();
  });
});

describe("loginAgentProvider", () => {
  it("stores a trimmed API key for a known provider", async () => {
    const keys: Array<{ provider: string; apiKey: string }> = [];
    const runtime = makeRuntime({
      setRuntimeApiKey: async (provider, apiKey) => {
        keys.push({ provider, apiKey });
      },
    });
    await expect(loginAgentProvider(runtime, "OpenRouter", "  sk-or-test  "))
      .resolves.toEqual({ provider: "openrouter" });
    expect(keys).toEqual([{ provider: "openrouter", apiKey: "sk-or-test" }]);
  });

  it("rejects an unknown provider, empty key, or runtime that cannot store credentials", async () => {
    const runtime = makeRuntime();
    await expect(loginAgentProvider(runtime, "missing", "sk-test"))
      .rejects.toThrow(/unknown provider "missing"/i);
    await expect(loginAgentProvider(runtime, "anthropic", "   "))
      .rejects.toThrow(/cannot be empty/i);
    await expect(loginAgentProvider(runtime, "anthropic", "sk-test"))
      .rejects.toThrow(/cannot store provider credentials/i);
  });
});

describe("agent model selection", () => {
  it("describes and then updates the selected agent model", () => {
    const configRoot = temporaryConfigRoot();
    const runtime = makeRuntime();

    expect(describeAgentModel(runtime, { HOME: configRoot })).toBeUndefined();

    const result = setAgentModel(runtime, { HOME: configRoot }, "anthropic", "claude-haiku-4-5");
    expect(result).toEqual({
      selection: { provider: "anthropic", model: "claude-haiku-4-5", thinking: "off" },
      adjustedThinking: true,
    });

    const summary = describeAgentModel(runtime, { HOME: configRoot });
    expect(summary).toMatchObject({
      provider: "anthropic",
      model: "claude-haiku-4-5",
      thinking: "off",
      authenticated: true,
      supportsThinking: false,
    });
  });

  it("preserves the current thinking level when switching between reasoning models", () => {
    const configRoot = temporaryConfigRoot();
    const runtime = makeRuntime();

    setAgentModel(runtime, { HOME: configRoot }, "openai", "gpt-5.2", { thinking: "high" });
    const result = setAgentModel(runtime, { HOME: configRoot }, "anthropic", "claude-sonnet-4-5");
    expect(result.selection.thinking).toBe("high");
    expect(result.adjustedThinking).toBe(false);
  });
});
