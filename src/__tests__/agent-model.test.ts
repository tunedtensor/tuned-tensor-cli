import { afterEach, describe, expect, it, vi } from "vitest";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import {
  createPiModelRuntime,
  getAgentModelsPath,
  resolveAgentModel,
  type AgentModelRuntime,
} from "../agent-model.js";
import { FEATURED_AGENT_PROVIDERS, recommendAgentModels } from "../agent-control.js";

const originalConfigHome = process.env.XDG_CONFIG_HOME;
const originalPiAgentDir = process.env.PI_CODING_AGENT_DIR;

afterEach(() => {
  if (originalConfigHome === undefined) delete process.env.XDG_CONFIG_HOME;
  else process.env.XDG_CONFIG_HOME = originalConfigHome;
  if (originalPiAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
  else process.env.PI_CODING_AGENT_DIR = originalPiAgentDir;
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

function isolatedConfigHome(): string {
  const root = mkdtempSync(join(tmpdir(), "tt-agent-runtime-"));
  process.env.XDG_CONFIG_HOME = root;
  return root;
}

function ttModelsPath(xdg: string): string {
  return join(xdg, "tuned-tensor", "agent", "models.json");
}

function writeTtModels(xdg: string, contents: string): string {
  const modelsPath = ttModelsPath(xdg);
  mkdirSync(dirname(modelsPath), { recursive: true });
  writeFileSync(modelsPath, contents);
  return modelsPath;
}

describe("local agent model resolution", () => {
  it("selects a configured model without accepting or returning credentials", () => {
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
    })).toThrow(/not authenticated.*\/login anthropic/i);
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

  it("loads the production provider catalog without credentials or network", async () => {
    const xdg = isolatedConfigHome();
    try {
      const fetchSpy = vi.spyOn(globalThis, "fetch");
      const productionRuntime = await createPiModelRuntime();
      expect(productionRuntime.getProviders().length).toBeGreaterThan(0);
      expect(productionRuntime.getModels().length).toBeGreaterThan(0);
      const providerIds = productionRuntime.getProviders().map((provider) => provider.id);
      expect(providerIds).toEqual(expect.arrayContaining([...FEATURED_AGENT_PROVIDERS]));
      expect(recommendAgentModels(productionRuntime).map((model) => `${model.provider}/${model.id}`))
        .toEqual([
          "openai/gpt-5.6-sol",
          "openrouter/deepseek/deepseek-v4-flash-0731",
        ]);
      expect(fetchSpy).not.toHaveBeenCalled();
    } finally {
      rmSync(xdg, { recursive: true, force: true });
    }
  });

  it("stores provider files under the TT config dir, not Pi's agent dir", async () => {
    const xdg = isolatedConfigHome();
    const piDir = mkdtempSync(join(tmpdir(), "tt-pi-agent-"));
    process.env.PI_CODING_AGENT_DIR = piDir;
    try {
      await createPiModelRuntime();
      expect(getAgentModelsPath()).toBe(ttModelsPath(xdg));
      expect(existsSync(ttModelsPath(xdg))).toBe(true);
      expect(readdirSync(piDir)).toEqual([]);
    } finally {
      rmSync(xdg, { recursive: true, force: true });
      rmSync(piDir, { recursive: true, force: true });
    }
  });

  it("adds OpenRouter app attribution headers to models.json", async () => {
    const xdg = isolatedConfigHome();
    try {
      await createPiModelRuntime();
      const modelsJson = JSON.parse(readFileSync(ttModelsPath(xdg), "utf-8"));
      expect(modelsJson.providers.openrouter.headers).toEqual({
        "HTTP-Referer": "https://tunedtensor.com",
        "X-OpenRouter-Title": "Tuned Tensor",
        "X-OpenRouter-Categories": "cli-agent",
        "X-Title": "Tuned Tensor",
      });
    } finally {
      rmSync(xdg, { recursive: true, force: true });
    }
  });

  it("preserves existing models.json config when adding OpenRouter headers", async () => {
    const xdg = isolatedConfigHome();
    try {
      const modelsPath = writeTtModels(
        xdg,
        JSON.stringify({
          providers: {
            openrouter: {
              headers: {
                "X-Custom": "keep-me",
                "http-referer": "https://user.example",
                "x-OpenRouter-title": "User title",
                "X-OPENROUTER-CATEGORIES": "general-chat",
                "x-title": "Legacy user title",
              },
            },
            ollama: { baseUrl: "http://localhost:11434/v1" },
          },
        }),
      );
      const runtime = await createPiModelRuntime();
      const modelsJson = JSON.parse(readFileSync(modelsPath, "utf-8"));
      expect(modelsJson.providers.openrouter.headers).toEqual({
        "X-Custom": "keep-me",
        "HTTP-Referer": "https://tunedtensor.com",
        "X-OpenRouter-Title": "Tuned Tensor",
        "X-OpenRouter-Categories": "cli-agent",
        "X-Title": "Tuned Tensor",
      });
      const openRouterModel = runtime.getModels("openrouter")[0];
      expect(openRouterModel).toBeDefined();
      await runtime.setRuntimeApiKey("openrouter", "synthetic-test-key");
      const auth = await runtime.getAuth(openRouterModel!);
      expect(auth?.auth.headers).toEqual(modelsJson.providers.openrouter.headers);
      expect(modelsJson.providers.ollama.baseUrl).toBe(
        "http://localhost:11434/v1",
      );
    } finally {
      rmSync(xdg, { recursive: true, force: true });
    }
  });

  it("leaves an unparseable models.json untouched", async () => {
    const xdg = isolatedConfigHome();
    try {
      const modelsPath = writeTtModels(xdg, "{ not json");
      await createPiModelRuntime();
      expect(readFileSync(modelsPath, "utf-8")).toBe("{ not json");
    } finally {
      rmSync(xdg, { recursive: true, force: true });
    }
  });
});
