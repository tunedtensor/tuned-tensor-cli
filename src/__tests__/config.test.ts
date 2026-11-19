import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { canonicalizeBaseModel } from "../base-models.js";

let configModule: typeof import("../config.js");

const TEST_DIR = join(tmpdir(), `tt-test-config-${process.pid}`);

beforeEach(async () => {
  rmSync(TEST_DIR, { recursive: true, force: true });
  process.env.TUNED_TENSOR_HOME = TEST_DIR;
  configModule = await import("../config.js");
});

afterEach(() => {
  rmSync(TEST_DIR, { recursive: true, force: true });
  delete process.env.TUNED_TENSOR_HOME;
  delete process.env.TUNED_TENSOR_AGENT_PROVIDER;
  delete process.env.TUNED_TENSOR_AGENT_MODEL;
  delete process.env.TUNED_TENSOR_AGENT_THINKING;
});

describe("config", () => {
  it("canonicalizes the certified Nemotron Lightning base model", () => {
    expect(canonicalizeBaseModel("nvidia/nvidia-nemotron-3.5-lightning-30b-a3b-bf16"))
      .toBe("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16");
  });

  it("canonicalizes the certified Muse Glimmer base model", () => {
    expect(canonicalizeBaseModel("meta-models/muse-glimmer-30b"))
      .toBe("meta-models/Muse-Glimmer-30B");
  });
  describe("readConfig", () => {
    it("returns empty object when no config file exists", () => {
      expect(configModule.readConfig()).toEqual({});
    });

    it("reads existing config", () => {
      const dir = TEST_DIR;
      mkdirSync(dir, { recursive: true });
      writeFileSync(
        join(dir, "config.json"),
        JSON.stringify({ agent: { provider: "openai", model: "gpt-5.2", thinking: "high" } }),
      );
      expect(configModule.readConfig()).toEqual({ agent: { provider: "openai", model: "gpt-5.2", thinking: "high" } });
    });

    it("returns empty object for malformed JSON", () => {
      const dir = TEST_DIR;
      mkdirSync(dir, { recursive: true });
      writeFileSync(join(dir, "config.json"), "not json");
      expect(configModule.readConfig()).toEqual({});
    });
  });

  describe("writeConfig", () => {
    it("creates config dir and file", () => {
      const agent = { provider: "openai", model: "gpt-5.2", thinking: "high" } as const;
      configModule.writeConfig({ agent });
      const path = join(TEST_DIR, "config.json");
      expect(existsSync(path)).toBe(true);
      expect(JSON.parse(readFileSync(path, "utf-8"))).toEqual({ agent });
      if (process.platform !== "win32") {
        expect(
          statSync(TEST_DIR).mode & 0o777,
        ).toBe(0o700);
        expect(statSync(path).mode & 0o777).toBe(0o600);
      }
    });
  });

  describe("updateConfig", () => {
    it("merges partial config into existing", () => {
      mkdirSync(TEST_DIR, { recursive: true });
      writeFileSync(join(TEST_DIR, "config.json"), JSON.stringify({ project_note: "keep" }));
      const agent = { provider: "openai", model: "gpt-5.2", thinking: "high" } as const;
      configModule.updateConfig({ agent });
      expect(configModule.readConfig()).toEqual({ project_note: "keep", agent });
    });

    it("removes the retired hosted-service token and URL on the next write", () => {
      mkdirSync(TEST_DIR, { recursive: true });
      writeFileSync(join(TEST_DIR, "config.json"), JSON.stringify({
        api_key: "tt_retired", base_url: "https://tunedtensor.com",
      }));
      const agent = { provider: "openai", model: "gpt-5.2", thinking: "high" } as const;
      configModule.updateConfig({ agent });
      expect(configModule.readConfig()).toEqual({ agent });
      expect(readFileSync(join(TEST_DIR, "config.json"), "utf8")).not.toContain("tt_retired");
    });
  });

  describe("clearConfig", () => {
    it("resets config to empty object", () => {
      configModule.writeConfig({ agent: { provider: "openai", model: "gpt-5.2", thinking: "high" } });
      configModule.clearConfig();
      expect(configModule.readConfig()).toEqual({});
    });

    it("does nothing when no config file exists", () => {
      expect(() => configModule.clearConfig()).not.toThrow();
    });
  });

  describe("getAgentConfigDir", () => {
    it("places agent files under the TT config directory", () => {
      expect(configModule.getAgentConfigDir()).toBe(join(TEST_DIR, "agent"));
    });
  });

  describe("local agent model selection", () => {
    it("is unconfigured until the user chooses a provider, even with a leftover TT token", () => {
      expect(configModule.getAgentSelection({})).toBeUndefined();
      mkdirSync(TEST_DIR, { recursive: true });
      writeFileSync(join(TEST_DIR, "config.json"), JSON.stringify({ api_key: "tt_saved" }));
      expect(configModule.getAgentSelection({})).toBeUndefined();
      expect(configModule.getAgentSelection({ TUNED_TENSOR_API_KEY: "tt_env" })).toBeUndefined();
    });

    it("treats a saved selection of the retired managed model as unconfigured", () => {
      mkdirSync(TEST_DIR, { recursive: true });
      writeFileSync(join(TEST_DIR, "config.json"), JSON.stringify({
        api_key: "tt_saved",
        agent: { provider: "tunedtensor", model: "managed", thinking: "off" },
      }));
      expect(configModule.getAgentSelection({})).toBeUndefined();
      expect(configModule.getAgentSelection({
        TUNED_TENSOR_AGENT_PROVIDER: "openai", TUNED_TENSOR_AGENT_MODEL: "gpt-5.2",
      })).toEqual({ provider: "openai", model: "gpt-5.2", thinking: "medium" });
    });

    it("explains that managed inference is retired when the environment selects it", () => {
      expect(() => configModule.getAgentSelection({ TUNED_TENSOR_AGENT_PROVIDER: "tunedtensor" }))
        .toThrow(/managed inference has been retired/i);
    });

    it("preserves an explicit selection of the user's own provider", () => {
      const agent = { provider: "openrouter", model: "vendor/new-model", thinking: "high" } as const;
      configModule.writeConfig({ agent });
      expect(configModule.getAgentSelection({})).toEqual(agent);
    });

    it("stores only non-secret metadata and applies environment overrides", () => {
      configModule.updateConfig({
        agent: {
          provider: "anthropic",
          model: "claude-sonnet-4-5",
          thinking: "medium",
        },
      });

      process.env.TUNED_TENSOR_AGENT_PROVIDER = "openai";
      process.env.TUNED_TENSOR_AGENT_MODEL = "gpt-5.2";
      process.env.TUNED_TENSOR_AGENT_THINKING = "high";

      expect(configModule.getAgentSelection()).toEqual({
        provider: "openai",
        model: "gpt-5.2",
        thinking: "high",
      });
      expect(readFileSync(
        join(TEST_DIR, "config.json"),
        "utf8",
      )).not.toContain("api_key");
    });

    it("rejects unsupported thinking levels", () => {
      process.env.TUNED_TENSOR_AGENT_PROVIDER = "openai";
      process.env.TUNED_TENSOR_AGENT_MODEL = "gpt-5.2";
      process.env.TUNED_TENSOR_AGENT_THINKING = "unlimited";

      expect(() => configModule.getAgentSelection()).toThrow(
        /thinking.*off, minimal, low, medium, high, xhigh, or max/i,
      );
    });

    it("accepts the max thinking level", () => {
      process.env.TUNED_TENSOR_AGENT_PROVIDER = "openai";
      process.env.TUNED_TENSOR_AGENT_MODEL = "gpt-5.2";
      process.env.TUNED_TENSOR_AGENT_THINKING = "max";

      expect(configModule.getAgentSelection()).toEqual({
        provider: "openai",
        model: "gpt-5.2",
        thinking: "max",
      });
    });
  });
});
