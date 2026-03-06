import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

let configModule: typeof import("../config.js");

const TEST_DIR = join(tmpdir(), `tt-test-config-${process.pid}`);

beforeEach(async () => {
  rmSync(TEST_DIR, { recursive: true, force: true });
  process.env.XDG_CONFIG_HOME = TEST_DIR;
  delete process.env.TUNED_TENSOR_URL;
  delete process.env.TUNED_TENSOR_API_KEY;
  configModule = await import("../config.js");
});

afterEach(() => {
  rmSync(TEST_DIR, { recursive: true, force: true });
  delete process.env.XDG_CONFIG_HOME;
});

describe("config", () => {
  describe("readConfig", () => {
    it("returns empty object when no config file exists", () => {
      expect(configModule.readConfig()).toEqual({});
    });

    it("reads existing config", () => {
      const dir = join(TEST_DIR, "tuned-tensor");
      mkdirSync(dir, { recursive: true });
      writeFileSync(
        join(dir, "config.json"),
        JSON.stringify({ api_key: "tt_test123" }),
      );
      expect(configModule.readConfig()).toEqual({ api_key: "tt_test123" });
    });

    it("returns empty object for malformed JSON", () => {
      const dir = join(TEST_DIR, "tuned-tensor");
      mkdirSync(dir, { recursive: true });
      writeFileSync(join(dir, "config.json"), "not json");
      expect(configModule.readConfig()).toEqual({});
    });
  });

  describe("writeConfig", () => {
    it("creates config dir and file", () => {
      configModule.writeConfig({ api_key: "tt_abc" });
      const path = join(TEST_DIR, "tuned-tensor", "config.json");
      expect(existsSync(path)).toBe(true);
      expect(JSON.parse(readFileSync(path, "utf-8"))).toEqual({
        api_key: "tt_abc",
      });
    });
  });

  describe("updateConfig", () => {
    it("merges partial config into existing", () => {
      configModule.writeConfig({ api_key: "tt_old" });
      configModule.updateConfig({ base_url: "https://example.com" });
      const config = configModule.readConfig();
      expect(config.api_key).toBe("tt_old");
      expect(config.base_url).toBe("https://example.com");
    });
  });

  describe("clearConfig", () => {
    it("resets config to empty object", () => {
      configModule.writeConfig({ api_key: "tt_old" });
      configModule.clearConfig();
      expect(configModule.readConfig()).toEqual({});
    });

    it("does nothing when no config file exists", () => {
      expect(() => configModule.clearConfig()).not.toThrow();
    });
  });

  describe("getBaseUrl", () => {
    it("returns default URL when nothing is set", () => {
      expect(configModule.getBaseUrl()).toBe("https://www.tunedtensor.com");
    });

    it("uses opts.baseUrl first", () => {
      expect(configModule.getBaseUrl({ baseUrl: "https://custom.com" })).toBe(
        "https://custom.com",
      );
    });

    it("uses TUNED_TENSOR_URL env var", () => {
      process.env.TUNED_TENSOR_URL = "https://env.com";
      expect(configModule.getBaseUrl()).toBe("https://env.com");
    });

    it("uses stored config", () => {
      configModule.writeConfig({ base_url: "https://stored.com" });
      expect(configModule.getBaseUrl()).toBe("https://stored.com");
    });
  });

  describe("getApiKey", () => {
    it("returns undefined when nothing is set", () => {
      expect(configModule.getApiKey()).toBeUndefined();
    });

    it("uses opts.apiKey first", () => {
      expect(configModule.getApiKey({ apiKey: "tt_opts" })).toBe("tt_opts");
    });

    it("uses TUNED_TENSOR_API_KEY env var", () => {
      process.env.TUNED_TENSOR_API_KEY = "tt_env";
      expect(configModule.getApiKey()).toBe("tt_env");
    });

    it("uses stored config", () => {
      configModule.writeConfig({ api_key: "tt_stored" });
      expect(configModule.getApiKey()).toBe("tt_stored");
    });
  });
});
