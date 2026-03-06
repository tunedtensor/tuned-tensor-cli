import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { Command } from "commander";
import { registerAuthCommands } from "../../commands/auth.js";
import * as config from "../../config.js";
import { rmSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

const TEST_DIR = join(tmpdir(), `tt-test-auth-${process.pid}`);

beforeEach(() => {
  rmSync(TEST_DIR, { recursive: true, force: true });
  process.env.XDG_CONFIG_HOME = TEST_DIR;
  delete process.env.TUNED_TENSOR_API_KEY;
  delete process.env.TUNED_TENSOR_URL;
});

afterEach(() => {
  rmSync(TEST_DIR, { recursive: true, force: true });
  delete process.env.XDG_CONFIG_HOME;
});

function buildProgram() {
  const program = new Command();
  program
    .option("-k, --api-key <key>", "API key")
    .option("-u, --base-url <url>", "Base URL")
    .option("--json", "JSON mode");
  registerAuthCommands(program);
  program.exitOverride();
  return program;
}

const VALID_KEY = "tt_" + "a".repeat(48);

describe("auth commands", () => {
  describe("auth login", () => {
    it("stores a valid API key passed as argument", async () => {
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "auth", "login", VALID_KEY]);
      const stored = config.readConfig();
      expect(stored.api_key).toBe(VALID_KEY);
    });

    it("rejects keys that don't start with tt_", async () => {
      vi.spyOn(console, "error").mockImplementation(() => {});
      const exitSpy = vi.spyOn(process, "exit").mockImplementation((code) => {
        throw new Error(`process.exit(${code})`);
      });
      const program = buildProgram();
      await expect(
        program.parseAsync(["node", "tt", "auth", "login", "bad_key_here_x".repeat(4)]),
      ).rejects.toThrow();
    });

    it("rejects keys with wrong length", async () => {
      vi.spyOn(console, "error").mockImplementation(() => {});
      vi.spyOn(process, "exit").mockImplementation((code) => {
        throw new Error(`process.exit(${code})`);
      });
      const program = buildProgram();
      await expect(
        program.parseAsync(["node", "tt", "auth", "login", "tt_tooshort"]),
      ).rejects.toThrow();
    });
  });

  describe("auth logout", () => {
    it("clears stored credentials", async () => {
      vi.spyOn(console, "log").mockImplementation(() => {});
      config.writeConfig({ api_key: VALID_KEY });
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "auth", "logout"]);
      expect(config.readConfig()).toEqual({});
    });
  });

  describe("auth status", () => {
    it("shows authenticated state when key exists", async () => {
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      config.writeConfig({ api_key: VALID_KEY });
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "auth", "status"]);
      const allOutput = spy.mock.calls.map((c) => String(c[0])).join("\n");
      expect(allOutput).toContain("Yes");
      expect(allOutput).toContain(VALID_KEY.slice(0, 8));
    });

    it("shows unauthenticated state when no key", async () => {
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      const program = buildProgram();
      await program.parseAsync(["node", "tt", "auth", "status"]);
      const allOutput = spy.mock.calls.map((c) => String(c[0])).join("\n");
      expect(allOutput).toContain("tt auth login");
    });
  });
});
