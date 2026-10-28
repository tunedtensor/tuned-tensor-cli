import { EventEmitter } from "node:events";
import { win32 } from "node:path";
import { Command } from "commander";
import { afterEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  spawn: vi.fn(), existsSync: vi.fn(), realpathSync: vi.fn(), accessSync: vi.fn(), readFileSync: vi.fn(),
}));
vi.mock("node:child_process", () => ({ spawn: mocks.spawn }));
vi.mock("node:fs", async (original) => ({ ...await original<typeof import("node:fs")>(), ...mocks }));

import { registerUpgradeCommand, resolveWindowsNpmCli } from "../commands/upgrade.js";

afterEach(() => vi.resetAllMocks());

describe("native Windows upgrades", () => {
  it("finds npm on a mixed-case PATH when the runtime has no adjacent npm", () => {
    const cli = "C:\\Program Files\\nodejs\\node_modules\\npm\\bin\\npm-cli.js";
    mocks.existsSync.mockImplementation(path => path === cli);
    expect(resolveWindowsNpmCli({ Path: ';"C:\\Program Files\\nodejs";C:\\other' }, "C:\\runtime\\node.exe")).toBe(cli);
    mocks.existsSync.mockReturnValue(false);
    expect(() => resolveWindowsNpmCli({}, "C:\\runtime\\node.exe")).toThrow("Could not locate npm-cli.js");
  });

  it("passes spaces and cmd metacharacters literally to Node with no shell", async () => {
    const prefix = "C:\\Users\\A & B\\%TEMP%\\npm";
    const entrypoint = win32.join(prefix, "node_modules", "@tuned-tensor", "cli", "dist", "index.js");
    const npmCli = "C:\\Program Files\\nodejs\\node_modules\\npm\\bin\\npm-cli.js";
    mocks.realpathSync.mockReturnValue(entrypoint);
    mocks.existsSync.mockImplementation(path => path === npmCli);
    mocks.readFileSync.mockImplementation(() => { throw new Error("absent"); });
    mocks.spawn.mockImplementation(() => {
      const child = new EventEmitter();
      queueMicrotask(() => child.emit("close", 0));
      return child;
    });
    vi.spyOn(console, "log").mockImplementation(() => undefined);
    const env = { Path: "C:\\Program Files\\nodejs" };
    const program = new Command().exitOverride();
    registerUpgradeCommand(program, { version: "0.26.2", entrypoint, env, platform: "win32" });
    await program.parseAsync(["node", "tt", "upgrade", "beta"]);
    expect(mocks.spawn).toHaveBeenCalledWith(process.execPath, [npmCli,
      "install", "-g", "--ignore-scripts", "--prefix", prefix, "@tuned-tensor/cli@beta",
    ], { env, shell: false, stdio: ["ignore", "inherit", "inherit"] });
  });

  it("prints a PowerShell retry command with literal Windows separators and quotes", async () => {
    const prefix = "C:\\Users\\A & B's files\\npm";
    const entrypoint = win32.join(prefix, "node_modules", "@tuned-tensor", "cli", "dist", "index.js");
    mocks.realpathSync.mockReturnValue(entrypoint);
    mocks.readFileSync.mockImplementation(() => { throw new Error("absent"); });
    vi.spyOn(console, "log").mockImplementation(() => undefined);
    const program = new Command().exitOverride();
    registerUpgradeCommand(program, {
      version: "0.26.2", entrypoint, env: {}, platform: "win32", runNpm: async () => 1,
    });
    await expect(program.parseAsync(["node", "tt", "upgrade", "beta"]))
      .rejects.toThrow("npm.cmd install -g --ignore-scripts --prefix 'C:\\Users\\A & B''s files\\npm' @tuned-tensor/cli@beta");
  });
});
