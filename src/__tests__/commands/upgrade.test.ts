import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Command } from "commander";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { execFileSync } from "node:child_process";
import { registerUpgradeCommand, resolveGlobalInstall, type UpgradeCommandDeps } from "../../commands/upgrade.js";
import { setJsonMode } from "../../output.js";

let root: string;
let prefix: string;
let packageDir: string;
let entrypoint: string;
let logs: string[];

function registry(version: string) {
  return vi.fn(async () => new Response(JSON.stringify({ version }), { status: 200 }));
}

async function run(args: string[], deps: Partial<UpgradeCommandDeps> = {}) {
  const program = new Command().exitOverride();
  registerUpgradeCommand(program, {
    version: "0.21.0",
    entrypoint,
    env: { TUNED_TENSOR_HOME: join(root, "home") },
    platform: process.platform,
    ...deps,
  });
  await program.parseAsync(["node", "tt", "upgrade", ...args]);
}

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "tt-upgrade-"));
  prefix = join(root, "prefix");
  packageDir = join(prefix, ...(process.platform === "win32" ? [] : ["lib"]), "node_modules", "@tuned-tensor", "cli");
  mkdirSync(join(packageDir, "dist"), { recursive: true });
  entrypoint = join(packageDir, "dist", "index.js");
  writeFileSync(entrypoint, "");
  writeFileSync(join(packageDir, "package.json"), JSON.stringify({ version: "0.21.0" }));
  logs = [];
  vi.spyOn(console, "log").mockImplementation((line: string) => { logs.push(String(line)); });
});

afterEach(() => {
  setJsonMode(false);
  vi.restoreAllMocks();
  rmSync(root, { recursive: true, force: true });
});

describe("tt upgrade", () => {
  it("upgrades into the prefix that owns the running tt", async () => {
    const runNpm = vi.fn(async () => {
      writeFileSync(join(packageDir, "package.json"), JSON.stringify({ version: "0.22.0" }));
      return 0;
    });

    await run([], { fetchImpl: registry("0.22.0"), runNpm });

    expect(runNpm).toHaveBeenCalledWith([
      "install", "-g", "--ignore-scripts", "--prefix", prefix, "@tuned-tensor/cli@latest",
    ]);
    expect(logs.join("\n")).toContain("Upgraded tt 0.21.0 → 0.22.0");
    expect(JSON.parse(readFileSync(join(root, "home", "update-check.json"), "utf8")).latestVersion)
      .toBe("0.22.0");
  });

  it("does nothing when already on the latest release", async () => {
    const runNpm = vi.fn(async () => 0);
    await run([], { version: "0.22.0", fetchImpl: registry("0.22.0"), runNpm });
    expect(runNpm).not.toHaveBeenCalled();
    expect(logs.join("\n")).toContain("tt 0.22.0 is the latest version");
  });

  it("only reports with --check", async () => {
    const runNpm = vi.fn(async () => 0);
    setJsonMode(true);
    await run(["--check"], { fetchImpl: registry("0.22.0"), runNpm });
    expect(runNpm).not.toHaveBeenCalled();
    expect(JSON.parse(logs.join("\n"))).toEqual({
      current_version: "0.21.0",
      latest_version: "0.22.0",
      update_available: true,
    });
  });

  it("installs an explicit version or dist-tag without asking the registry", async () => {
    const fetchImpl = registry("0.22.0");
    const runNpm = vi.fn(async (_args: string[]) => 0);
    await run(["beta"], { fetchImpl, runNpm });
    expect(fetchImpl).not.toHaveBeenCalled();
    expect(runNpm.mock.calls[0]![0]).toContain("@tuned-tensor/cli@beta");
  });

  it("explains how to upgrade when tt was not installed with npm -g", async () => {
    const runNpm = vi.fn(async () => 0);
    const checkout = join(root, "checkout", "dist", "index.js");
    mkdirSync(join(root, "checkout", "dist"), { recursive: true });
    writeFileSync(checkout, "");
    await expect(run([], { entrypoint: checkout, fetchImpl: registry("0.22.0"), runNpm }))
      .rejects.toThrow("npm install -g --ignore-scripts @tuned-tensor/cli@latest");
    expect(runNpm).not.toHaveBeenCalled();
  });

  it("keeps installer-managed files out of npm upgrades", async () => {
    writeFileSync(join(prefix, "tt-install.json"), JSON.stringify({ manager: "windows-installer" }));
    const runNpm = vi.fn(async () => 0);
    const fetchImpl = registry("0.22.0");
    await expect(run([], { runNpm, fetchImpl })).rejects.toThrow("winget upgrade --id TunedTensor.CLI");
    expect(runNpm).not.toHaveBeenCalled();
    expect(fetchImpl).not.toHaveBeenCalled();
    await run(["--check"], { runNpm, fetchImpl });
    expect(fetchImpl).toHaveBeenCalled();
    expect(runNpm).not.toHaveBeenCalled();
  });

  it.each(["latest & whoami", "--help", "https://example.com/pkg.tgz", "%PATH%", "beta|echo"])("rejects non-version input %s", async (target) => {
    const runNpm = vi.fn(async () => 0);
    await expect(run(["--", target], { runNpm })).rejects.toThrow("exact version or npm dist-tag");
    expect(runNpm).not.toHaveBeenCalled();
  });

  it("reports npm failures with the command to retry", async () => {
    const npm = process.platform === "win32" ? "npm.cmd" : "npm";
    await expect(run([], { fetchImpl: registry("0.22.0"), runNpm: async () => 1 }))
      .rejects.toThrow(`${npm} install -g --ignore-scripts --prefix`);
  });

  it.skipIf(process.platform === "win32").each([
    { directory: "plain", escaped: "plain", quoted: false },
    { directory: "Team Tools", escaped: "Team Tools", quoted: true },
    { directory: "O'Brien", escaped: "O'\\''Brien", quoted: true },
  ])("preserves the POSIX retry command prefix for $directory", async ({ directory, escaped, quoted }) => {
    prefix = join(root, directory);
    packageDir = join(prefix, "lib", "node_modules", "@tuned-tensor", "cli");
    mkdirSync(join(packageDir, "dist"), { recursive: true });
    entrypoint = join(packageDir, "dist", "index.js");
    writeFileSync(entrypoint, "");
    const runNpm = vi.fn(async () => 1);
    let failure: unknown;
    try {
      await run(["beta"], { platform: "linux", runNpm });
    } catch (error) {
      failure = error;
    }
    expect(failure).toBeInstanceOf(Error);
    const retryCommand = (failure as Error).message.split("\n  ")[1]!;
    const renderedPrefix = quoted ? `'${join(root, escaped)}'` : prefix;
    expect(retryCommand).toBe(
      `npm install -g --ignore-scripts --prefix ${renderedPrefix} @tuned-tensor/cli@beta`,
    );
    const expectedArgs = ["install", "-g", "--ignore-scripts", "--prefix", prefix, "@tuned-tensor/cli@beta"];
    expect(runNpm).toHaveBeenCalledWith(expectedArgs);
    // Parse ordinary path fixtures with a real shell; never invoke npm or sudo.
    const parsedArgs = execFileSync("sh", ["-c", `set -- ${retryCommand}; printf '%s\\n' "$@"`], {
      encoding: "utf8",
    }).trimEnd().split("\n");
    expect(parsedArgs).toEqual(["npm", ...expectedArgs]);
  });

  it("fails clearly when the registry is unreachable", async () => {
    await expect(run([], { fetchImpl: async () => { throw new Error("offline"); } }))
      .rejects.toThrow("Could not reach the npm registry");
  });
});

describe("resolveGlobalInstall", () => {
  it("recognizes global npm layouts and rejects npx caches", () => {
    expect(resolveGlobalInstall(entrypoint)).toEqual({ prefix, packageDir });
    expect(resolveGlobalInstall(undefined)).toBeNull();
    expect(resolveGlobalInstall(join(root, "missing.js"))).toBeNull();

    const npx = join(root, "_npx", "abc", "lib", "node_modules", "@tuned-tensor", "cli", "dist");
    mkdirSync(npx, { recursive: true });
    writeFileSync(join(npx, "index.js"), "");
    expect(resolveGlobalInstall(join(npx, "index.js"))).toBeNull();
  });
});
