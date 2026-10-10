import { spawn } from "node:child_process";
import { accessSync, constants, existsSync, readFileSync, realpathSync } from "node:fs";
import { dirname, join } from "node:path";
import { Command } from "commander";
import chalk from "chalk";
import { isJsonMode, printJson, printSuccess } from "../output.js";
import {
  CLI_PACKAGE_NAME,
  compareCliVersions,
  fetchLatestCliVersion,
  getCliUpdateCacheFile,
  recordLatestCliVersion,
} from "../update-check.js";

export type NpmRunner = (args: string[]) => Promise<number>;

export interface UpgradeCommandDeps {
  version: string;
  /** Path of the running `tt` script (process.argv[1]). */
  entrypoint?: string;
  env: NodeJS.ProcessEnv;
  platform?: NodeJS.Platform;
  fetchImpl?: typeof fetch;
  runNpm?: NpmRunner;
}

export interface GlobalInstall {
  /** npm prefix that owns the running install. */
  prefix: string;
  /** Directory holding the installed package.json. */
  packageDir: string;
}

/**
 * Finds the npm global prefix that owns the running `tt`, so upgrades replace
 * the binary on PATH instead of installing a second copy somewhere else (the
 * curl installer falls back to ~/.local when npm's default prefix is not
 * writable). Returns null for checkouts, npx caches and other package managers.
 */
export function resolveGlobalInstall(
  entrypoint: string | undefined,
  platform: NodeJS.Platform = process.platform,
): GlobalInstall | null {
  if (!entrypoint) return null;
  let script: string;
  try {
    script = realpathSync(entrypoint);
  } catch {
    return null;
  }
  if (/[\\/]_npx[\\/]/.test(script)) return null;
  const match = platform === "win32"
    ? script.match(/^(.*)\\node_modules\\@tuned-tensor\\cli\\dist\\index\.js$/i)
    : script.match(/^(.*)\/lib\/node_modules\/@tuned-tensor\/cli\/dist\/index\.js$/);
  if (!match?.[1]) return null;
  return { prefix: match[1], packageDir: dirname(dirname(script)) };
}

function prefixIsWritable(prefix: string, platform: NodeJS.Platform): boolean {
  const modules = platform === "win32" ? join(prefix, "node_modules") : join(prefix, "lib", "node_modules");
  try {
    accessSync(existsSync(modules) ? modules : prefix, constants.W_OK);
    return true;
  } catch {
    return false;
  }
}

function installedVersion(packageDir: string): string | undefined {
  try {
    const manifest = JSON.parse(readFileSync(join(packageDir, "package.json"), "utf8")) as { version?: unknown };
    return typeof manifest.version === "string" ? manifest.version : undefined;
  } catch {
    return undefined;
  }
}

function quoteForCmd(arg: string): string {
  return /[\s"&|<>^]/.test(arg) ? `"${arg.replace(/"/g, '""')}"` : arg;
}

function defaultNpmRunner(platform: NodeJS.Platform): NpmRunner {
  return async (args) => await new Promise<number>((resolve, reject) => {
    const windows = platform === "win32";
    // npm.cmd can only be spawned through a shell on Windows.
    const child = spawn(
      windows ? "npm.cmd" : "npm",
      windows ? args.map(quoteForCmd) : args,
      {
        shell: windows,
        // Keep stdout clean for --json; npm progress goes to stderr.
        stdio: ["ignore", isJsonMode() ? 2 : "inherit", "inherit"],
      },
    );
    child.once("error", reject);
    child.once("close", (code) => resolve(code ?? 1));
  });
}

function formatCommand(args: string[]): string {
  return ["npm", ...args].map((arg) => (/\s/.test(arg) ? JSON.stringify(arg) : arg)).join(" ");
}

export function registerUpgradeCommand(program: Command, deps: UpgradeCommandDeps): void {
  program
    .command("upgrade")
    .alias("update")
    .description("Upgrade tt to the latest release")
    .argument("[version]", "Version or npm dist-tag to install (default: latest)")
    .option("--check", "Only report whether an update is available")
    .addHelpText(
      "after",
      `
Examples:
  tt upgrade             Install the latest stable release
  tt upgrade --check     Report whether a newer release exists
  tt upgrade beta        Opt in to the beta channel
  tt upgrade 0.21.0      Install a specific version

Set TT_NO_UPDATE_CHECK=1 to silence update notices.
`,
    )
    .action(async (requested: string | undefined, options: { check?: boolean }) => {
      const platform = deps.platform ?? process.platform;
      const cacheFile = getCliUpdateCacheFile(deps.env);
      const current = deps.version;
      const target = requested?.trim() || "latest";

      let latest: string | undefined;
      if (target === "latest") {
        latest = await fetchLatestCliVersion({ fetchImpl: deps.fetchImpl, timeoutMs: 10_000 }) ?? undefined;
        if (!latest) {
          throw new Error(
            "Could not reach the npm registry to look up the latest tt release. Check your connection and try again.",
          );
        }
        recordLatestCliVersion(cacheFile, latest);
        const comparison = compareCliVersions(latest, current);
        if (comparison !== null && comparison <= 0) {
          if (isJsonMode()) {
            printJson({ current_version: current, latest_version: latest, update_available: false });
          } else {
            printSuccess(`tt ${current} is the latest version.`);
          }
          return;
        }
      }

      if (options.check) {
        if (isJsonMode()) {
          printJson({ current_version: current, latest_version: latest ?? null, update_available: Boolean(latest) });
        } else if (latest) {
          console.log(`Update available: tt ${current} → ${chalk.bold(latest)}`);
          console.log(`Run ${chalk.bold("tt upgrade")} to install it.`);
        } else {
          console.log(`tt ${current} is installed. Run ${chalk.bold(`tt upgrade ${target}`)} to install ${target}.`);
        }
        return;
      }

      const install = resolveGlobalInstall(deps.entrypoint, platform);
      const packageSpec = `${CLI_PACKAGE_NAME}@${target}`;
      if (!install) {
        throw new Error(
          `This tt (${deps.entrypoint ?? "unknown path"}) was not installed with npm -g, so it can't upgrade itself.\n`
          + `Upgrade it the way you installed it, for example:\n  npm install -g --ignore-scripts ${packageSpec}`,
        );
      }
      const npmArgs = ["install", "-g", "--ignore-scripts", "--prefix", install.prefix, packageSpec];
      if (!prefixIsWritable(install.prefix, platform)) {
        throw new Error(
          `tt is installed in ${install.prefix}, which this user can't write to. Run:\n  sudo ${formatCommand(npmArgs)}`,
        );
      }

      if (!isJsonMode()) {
        console.log(`Upgrading tt ${current} → ${chalk.bold(latest ?? target)}`);
        console.log(chalk.dim(`$ ${formatCommand(npmArgs)}`));
      }
      let code: number;
      try {
        code = await (deps.runNpm ?? defaultNpmRunner(platform))(npmArgs);
      } catch (error) {
        throw new Error(
          `Could not run npm (${error instanceof Error ? error.message : String(error)}). Run it yourself:\n  ${formatCommand(npmArgs)}`,
        );
      }
      if (code !== 0) {
        throw new Error(`npm exited with code ${code}. Run it yourself to see details:\n  ${formatCommand(npmArgs)}`);
      }

      const installed = installedVersion(install.packageDir) ?? latest ?? target;
      if (isJsonMode()) {
        printJson({ previous_version: current, version: installed, prefix: install.prefix });
      } else {
        printSuccess(`Upgraded tt ${current} → ${installed}. Restart any open tt sessions to use it.`);
      }
    });
}
