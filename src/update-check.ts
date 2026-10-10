import { spawn } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import chalk from "chalk";
import { getTunedTensorHome } from "./paths.js";

export const CLI_PACKAGE_NAME = "@tuned-tensor/cli";
const PACKAGE_LATEST_URL =
  "https://registry.npmjs.org/@tuned-tensor%2fcli/latest";
const DEFAULT_CACHE_MAX_AGE_MS = 12 * 60 * 60 * 1000;

export interface CliUpdate {
  currentVersion: string;
  latestVersion: string;
}

export interface CliUpdateCheckOptions {
  fetchImpl?: typeof fetch;
  timeoutMs?: number;
  /** Remembers the last registry answer so slow networks still see updates. */
  cacheFile?: string;
  cacheMaxAgeMs?: number;
  now?: () => number;
  signal?: AbortSignal;
}

interface UpdateCache {
  checkedAt: number;
  latestVersion: string;
}

interface ParsedVersion {
  major: bigint;
  minor: bigint;
  patch: bigint;
  prerelease: string[];
}

function parseVersion(version: string): ParsedVersion | null {
  const match = version.trim().match(
    /^v?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$/,
  );
  if (!match) return null;
  const prerelease = match[4]?.split(".") ?? [];
  if (prerelease.some((part) => /^0\d+$/.test(part))) return null;
  return {
    major: BigInt(match[1]),
    minor: BigInt(match[2]),
    patch: BigInt(match[3]),
    prerelease,
  };
}

function comparePrerelease(left: string[], right: string[]): number {
  if (left.length === 0 && right.length === 0) return 0;
  if (left.length === 0) return 1;
  if (right.length === 0) return -1;

  for (let index = 0; index < Math.max(left.length, right.length); index += 1) {
    const leftPart = left[index];
    const rightPart = right[index];
    if (leftPart === undefined) return -1;
    if (rightPart === undefined) return 1;
    if (leftPart === rightPart) continue;

    const leftNumeric = /^\d+$/.test(leftPart);
    const rightNumeric = /^\d+$/.test(rightPart);
    if (leftNumeric && rightNumeric) {
      if (leftPart.length !== rightPart.length) {
        return leftPart.length > rightPart.length ? 1 : -1;
      }
      return leftPart > rightPart ? 1 : -1;
    }
    if (leftNumeric !== rightNumeric) return leftNumeric ? -1 : 1;
    return leftPart > rightPart ? 1 : -1;
  }
  return 0;
}

function compareVersions(left: ParsedVersion, right: ParsedVersion): number {
  for (const key of ["major", "minor", "patch"] as const) {
    if (left[key] !== right[key]) return left[key] > right[key] ? 1 : -1;
  }
  return comparePrerelease(left.prerelease, right.prerelease);
}

export function getCliUpdateCacheFile(
  env: NodeJS.ProcessEnv = process.env,
): string {
  return join(getTunedTensorHome(env), "update-check.json");
}

/** Update checks are advisory; CI and explicit opt-outs never contact npm. */
export function cliUpdateChecksDisabled(
  env: NodeJS.ProcessEnv = process.env,
): boolean {
  return [env.TT_NO_UPDATE_CHECK, env.NO_UPDATE_NOTIFIER, env.CI]
    .some((value) => value !== undefined && value !== "" && value !== "0" && value !== "false");
}

function readUpdateCache(file: string): UpdateCache | null {
  try {
    const payload = JSON.parse(readFileSync(file, "utf8")) as Partial<UpdateCache>;
    if (
      typeof payload.checkedAt !== "number" ||
      typeof payload.latestVersion !== "string" ||
      !parseVersion(payload.latestVersion)
    ) return null;
    return { checkedAt: payload.checkedAt, latestVersion: payload.latestVersion };
  } catch {
    return null;
  }
}

function writeUpdateCache(file: string, cache: UpdateCache): void {
  try {
    mkdirSync(dirname(file), { recursive: true });
    writeFileSync(file, `${JSON.stringify(cache)}\n`);
  } catch {
    // A read-only home directory only costs the cache, not the check.
  }
}

export function recordLatestCliVersion(
  cacheFile: string,
  latestVersion: string,
  now: number = Date.now(),
): void {
  writeUpdateCache(cacheFile, { checkedAt: now, latestVersion });
}

export function compareCliVersions(left: string, right: string): number | null {
  const parsedLeft = parseVersion(left);
  const parsedRight = parseVersion(right);
  if (!parsedLeft || !parsedRight) return null;
  return compareVersions(parsedLeft, parsedRight);
}

function newerStableRelease(
  currentVersion: string,
  latestVersion: string | undefined,
): CliUpdate | null {
  if (!latestVersion) return null;
  const current = parseVersion(currentVersion);
  const latest = parseVersion(latestVersion);
  if (
    !current ||
    !latest ||
    latest.prerelease.length > 0 ||
    compareVersions(latest, current) <= 0
  ) return null;
  return { currentVersion, latestVersion };
}

/** Returns the newest stable version on npm, or null when it is unreachable. */
export async function fetchLatestCliVersion(
  options: Pick<CliUpdateCheckOptions, "fetchImpl" | "timeoutMs" | "signal"> = {},
): Promise<string | null> {
  const controller = new AbortController();
  const abort = () => controller.abort();
  options.signal?.addEventListener("abort", abort, { once: true });
  const timeoutMs = options.timeoutMs ?? 1500;
  let timeout: ReturnType<typeof setTimeout> | undefined;
  const request = (async (): Promise<string | null> => {
    try {
      const response = await (options.fetchImpl ?? fetch)(PACKAGE_LATEST_URL, {
        headers: { accept: "application/json" },
        signal: controller.signal,
      });
      if (!response.ok) return null;
      const payload = await response.json() as { version?: unknown };
      if (typeof payload.version !== "string") return null;
      const latest = parseVersion(payload.version);
      return latest && latest.prerelease.length === 0 ? payload.version : null;
    } catch {
      return null;
    }
  })();
  const deadline = new Promise<null>((resolve) => {
    timeout = setTimeout(() => {
      controller.abort();
      resolve(null);
    }, timeoutMs);
  });

  try {
    return await Promise.race([request, deadline]);
  } finally {
    if (timeout) clearTimeout(timeout);
    options.signal?.removeEventListener("abort", abort);
  }
}

/** Reads the last recorded registry answer without touching the network. */
export function cachedCliUpdate(
  currentVersion: string,
  cacheFile: string,
): CliUpdate | null {
  return newerStableRelease(currentVersion, readUpdateCache(cacheFile)?.latestVersion);
}

/** True when the cache is missing or older than the refresh interval. */
export function cliUpdateCacheIsStale(
  cacheFile: string,
  options: Pick<CliUpdateCheckOptions, "cacheMaxAgeMs" | "now"> = {},
): boolean {
  const cache = readUpdateCache(cacheFile);
  const now = (options.now ?? Date.now)();
  return !cache || now - cache.checkedAt >= (options.cacheMaxAgeMs ?? DEFAULT_CACHE_MAX_AGE_MS);
}

export async function refreshCliUpdateCache(
  cacheFile: string,
  options: CliUpdateCheckOptions = {},
): Promise<string | null> {
  const latest = await fetchLatestCliVersion(options);
  if (latest) recordLatestCliVersion(cacheFile, latest, (options.now ?? Date.now)());
  return latest;
}

// Runs in a detached `node -e` child: argv[1] is the URL, argv[2] the cache file.
const REFRESH_SCRIPT = `
const [url, file] = process.argv.slice(1);
const fs = require("node:fs");
const path = require("node:path");
fetch(url, { headers: { accept: "application/json" }, signal: AbortSignal.timeout(10000) })
  .then((response) => response.ok ? response.json() : null)
  .then((payload) => {
    if (!payload || typeof payload.version !== "string") return;
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, JSON.stringify({ checkedAt: Date.now(), latestVersion: payload.version }) + "\\n");
  })
  .catch(() => {});
`;

/**
 * Refreshes the cache from a detached process so commands never wait on npm
 * and short commands do not cut the request off when they exit.
 */
export function spawnCliUpdateRefresh(cacheFile: string): void {
  try {
    const child = spawn(
      process.execPath,
      ["-e", REFRESH_SCRIPT, PACKAGE_LATEST_URL, cacheFile],
      { detached: true, stdio: "ignore", windowsHide: true },
    );
    child.on("error", () => {});
    child.unref();
  } catch {
    // Advisory only.
  }
}

export async function checkForCliUpdate(
  currentVersion: string,
  options: CliUpdateCheckOptions = {},
): Promise<CliUpdate | null> {
  if (!parseVersion(currentVersion)) return null;
  const { cacheFile } = options;
  if (cacheFile && !cliUpdateCacheIsStale(cacheFile, options)) {
    return cachedCliUpdate(currentVersion, cacheFile);
  }
  const latest = cacheFile
    ? await refreshCliUpdateCache(cacheFile, options)
    : await fetchLatestCliVersion(options);
  // An unreachable registry falls back to the last answer it gave.
  return newerStableRelease(
    currentVersion,
    latest ?? (cacheFile ? readUpdateCache(cacheFile)?.latestVersion : undefined),
  );
}

export function formatCliUpdateNotice(update: CliUpdate): string {
  return [
    chalk.yellow(`Update available: tt ${update.currentVersion} → ${chalk.bold(update.latestVersion)}`),
    `Run ${chalk.bold("tt upgrade")} to install it.`,
  ].join("\n");
}
