const PACKAGE_LATEST_URL =
  "https://registry.npmjs.org/@tuned-tensor%2fcli/latest";

export interface CliUpdate {
  currentVersion: string;
  latestVersion: string;
}

export interface CliUpdateCheckOptions {
  fetchImpl?: typeof fetch;
  timeoutMs?: number;
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

export async function checkForCliUpdate(
  currentVersion: string,
  options: CliUpdateCheckOptions = {},
): Promise<CliUpdate | null> {
  const current = parseVersion(currentVersion);
  if (!current) return null;

  const controller = new AbortController();
  const timeoutMs = options.timeoutMs ?? 750;
  let timeout: ReturnType<typeof setTimeout> | undefined;
  const request = (async (): Promise<CliUpdate | null> => {
    try {
      const response = await (options.fetchImpl ?? fetch)(PACKAGE_LATEST_URL, {
        headers: { accept: "application/json" },
        signal: controller.signal,
      });
      if (!response.ok) return null;
      const payload = await response.json() as { version?: unknown };
      if (typeof payload.version !== "string") return null;
      const latest = parseVersion(payload.version);
      if (
        !latest ||
        latest.prerelease.length > 0 ||
        compareVersions(latest, current) <= 0
      ) return null;
      return {
        currentVersion,
        latestVersion: payload.version,
      };
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
  }
}

export function formatCliUpdateNotice(update: CliUpdate): string {
  return [
    `Update available: tt ${update.currentVersion} → ${update.latestVersion}`,
    "Upgrade with: npm install -g @tuned-tensor/cli@latest",
  ].join("\n");
}
