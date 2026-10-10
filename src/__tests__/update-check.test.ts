import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { describe, expect, it, vi } from "vitest";
import {
  cachedCliUpdate,
  checkForCliUpdate,
  cliUpdateChecksDisabled,
  formatCliUpdateNotice,
} from "../update-check.js";

describe("CLI update checks", () => {
  it("recommends the latest stable release when it is newer", async () => {
    const fetchImpl = vi.fn(async () => new Response(
      JSON.stringify({ version: "0.11.0" }),
      { status: 200 },
    ));

    const update = await checkForCliUpdate("0.10.0", {
      fetchImpl,
      timeoutMs: 50,
    });

    expect(update).toEqual({ currentVersion: "0.10.0", latestVersion: "0.11.0" });
    expect(formatCliUpdateNotice(update!)).toContain(
      "tt upgrade",
    );
  });

  it("stays quiet for current, older, malformed, and unavailable registry responses", async () => {
    const responses = [
      new Response(JSON.stringify({ version: "0.10.0" }), { status: 200 }),
      new Response(JSON.stringify({ version: "0.9.9" }), { status: 200 }),
      new Response(JSON.stringify({ version: "0.11.0-beta.1" }), { status: 200 }),
      new Response(JSON.stringify({ version: "not-semver" }), { status: 200 }),
      new Response("unavailable", { status: 503 }),
    ];

    for (const response of responses) {
      expect(await checkForCliUpdate("0.10.0", {
        fetchImpl: async () => response,
        timeoutMs: 50,
      })).toBeNull();
    }

    expect(await checkForCliUpdate("0.10.0", {
      fetchImpl: async () => {
        throw new Error("offline");
      },
      timeoutMs: 50,
    })).toBeNull();
  });

  it("treats a stable release as newer than its prerelease", async () => {
    const update = await checkForCliUpdate("0.11.0-beta.1", {
      fetchImpl: async () => new Response(
        JSON.stringify({ version: "0.11.0" }),
        { status: 200 },
      ),
      timeoutMs: 50,
    });

    expect(update?.latestVersion).toBe("0.11.0");
  });

  it("enforces its own deadline when fetch ignores abort", async () => {
    const startedAt = Date.now();
    const update = await checkForCliUpdate("0.10.0", {
      fetchImpl: async () => await new Promise<Response>(() => {}),
      timeoutMs: 20,
    });

    expect(update).toBeNull();
    expect(Date.now() - startedAt).toBeLessThan(200);
  });

  it("compares arbitrarily large numeric identifiers exactly", async () => {
    const update = await checkForCliUpdate(
      "9007199254740992.0.0",
      {
        fetchImpl: async () => new Response(
          JSON.stringify({ version: "9007199254740993.0.0" }),
          { status: 200 },
        ),
        timeoutMs: 50,
      },
    );

    expect(update?.latestVersion).toBe("9007199254740993.0.0");
    expect(await checkForCliUpdate("1.0.0", {
      fetchImpl: async () => new Response(
        JSON.stringify({ version: "1.0.1-01" }),
        { status: 200 },
      ),
      timeoutMs: 50,
    })).toBeNull();
  });

  it("remembers the registry answer so offline and slow launches still see updates", async () => {
    const dir = mkdtempSync(join(tmpdir(), "tt-update-cache-"));
    const cacheFile = join(dir, "nested", "update-check.json");
    try {
      const fetchImpl = vi.fn(async () => new Response(
        JSON.stringify({ version: "0.11.0" }),
        { status: 200 },
      ));
      expect(await checkForCliUpdate("0.10.0", { fetchImpl, cacheFile, now: () => 1_000 }))
        .toEqual({ currentVersion: "0.10.0", latestVersion: "0.11.0" });
      expect(JSON.parse(readFileSync(cacheFile, "utf8"))).toEqual({ checkedAt: 1_000, latestVersion: "0.11.0" });

      // Fresh cache: no network round trip.
      expect((await checkForCliUpdate("0.10.0", { fetchImpl, cacheFile, now: () => 2_000 }))?.latestVersion)
        .toBe("0.11.0");
      expect(fetchImpl).toHaveBeenCalledTimes(1);

      // Stale cache and an unreachable registry fall back to the last answer.
      const offline = async () => { throw new Error("offline"); };
      expect((await checkForCliUpdate("0.10.0", {
        fetchImpl: offline,
        cacheFile,
        now: () => 1_000 + 13 * 60 * 60 * 1000,
        timeoutMs: 50,
      }))?.latestVersion).toBe("0.11.0");

      // Once upgraded, the cached release is no longer newer.
      expect(cachedCliUpdate("0.11.0", cacheFile)).toBeNull();
      writeFileSync(cacheFile, "not json");
      expect(cachedCliUpdate("0.10.0", cacheFile)).toBeNull();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("honors CI and explicit opt-outs", () => {
    expect(cliUpdateChecksDisabled({})).toBe(false);
    expect(cliUpdateChecksDisabled({ CI: "true" })).toBe(true);
    expect(cliUpdateChecksDisabled({ TT_NO_UPDATE_CHECK: "1" })).toBe(true);
    expect(cliUpdateChecksDisabled({ NO_UPDATE_NOTIFIER: "1" })).toBe(true);
    expect(cliUpdateChecksDisabled({ TT_NO_UPDATE_CHECK: "0" })).toBe(false);
  });
});
