import { describe, expect, it, vi } from "vitest";
import {
  checkForCliUpdate,
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
      "npm install -g @tuned-tensor/cli@latest",
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
});
