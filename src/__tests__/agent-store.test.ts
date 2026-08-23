import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { mkdtempSync, readFileSync, readdirSync, rmSync, statSync } from "node:fs";
import { randomUUID } from "node:crypto";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { LocalAgentStore } from "../agent-store.js";

const THREAD_ID = "251f122f-dd8e-4894-a0ab-99965e976e29";
const ACTION_ID = "11d31fae-70c6-4fbf-98e3-037cb16c150f";
let root: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "tt-agent-store-"));
});

afterEach(() => rmSync(root, { recursive: true, force: true }));

describe("local agent store", () => {
  it("atomically persists resumable transcript/action state with private modes", async () => {
    const store = new LocalAgentStore(root, { secretValues: ["tt_super_secret"] });
    await store.save({
      thread: {
        id: THREAD_ID,
        title: "Inspect runs",
        status: "active",
        last_message_at: "2026-08-09T10:00:00.000Z",
        created_at: "2026-08-09T10:00:00.000Z",
        updated_at: "2026-08-09T10:00:00.000Z",
      },
      messages: [{ role: "user", content: "do not leak tt_super_secret" }],
      actions: [{ id: "action-1", title: "Cancel", summary: "", risk: "medium" }],
    });

    expect(await store.load(THREAD_ID)).toMatchObject({
      thread: { id: THREAD_ID },
      actions: [{ id: "action-1" }],
    });
    const file = join(root, "threads", `${THREAD_ID}.json`);
    expect(readFileSync(file, "utf8")).not.toContain("tt_super_secret");
    if (process.platform !== "win32") {
      expect(statSync(root).mode & 0o777).toBe(0o700);
      expect(statSync(join(root, "threads")).mode & 0o777).toBe(0o700);
      expect(statSync(file).mode & 0o777).toBe(0o600);
    }
  });

  it("rejects unsafe thread IDs before resolving a path", async () => {
    const store = new LocalAgentStore(root);
    await expect(store.load("../../config.json")).rejects.toThrow(/thread id/i);
  });

  it("atomically claims an action only once across store instances", async () => {
    const first = new LocalAgentStore(root);
    const second = new LocalAgentStore(root);
    await expect(first.claimAction(ACTION_ID)).resolves.toBeUndefined();
    await expect(second.claimAction(ACTION_ID)).rejects.toThrow(/already.*settled/i);
  });

  it("bounds transcript bytes, retained threads, and terminal action claims", async () => {
    const store = new LocalAgentStore(root, {
      maxThreadBytes: 4_000,
      maxMessages: 5,
      maxThreads: 2,
      maxClaims: 2,
    });
    const ids = [randomUUID(), randomUUID(), randomUUID()];
    for (const [index, id] of ids.entries()) {
      await store.save({
        thread: {
          id,
          title: `Thread ${index}`,
          status: "active",
          last_message_at: null,
          created_at: `2026-08-09T10:00:0${index}.000Z`,
          updated_at: `2026-08-09T10:00:0${index}.000Z`,
        },
        messages: Array.from({ length: 10 }, (_, message) => ({
          role: "assistant",
          content: `${message}:${"x".repeat(900)}`,
        })),
        actions: index === 0 ? [{
          id: randomUUID(),
          title: "Inspect uncertain create",
          summary: "Remote outcome must remain inspectable",
          risk: "medium",
          status: "outcome_unknown",
        }] : [],
      });
    }

    const retained = await store.list();
    expect(retained).toHaveLength(2);
    expect(retained.map((state) => state.thread.id)).toContain(ids[0]);
    const latest = await store.load(ids[2]!);
    expect(latest.messages.length).toBeLessThanOrEqual(5);
    expect(statSync(join(root, "threads", `${ids[2]}.json`)).size).toBeLessThanOrEqual(4_000);

    const actionIds = [randomUUID(), randomUUID(), randomUUID()];
    for (const id of actionIds) await store.claimAction(id);
    expect(readdirSync(join(root, "action-claims"))).toHaveLength(2);
  });

  it("re-reads provider secrets on each persist so later login keys are redacted", async () => {
    const stored = ["sk-login-old-secret"];
    const store = new LocalAgentStore(root, {
      secretValueProvider: () => stored,
    });
    const thread = {
      id: THREAD_ID,
      title: "Inspect runs",
      status: "active" as const,
      last_message_at: "2026-08-09T10:00:00.000Z",
      created_at: "2026-08-09T10:00:00.000Z",
      updated_at: "2026-08-09T10:00:00.000Z",
    };
    await store.save({
      thread,
      messages: [{ role: "user", content: "old sk-login-old-secret new sk-login-new-secret" }],
      actions: [],
    });
    const file = join(root, "threads", `${THREAD_ID}.json`);
    expect(readFileSync(file, "utf8")).not.toContain("sk-login-old-secret");
    expect(readFileSync(file, "utf8")).toContain("sk-login-new-secret");

    stored.push("sk-login-new-secret");
    await store.save({
      thread,
      messages: [{ role: "user", content: "old sk-login-old-secret new sk-login-new-secret" }],
      actions: [],
    });
    expect(readFileSync(file, "utf8")).not.toContain("sk-login-old-secret");
    expect(readFileSync(file, "utf8")).not.toContain("sk-login-new-secret");
    expect(readFileSync(file, "utf8")).toContain("[REDACTED]");
  });
});
