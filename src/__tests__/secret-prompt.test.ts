import { EventEmitter } from "node:events";
import { describe, expect, it } from "vitest";
import { withDetachedKeypress } from "../secret-prompt.js";

describe("withDetachedKeypress", () => {
  it("keeps the parent keypress listener from seeing keys during the nested prompt", async () => {
    const input = new EventEmitter();
    const seen: string[] = [];
    input.on("keypress", () => seen.push("parent"));

    await withDetachedKeypress(input, async () => {
      input.on("keypress", () => seen.push("nested"));
      input.emit("keypress", "o", { name: "o" });
      input.emit("keypress", "p", { name: "p" });
    });

    expect(seen).toEqual(["nested", "nested"]);

    input.emit("keypress", "x", { name: "x" });
    expect(seen).toEqual(["nested", "nested", "parent"]);
  });

  it("restores parent listeners after a nested prompt failure", async () => {
    const input = new EventEmitter();
    const seen: string[] = [];
    input.on("keypress", () => seen.push("parent"));

    await expect(
      withDetachedKeypress(input, async () => {
        throw new Error("cancelled");
      }),
    ).rejects.toThrow("cancelled");

    input.emit("keypress", "x", { name: "x" });
    expect(seen).toEqual(["parent"]);
  });
});
