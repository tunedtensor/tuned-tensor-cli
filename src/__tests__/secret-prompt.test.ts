import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import { describe, expect, it } from "vitest";
import { keyChoiceFor, promptKeyChoice, withDetachedKeypress } from "../secret-prompt.js";

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

describe("keyChoiceFor", () => {
  it("maps y, n and escape, and ignores Enter", () => {
    expect(keyChoiceFor({ name: "y" })).toBe("approve");
    expect(keyChoiceFor({ name: "Y" })).toBe("approve");
    expect(keyChoiceFor({ name: "n" })).toBe("reject");
    expect(keyChoiceFor({ name: "escape" })).toBe("later");
    expect(keyChoiceFor({ name: "c", ctrl: true })).toBe("later");
    expect(keyChoiceFor({ name: "return" })).toBeNull();
    expect(keyChoiceFor({ name: "space" })).toBeNull();
  });
});

describe("promptKeyChoice", () => {
  function fakeTerminal() {
    const input = new PassThrough() as PassThrough & {
      isTTY: boolean;
      isRaw: boolean;
      setRawMode(mode: boolean): void;
    };
    input.isTTY = true;
    input.isRaw = true;
    input.setRawMode = (mode) => { input.isRaw = mode; };
    const written: string[] = [];
    const output = new PassThrough();
    output.on("data", (chunk) => written.push(String(chunk)));
    return { input, output, written };
  }

  it("waits for a decisive key and echoes the choice", async () => {
    const { input, output, written } = fakeTerminal();
    const parent: string[] = [];
    input.on("keypress", () => parent.push("parent"));

    const pending = promptKeyChoice("Save this spec edit?", input, output);
    input.emit("keypress", "\r", { name: "return" });
    input.emit("keypress", "y", { name: "y" });

    await expect(pending).resolves.toBe("approve");
    expect(parent).toEqual([]);
    expect(written.join("")).toContain("Save this spec edit? yes\n");
    expect(input.isRaw).toBe(true);
  });

  it("defers when the terminal is not interactive", async () => {
    const input = new PassThrough();
    await expect(promptKeyChoice("Approve?", input, new PassThrough())).resolves.toBe("later");
  });

  it("defers and restores key handlers when input closes during approval", async () => {
    const { input, output } = fakeTerminal();
    const parent = () => {};
    input.on("keypress", parent);

    const pending = promptKeyChoice("Approve?", input, output);
    input.emit("close");

    await expect(pending).resolves.toBe("later");
    expect(input.listeners("keypress")).toEqual([parent]);
    expect(input.isRaw).toBe(true);
  });

  it("restores raw mode even if the initial prompt write throws synchronously", async () => {
    const { input, output } = fakeTerminal();
    input.isRaw = false;
    const rawModeCalls: boolean[] = [];
    const originalSetRawMode = input.setRawMode;
    input.setRawMode = (mode) => {
      rawModeCalls.push(mode);
      originalSetRawMode(mode);
    };
    output.write = () => {
      throw new Error("EPIPE: write after destroyed stream");
    };

    await expect(promptKeyChoice("Approve?", input, output)).rejects.toThrow(
      "EPIPE: write after destroyed stream",
    );

    expect(rawModeCalls).toContain(true);
    expect(rawModeCalls).toContain(false);
    expect(input.isRaw).toBe(false);
  });
});
