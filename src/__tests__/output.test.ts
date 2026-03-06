import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  setJsonMode,
  isJsonMode,
  printJson,
  printSuccess,
  printWarning,
  printError,
  formatDate,
  formatStatus,
  truncate,
  shortId,
  printTable,
  printDetail,
} from "../output.js";

beforeEach(() => {
  setJsonMode(false);
});

describe("output", () => {
  describe("jsonMode", () => {
    it("defaults to false", () => {
      expect(isJsonMode()).toBe(false);
    });

    it("can be toggled on", () => {
      setJsonMode(true);
      expect(isJsonMode()).toBe(true);
    });
  });

  describe("printJson", () => {
    it("outputs formatted JSON to stdout", () => {
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      printJson({ a: 1 });
      expect(spy).toHaveBeenCalledWith(JSON.stringify({ a: 1 }, null, 2));
    });
  });

  describe("printSuccess", () => {
    it("logs a success message", () => {
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      printSuccess("done");
      expect(spy).toHaveBeenCalledTimes(1);
      const output = spy.mock.calls[0][0] as string;
      expect(output).toContain("done");
    });
  });

  describe("printWarning", () => {
    it("logs a warning message", () => {
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      printWarning("careful");
      expect(spy).toHaveBeenCalledTimes(1);
      const output = spy.mock.calls[0][0] as string;
      expect(output).toContain("careful");
    });
  });

  describe("printError", () => {
    it("logs to stderr", () => {
      const spy = vi.spyOn(console, "error").mockImplementation(() => {});
      printError("oops");
      expect(spy).toHaveBeenCalledTimes(1);
      const output = spy.mock.calls[0][0] as string;
      expect(output).toContain("oops");
    });
  });

  describe("printTable", () => {
    it("outputs a table to stdout", () => {
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      printTable(["Name", "Value"], [["foo", "bar"]]);
      expect(spy).toHaveBeenCalled();
    });

    it("shows pagination when meta is provided", () => {
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      printTable(["Name"], [["a"]], { page: 1, per_page: 10, total: 25 });
      const allOutput = spy.mock.calls.map((c) => c[0]).join("\n");
      expect(allOutput).toContain("Page 1/3");
      expect(allOutput).toContain("25 total");
    });
  });

  describe("printDetail", () => {
    it("outputs label-value pairs", () => {
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      printDetail([
        ["Name", "Alice"],
        ["Age", "30"],
      ]);
      expect(spy).toHaveBeenCalledTimes(2);
    });

    it("handles undefined values with a dash", () => {
      const spy = vi.spyOn(console, "log").mockImplementation(() => {});
      printDetail([["Name", undefined]]);
      const output = spy.mock.calls[0][0] as string;
      expect(output).toContain("—");
    });
  });

  describe("formatDate", () => {
    it("returns dash for null/undefined", () => {
      expect(formatDate(null)).toBe("—");
      expect(formatDate(undefined)).toBe("—");
    });

    it("formats valid ISO string", () => {
      const result = formatDate("2024-01-15T12:00:00Z");
      expect(result).toBeTruthy();
      expect(result).not.toBe("—");
    });
  });

  describe("formatStatus", () => {
    it("applies color to known statuses", () => {
      for (const status of [
        "completed",
        "running",
        "failed",
        "pending",
        "cancelled",
      ]) {
        const result = formatStatus(status);
        expect(result).toContain(status);
      }
    });

    it("handles unknown status", () => {
      const result = formatStatus("mystery");
      expect(result).toContain("mystery");
    });
  });

  describe("truncate", () => {
    it("returns string unchanged if within limit", () => {
      expect(truncate("short", 10)).toBe("short");
    });

    it("truncates and adds ellipsis", () => {
      expect(truncate("a very long string", 10)).toBe("a very lo…");
    });

    it("handles exact length", () => {
      expect(truncate("exact", 5)).toBe("exact");
    });
  });

  describe("shortId", () => {
    it("returns first 8 characters", () => {
      expect(shortId("abcdefghijklmnop")).toBe("abcdefgh");
    });
  });
});
