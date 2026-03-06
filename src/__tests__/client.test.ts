import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { get, post, put, patch, del, upload, ApiError } from "../client.js";

const FAKE_KEY = "tt_" + "a".repeat(48);
const EMPTY_CONFIG_DIR = join(tmpdir(), `tt-test-client-${process.pid}`);

function jsonResponse(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function errorResponse(code: string, message: string, status: number) {
  return new Response(JSON.stringify({ error: { code, message } }), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

beforeEach(() => {
  process.env.XDG_CONFIG_HOME = EMPTY_CONFIG_DIR;
  process.env.TUNED_TENSOR_API_KEY = FAKE_KEY;
  process.env.TUNED_TENSOR_URL = "https://test.tunedtensor.com";
});

afterEach(() => {
  delete process.env.XDG_CONFIG_HOME;
  delete process.env.TUNED_TENSOR_API_KEY;
  delete process.env.TUNED_TENSOR_URL;
});

describe("client", () => {
  describe("get", () => {
    it("sends GET request with auth header", async () => {
      const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
        jsonResponse({ data: [1, 2, 3] }),
      );
      const result = await get("/items");
      expect(fetchSpy).toHaveBeenCalledTimes(1);
      const [url, init] = fetchSpy.mock.calls[0];
      expect(url).toBe("https://test.tunedtensor.com/api/v1/items");
      expect(init?.method).toBe("GET");
      expect(init?.headers).toHaveProperty("Authorization", `Bearer ${FAKE_KEY}`);
      expect(result).toEqual({ data: [1, 2, 3] });
    });

    it("includes query parameters", async () => {
      const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
        jsonResponse({ data: [] }),
      );
      await get("/items", { page: 2, per_page: 10 });
      const url = fetchSpy.mock.calls[0][0] as string;
      expect(url).toContain("page=2");
      expect(url).toContain("per_page=10");
    });

    it("skips undefined query values", async () => {
      const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
        jsonResponse({ data: [] }),
      );
      await get("/items", { page: 1, filter: undefined });
      const url = fetchSpy.mock.calls[0][0] as string;
      expect(url).toContain("page=1");
      expect(url).not.toContain("filter");
    });
  });

  describe("post", () => {
    it("sends POST with JSON body", async () => {
      const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
        jsonResponse({ data: { id: "abc" } }),
      );
      await post("/items", { name: "test" });
      const [, init] = fetchSpy.mock.calls[0];
      expect(init?.method).toBe("POST");
      expect(init?.headers).toHaveProperty("Content-Type", "application/json");
      expect(init?.body).toBe(JSON.stringify({ name: "test" }));
    });
  });

  describe("put", () => {
    it("sends PUT request", async () => {
      const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
        jsonResponse({ data: {} }),
      );
      await put("/items/1", { name: "updated" });
      expect(fetchSpy.mock.calls[0][1]?.method).toBe("PUT");
    });
  });

  describe("patch", () => {
    it("sends PATCH request", async () => {
      const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
        jsonResponse({ data: {} }),
      );
      await patch("/items/1", { name: "patched" });
      expect(fetchSpy.mock.calls[0][1]?.method).toBe("PATCH");
    });
  });

  describe("del", () => {
    it("sends DELETE request and handles 204", async () => {
      vi.spyOn(globalThis, "fetch").mockResolvedValue(
        new Response(null, { status: 204 }),
      );
      const result = await del("/items/1");
      expect(result).toEqual({ data: null });
    });
  });

  describe("error handling", () => {
    it("throws ApiError for non-ok responses with JSON body", async () => {
      vi.spyOn(globalThis, "fetch").mockImplementation(() =>
        Promise.resolve(errorResponse("NOT_FOUND", "Item not found", 404)),
      );
      try {
        await get("/items/bad");
        expect.unreachable("should have thrown");
      } catch (e) {
        expect(e).toBeInstanceOf(ApiError);
        const err = e as ApiError;
        expect(err.status).toBe(404);
        expect(err.code).toBe("NOT_FOUND");
        expect(err.message).toBe("Item not found");
      }
    });

    it("throws ApiError with UNKNOWN for non-JSON error responses", async () => {
      vi.spyOn(globalThis, "fetch").mockResolvedValue(
        new Response("Internal error", {
          status: 500,
          statusText: "Internal Server Error",
        }),
      );
      await expect(get("/items")).rejects.toThrow(ApiError);
    });

    it("throws when no API key is available", async () => {
      delete process.env.TUNED_TENSOR_API_KEY;
      await expect(get("/items")).rejects.toThrow("No API key found");
    });
  });

  describe("opts override", () => {
    it("uses opts.baseUrl over env var", async () => {
      const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
        jsonResponse({ data: {} }),
      );
      await get("/items", undefined, { baseUrl: "https://custom.com" });
      const url = fetchSpy.mock.calls[0][0] as string;
      expect(url).toContain("https://custom.com/api/v1/items");
    });

    it("uses opts.apiKey over env var", async () => {
      const customKey = "tt_" + "b".repeat(48);
      const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
        jsonResponse({ data: {} }),
      );
      await get("/items", undefined, { apiKey: customKey });
      expect(fetchSpy.mock.calls[0][1]?.headers).toHaveProperty(
        "Authorization",
        `Bearer ${customKey}`,
      );
    });
  });
});
