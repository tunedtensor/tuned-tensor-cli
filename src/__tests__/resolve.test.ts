import { describe, it, expect, vi, beforeEach } from "vitest";
import { resolveSpecId, resolveRunId, isFullUuid, ResolveError } from "../resolve.js";
import * as client from "../client.js";

vi.mock("../client.js", async (importOriginal) => {
  const actual = await importOriginal<typeof client>();
  return { ...actual, get: vi.fn() };
});

const SPEC_A = "11111111-1111-4111-8111-111111111111";
const SPEC_B = "1111aaaa-1111-4111-8111-111111111111";
const RUN_A = "33333333-3333-4333-8333-333333333333";

beforeEach(() => {
  vi.mocked(client.get).mockReset();
});

describe("isFullUuid", () => {
  it("recognizes a canonical v4 UUID", () => {
    expect(isFullUuid(SPEC_A)).toBe(true);
  });

  it("rejects shorter strings", () => {
    expect(isFullUuid("11111111")).toBe(false);
    expect(isFullUuid("11111111-1111")).toBe(false);
  });

  it("is case insensitive", () => {
    expect(isFullUuid(SPEC_A.toUpperCase())).toBe(true);
  });
});

describe("resolveSpecId", () => {
  it("returns full UUIDs unchanged without an API call", async () => {
    const result = await resolveSpecId(SPEC_A);
    expect(result).toBe(SPEC_A);
    expect(client.get).not.toHaveBeenCalled();
  });

  it("rejects prefixes shorter than 4 chars", async () => {
    await expect(resolveSpecId("abc")).rejects.toThrow(ResolveError);
    await expect(resolveSpecId("abc")).rejects.toThrow(/too short/);
    expect(client.get).not.toHaveBeenCalled();
  });

  it("resolves a unique 8-char prefix", async () => {
    vi.mocked(client.get).mockResolvedValue({
      data: [
        { id: SPEC_A, name: "Alpha" },
        { id: "22222222-2222-4222-8222-222222222222", name: "Beta" },
      ],
      meta: { page: 1, per_page: 100, total: 2 },
    });

    const result = await resolveSpecId(SPEC_A.slice(0, 8));
    expect(result).toBe(SPEC_A);
    expect(client.get).toHaveBeenCalledWith(
      "/behavior-specs",
      { page: 1, per_page: 100 },
      undefined,
    );
  });

  it("errors with helpful list when prefix is ambiguous", async () => {
    vi.mocked(client.get).mockResolvedValue({
      data: [
        { id: SPEC_A, name: "Alpha" },
        { id: SPEC_B, name: "Beta" },
      ],
      meta: { page: 1, per_page: 100, total: 2 },
    });

    await expect(resolveSpecId("1111")).rejects.toThrow(/ambiguous \(2 matches\)/);
    await expect(resolveSpecId("1111")).rejects.toThrow(/Alpha/);
    await expect(resolveSpecId("1111")).rejects.toThrow(/Beta/);
  });

  it("errors when no spec matches the prefix", async () => {
    vi.mocked(client.get).mockResolvedValue({
      data: [{ id: SPEC_A, name: "Alpha" }],
      meta: { page: 1, per_page: 100, total: 1 },
    });

    await expect(resolveSpecId("ffffffff")).rejects.toThrow(
      /No spec found with ID prefix "ffffffff"/,
    );
  });

  it("hints to use the full UUID when there are more than one page of specs", async () => {
    vi.mocked(client.get).mockResolvedValue({
      data: new Array(100).fill(0).map((_, i) => ({
        id: `aaaaaaaa-${String(i).padStart(4, "0")}-4000-8000-000000000000`,
        name: `S${i}`,
      })),
      meta: { page: 1, per_page: 100, total: 250 },
    });

    await expect(resolveSpecId("ffffffff")).rejects.toThrow(/250 total/);
    await expect(resolveSpecId("ffffffff")).rejects.toThrow(/full UUID/);
  });
});

describe("resolveRunId", () => {
  it("returns full UUIDs unchanged without an API call", async () => {
    const result = await resolveRunId(RUN_A);
    expect(result).toBe(RUN_A);
    expect(client.get).not.toHaveBeenCalled();
  });

  it("resolves a unique prefix from /runs", async () => {
    vi.mocked(client.get).mockResolvedValue({
      data: [{ id: RUN_A, run_number: 7 }],
      meta: { page: 1, per_page: 100, total: 1 },
    });

    const result = await resolveRunId(RUN_A.slice(0, 8));
    expect(result).toBe(RUN_A);
    expect(client.get).toHaveBeenCalledWith(
      "/runs",
      { page: 1, per_page: 100 },
      undefined,
    );
  });

  it("includes run_number in ambiguity hint", async () => {
    vi.mocked(client.get).mockResolvedValue({
      data: [
        { id: "33333333-3333-4333-8333-aaaaaaaaaaaa", run_number: 1 },
        { id: "33333333-3333-4333-8333-bbbbbbbbbbbb", run_number: 2 },
      ],
      meta: { page: 1, per_page: 100, total: 2 },
    });

    await expect(resolveRunId("33333333")).rejects.toThrow(/run #1/);
    await expect(resolveRunId("33333333")).rejects.toThrow(/run #2/);
  });
});
