import { describe, it, expect, vi, beforeEach } from "vitest";
import { Command } from "commander";
import { registerTopupCommands } from "../../commands/topup.js";
import * as client from "../../client.js";
import { setJsonMode } from "../../output.js";

vi.mock("../../client.js", async (importOriginal) => {
  const actual = await importOriginal<typeof client>();
  return {
    ...actual,
    post: vi.fn(),
  };
});

vi.mock("open", () => ({
  default: vi.fn().mockResolvedValue(undefined),
}));

const FAKE_KEY = "tt_" + "a".repeat(48);

function buildProgram() {
  const program = new Command();
  program
    .option("-k, --api-key <key>", "API key")
    .option("-u, --base-url <url>", "Base URL")
    .option("--json", "JSON mode");
  registerTopupCommands(program);
  program.exitOverride();
  return program;
}

const mockResponse = {
  checkout_url: "https://checkout.stripe.com/c/pay/test_123",
  session_id: "cs_test_123",
};

beforeEach(() => {
  setJsonMode(false);
  process.env.TUNED_TENSOR_API_KEY = FAKE_KEY;
  vi.mocked(client.post).mockReset();
  vi.mocked(client.post).mockResolvedValue({ data: mockResponse });
});

describe("topup command", () => {
  it("creates a Stripe Checkout session for $25", async () => {
    const spy = vi.spyOn(console, "log").mockImplementation(() => {});
    const program = buildProgram();
    await program.parseAsync([
      "node",
      "tt",
      "topup",
      "--amount",
      "25",
      "--no-open",
    ]);
    expect(client.post).toHaveBeenCalledWith(
      "/billing/topup",
      { amount_cents: 2500 },
      expect.anything(),
    );
    const out = spy.mock.calls.map((c) => String(c[0])).join("\n");
    expect(out).toContain("checkout.stripe.com");
  });

  it("strips dollar sign from amount", async () => {
    vi.spyOn(console, "log").mockImplementation(() => {});
    const program = buildProgram();
    await program.parseAsync([
      "node",
      "tt",
      "topup",
      "--amount",
      "$50",
      "--no-open",
    ]);
    expect(client.post).toHaveBeenCalledWith(
      "/billing/topup",
      { amount_cents: 5000 },
      expect.anything(),
    );
  });

  it("outputs JSON in json mode", async () => {
    setJsonMode(true);
    const spy = vi.spyOn(console, "log").mockImplementation(() => {});
    const program = buildProgram();
    await program.parseAsync([
      "node",
      "tt",
      "topup",
      "--amount",
      "10",
      "--no-open",
    ]);
    const parsed = JSON.parse(spy.mock.calls[0][0] as string);
    expect(parsed).toEqual(mockResponse);
  });

  it("rejects amount below minimum", async () => {
    const program = buildProgram();
    await expect(
      program.parseAsync([
        "node",
        "tt",
        "topup",
        "--amount",
        "1",
        "--no-open",
      ]),
    ).rejects.toThrow(/minimum/i);
    expect(client.post).not.toHaveBeenCalled();
  });

  it("rejects amount above maximum", async () => {
    const program = buildProgram();
    await expect(
      program.parseAsync([
        "node",
        "tt",
        "topup",
        "--amount",
        "999999",
        "--no-open",
      ]),
    ).rejects.toThrow(/maximum/i);
  });

  it("rejects invalid amount", async () => {
    const program = buildProgram();
    await expect(
      program.parseAsync([
        "node",
        "tt",
        "topup",
        "--amount",
        "abc",
        "--no-open",
      ]),
    ).rejects.toThrow(/invalid amount/i);
  });

  it("requires --amount in json mode", async () => {
    setJsonMode(true);
    const program = buildProgram();
    await expect(
      program.parseAsync(["node", "tt", "topup", "--no-open"]),
    ).rejects.toThrow(/amount is required/i);
  });
});
