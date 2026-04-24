import { describe, it, expect, vi, beforeEach } from "vitest";
import { Command } from "commander";
import { registerBalanceCommands } from "../../commands/balance.js";
import * as client from "../../client.js";
import { setJsonMode } from "../../output.js";

vi.mock("../../client.js", async (importOriginal) => {
  const actual = await importOriginal<typeof client>();
  return {
    ...actual,
    get: vi.fn(),
  };
});

const FAKE_KEY = "tt_" + "a".repeat(48);

function buildProgram() {
  const program = new Command();
  program
    .option("-k, --api-key <key>", "API key")
    .option("-u, --base-url <url>", "Base URL")
    .option("--json", "JSON mode");
  registerBalanceCommands(program);
  program.exitOverride();
  return program;
}

const mockBalance = {
  balance_cents: 480,
  lifetime_topup_cents: 2500,
  signup_bonus_cents: 500,
  signup_bonus_granted: true,
};

const mockTransactions = [
  {
    id: "11111111-1111-1111-1111-111111111111",
    kind: "topup",
    amount_cents: 2500,
    balance_after_cents: 2980,
    reference_id: "pi_test_1",
    created_at: "2026-04-20T00:00:00Z",
  },
  {
    id: "22222222-2222-2222-2222-222222222222",
    kind: "debit_run",
    amount_cents: -2500,
    balance_after_cents: 480,
    reference_id: "00000000-0000-0000-0000-0000000000aa",
    created_at: "2026-04-21T00:00:00Z",
  },
];

beforeEach(() => {
  setJsonMode(false);
  process.env.TUNED_TENSOR_API_KEY = FAKE_KEY;
  vi.mocked(client.get).mockReset();
  vi.mocked(client.get).mockImplementation(((path: string) => {
    if (path === "/billing/balance") {
      return Promise.resolve({ data: mockBalance });
    }
    if (path === "/billing/transactions") {
      return Promise.resolve({ data: mockTransactions });
    }
    return Promise.reject(new Error(`unexpected path: ${path}`));
  }) as typeof client.get);
});

describe("balance command", () => {
  it("fetches and displays balance + transactions", async () => {
    const spy = vi.spyOn(console, "log").mockImplementation(() => {});
    const program = buildProgram();
    await program.parseAsync(["node", "tt", "balance"]);
    expect(client.get).toHaveBeenCalledWith(
      "/billing/balance",
      undefined,
      expect.anything(),
    );
    expect(client.get).toHaveBeenCalledWith(
      "/billing/transactions",
      { per_page: 10 },
      expect.anything(),
    );
    const out = spy.mock.calls.map((c) => String(c[0])).join("\n");
    expect(out).toContain("$4.80");
    expect(out).toContain("Top-up");
    expect(out).toContain("Run");
  });

  it("outputs JSON in json mode", async () => {
    setJsonMode(true);
    const spy = vi.spyOn(console, "log").mockImplementation(() => {});
    const program = buildProgram();
    await program.parseAsync(["node", "tt", "balance"]);
    const parsed = JSON.parse(spy.mock.calls[0][0] as string);
    expect(parsed).toHaveProperty("balance");
    expect(parsed).toHaveProperty("transactions");
    expect(parsed.balance.balance_cents).toBe(480);
    expect(parsed.transactions).toHaveLength(2);
  });

  it("warns on low balance", async () => {
    vi.mocked(client.get).mockImplementation(((path: string) => {
      if (path === "/billing/balance") {
        return Promise.resolve({
          data: { ...mockBalance, balance_cents: 50 },
        });
      }
      return Promise.resolve({ data: [] });
    }) as typeof client.get);
    const spy = vi.spyOn(console, "log").mockImplementation(() => {});
    const program = buildProgram();
    await program.parseAsync(["node", "tt", "balance"]);
    const out = spy.mock.calls.map((c) => String(c[0])).join("\n");
    expect(out).toMatch(/low/i);
    expect(out).toMatch(/topup/i);
  });

  it("respects --limit option", async () => {
    const spy = vi.spyOn(console, "log").mockImplementation(() => {});
    const program = buildProgram();
    await program.parseAsync(["node", "tt", "balance", "--limit", "25"]);
    expect(client.get).toHaveBeenCalledWith(
      "/billing/transactions",
      { per_page: 25 },
      expect.anything(),
    );
    expect(spy).toHaveBeenCalled();
  });
});
