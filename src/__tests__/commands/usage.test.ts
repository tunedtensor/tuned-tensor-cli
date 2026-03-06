import { describe, it, expect, vi, beforeEach } from "vitest";
import { Command } from "commander";
import { registerUsageCommands } from "../../commands/usage.js";
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
  registerUsageCommands(program);
  program.exitOverride();
  return program;
}

const mockUsageResponse = {
  usage: {
    tier: "free",
    tierName: "Free",
    runsUsed: 3,
    runsLimit: 10,
    periodStart: "2024-01-01T00:00:00Z",
    periodEnd: "2024-02-01T00:00:00Z",
    canStartRun: true,
  },
  tiers: [
    {
      slug: "free",
      name: "Free",
      runsPerMonth: 10,
      monthlyPriceUsd: 0,
      features: ["10 runs/month", "Community support"],
    },
    {
      slug: "pro",
      name: "Pro",
      runsPerMonth: 100,
      monthlyPriceUsd: 49,
      features: ["100 runs/month", "Priority support"],
    },
  ],
};

beforeEach(() => {
  setJsonMode(false);
  process.env.TUNED_TENSOR_API_KEY = FAKE_KEY;
});

describe("usage command", () => {
  it("fetches and displays usage summary", async () => {
    vi.mocked(client.get).mockResolvedValue({ data: mockUsageResponse });
    const spy = vi.spyOn(console, "log").mockImplementation(() => {});
    const program = buildProgram();
    await program.parseAsync(["node", "tt", "usage"]);
    expect(client.get).toHaveBeenCalledWith(
      "/usage",
      undefined,
      expect.anything(),
    );
    const allOutput = spy.mock.calls.map((c) => String(c[0])).join("\n");
    expect(allOutput).toContain("Free");
    expect(allOutput).toContain("3 / 10");
  });

  it("outputs JSON in json mode", async () => {
    setJsonMode(true);
    vi.mocked(client.get).mockResolvedValue({ data: mockUsageResponse });
    const spy = vi.spyOn(console, "log").mockImplementation(() => {});
    const program = buildProgram();
    await program.parseAsync(["node", "tt", "usage"]);
    const output = JSON.parse(spy.mock.calls[0][0]);
    expect(output).toHaveProperty("usage");
    expect(output).toHaveProperty("tiers");
  });

  it("displays tier information", async () => {
    vi.mocked(client.get).mockResolvedValue({ data: mockUsageResponse });
    const spy = vi.spyOn(console, "log").mockImplementation(() => {});
    const program = buildProgram();
    await program.parseAsync(["node", "tt", "usage"]);
    const allOutput = spy.mock.calls.map((c) => String(c[0])).join("\n");
    expect(allOutput).toContain("Pro");
    expect(allOutput).toContain("$49/mo");
  });

  it("shows current tier marker", async () => {
    vi.mocked(client.get).mockResolvedValue({ data: mockUsageResponse });
    const spy = vi.spyOn(console, "log").mockImplementation(() => {});
    const program = buildProgram();
    await program.parseAsync(["node", "tt", "usage"]);
    const allOutput = spy.mock.calls.map((c) => String(c[0])).join("\n");
    expect(allOutput).toContain("current");
  });
});
