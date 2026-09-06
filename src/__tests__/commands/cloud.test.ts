import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createProgram } from "../../cli.js";
import * as client from "../../client.js";
import { setJsonMode } from "../../output.js";

vi.mock("../../client.js", async (original) => ({
  ...await original<typeof client>(), get: vi.fn(), post: vi.fn(),
}));

const runId = "33333333-3333-4333-8333-333333333333";
const token = `tt_${"a".repeat(48)}`;

beforeEach(() => { vi.clearAllMocks(); });
afterEach(() => { setJsonMode(false); vi.restoreAllMocks(); });

describe("explicit cloud operations", () => {
  it("routes cloud runs to the API with inherited token/URL and preserves root local routing", async () => {
    vi.mocked(client.get).mockResolvedValue({ data: [], meta: { page: 1, per_page: 20, total: 0 } });
    vi.spyOn(console, "log").mockImplementation(() => {});
    const local = vi.fn().mockResolvedValue({ exitCode: 0, signal: null });
    const program = createProgram("test", { runLocalCommand: local });
    program.exitOverride();
    await program.parseAsync(["node", "tt", "--api-key", token, "--base-url", "https://custom.example", "--json", "cloud", "runs", "list"]);
    expect(client.get).toHaveBeenCalledWith("/runs", { page: "1", per_page: "20" }, expect.objectContaining({ apiKey: token, baseUrl: "https://custom.example" }));
    expect(local).not.toHaveBeenCalled();
    vi.mocked(client.get).mockClear();
    await createProgram("test", { runLocalCommand: local }).parseAsync(["node", "tt", "runs", "list"]);
    expect(local.mock.calls[0]?.[0]).toEqual(["runs", "list"]);
    expect(client.get).not.toHaveBeenCalled();
  });

  it("reports managed usage in JSON independently of the BYO model selection", async () => {
    const usage = {
      period_start: "2026-09-06T00:00:00.000Z", resets_at: "2026-09-07T00:00:00.000Z",
      requests: 4, completed_requests: 2, failed_requests: 1, cancelled_requests: 1,
      running_requests: 0, usage_reported_requests: 2, cost_reported_requests: 2, prompt_tokens: 120, completion_tokens: 30,
      provider_cost_usd: 0.01, daily_request_limit: 100, remaining_requests: 96,
      billing: "included_allowance",
    };
    vi.mocked(client.get).mockResolvedValue({ data: usage });
    const log = vi.spyOn(console, "log").mockImplementation(() => {});
    await createProgram("test").parseAsync(["node", "tt", "--json", "--api-key", token, "usage"]);
    expect(client.get).toHaveBeenCalledWith("/agent/usage", undefined, expect.objectContaining({ apiKey: token }));
    expect(JSON.parse(String(log.mock.calls[0]?.[0]))).toEqual(usage);
  });

  it("archives a published report through the existing API", async () => {
    vi.mocked(client.post).mockResolvedValue({ data: { id: runId, archived: true } });
    vi.spyOn(console, "log").mockImplementation(() => {});
    await createProgram("test").parseAsync(["node", "tt", "--json", "--api-key", token, "cloud", "runs", "archive", runId]);
    expect(client.post).toHaveBeenCalledWith(`/runs/${runId}/archive`, undefined, expect.objectContaining({ apiKey: token }));
  });
});
