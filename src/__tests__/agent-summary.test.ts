import { describe, expect, it, vi } from "vitest";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import { createConversationContext, type ConversationSummary } from "../agent-context.js";

const user = (content: string) => ({ role: "user" as const, content, timestamp: 1 });
const assistant = (text: string) => ({ role: "assistant", content: [{ type: "text", text }], timestamp: 2,
  usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0 }, stopReason: "stop" });
const history = () => [
  { role: "system", content: "Current host instructions" },
  user("Create a Python coding agent. Never start training. " + "Earlier discussion. ".repeat(2000)),
  assistant("Muse downloaded; revision abc; checksums verified."),
  user("Recent question"), assistant("Recent answer"), user("Do we have Muse already?"),
] as AgentMessage[];

function setup(overrides: { summary?: ConversationSummary; stopReason?: string; text?: string; onCall?: () => void } = {}) {
  const stream = vi.fn((_model, _context, _options) => {
    overrides.onCall?.();
    return { result: async () => ({ ...assistant(overrides.text ?? "Python coding agent. Never train. Muse verified in cache."),
      stopReason: overrides.stopReason ?? "stop", errorMessage: "Provider unavailable" }) };
  });
  const onSummary = vi.fn(async (_summary: ConversationSummary) => {});
  const onSummarizing = vi.fn();
  const options = {
    contextWindow: 32768, maxTokens: 4096,
    // Include tool declaration overhead in the automatic trigger.
    tools: [{ name: "inspect", description: "d".repeat(36000), parameters: {} }] as never[],
    model: { contextWindow: 32768, maxTokens: 4096 } as never,
    stream: stream as never, summary: overrides.summary, onSummary, onSummarizing,
  };
  return { transform: createConversationContext(options), options, stream, onSummary, onSummarizing };
}

describe("automatic Pi conversation summaries", () => {
  it("reports progress while waiting and cleans up the heartbeat after failure", async () => {
    vi.useFakeTimers();
    try {
      const s = setup();
      let fail!: (error: Error) => void;
      s.stream.mockImplementationOnce(() => ({ result: () => new Promise((_resolve, reject) => { fail = reject; }) }) as never);
      const pending = s.transform(history());
      const rejected = expect(pending).rejects.toThrow("Offline");
      await vi.advanceTimersByTimeAsync(15000);
      expect(s.onSummarizing).toHaveBeenLastCalledWith(true);
      fail(new Error("Offline"));
      await rejected;
      expect(vi.getTimerCount()).toBe(0);
      expect(s.onSummary).not.toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
  });

  it("summarizes complete old turns with Pi, keeps current tool pairs, and leaves the transcript untouched", async () => {
    const s = setup();
    const input = history();
    input.push({ ...assistant(""), content: [{ type: "toolCall", id: "current", name: "inspect", arguments: {} }] } as never,
      { role: "toolResult", toolCallId: "current", toolName: "inspect", isError: false, content: [{ type: "text", text: "Current result" }], timestamp: 3 });
    const original = structuredClone(input);
    const result = await s.transform(input);
    expect(s.onSummarizing).toHaveBeenCalledOnce();
    expect(s.onSummary).toHaveBeenCalledWith({ text: expect.stringContaining("Muse verified"), messageCount: 2 });
    expect(result[0]).toEqual(input[0]);
    expect(result[1]).toMatchObject({ role: "user", content: expect.stringContaining("Earlier conversation summary") });
    expect(result.slice(2)).toEqual(input.slice(3));
    expect(input).toEqual(original);
    const summaryContext = JSON.stringify(s.stream.mock.calls[0]![1]);
    expect(summaryContext).toContain("Create a Python coding agent");
    expect(summaryContext).not.toContain("Current result");
    expect(summaryContext).not.toContain("Recent question");
    expect(summaryContext).not.toContain("Current host instructions");
  });

  it("reuses a checkpoint after resume and rolls it forward when more history accumulates", async () => {
    const s = setup();
    const input = history();
    await s.transform(input);
    const summary = s.onSummary.mock.calls[0]![0];
    const resumed = setup({ summary });
    await resumed.transform(input);
    expect(resumed.stream).not.toHaveBeenCalled();
    input.push(assistant("Another lengthy discussion. ".repeat(1500)) as never, user("What next?"));
    await resumed.transform(input);
    expect(resumed.onSummary).toHaveBeenCalledWith({ text: expect.any(String), messageCount: 6 });
    expect(JSON.stringify(resumed.stream.mock.calls[0]![1])).toContain("previous-summary");
    expect(JSON.stringify(resumed.stream.mock.calls[0]![1])).not.toContain("Earlier discussion.");
  });

  it("batches long legacy histories into bounded summary requests", async () => {
    const s = setup();
    const input = history();
    input.splice(3, 0, user("Another old turn. ".repeat(2400)) as never, assistant("Done") as never);
    await s.transform(input);
    expect(s.stream).toHaveBeenCalledTimes(2);
    expect(s.onSummary).toHaveBeenCalledOnce();
    expect(JSON.stringify(s.stream.mock.calls[1]![1])).toContain("previous-summary");
  });

  it.each(["error", "length"])("does not checkpoint a %s response", async (stopReason) => {
    const s = setup({ stopReason });
    await expect(s.transform(history())).rejects.toThrow(/Summarization failed/);
    expect(s.onSummary).not.toHaveBeenCalled();
  });

  it("does not checkpoint an empty or cancelled summary", async () => {
    const empty = setup({ text: " " });
    await expect(empty.transform(history())).rejects.toThrow(/no text/);
    expect(empty.onSummary).not.toHaveBeenCalled();
    const controller = new AbortController();
    const cancelled = setup({ onCall: () => controller.abort() });
    await expect(cancelled.transform(history(), controller.signal)).rejects.toThrow(/abort/i);
    expect(cancelled.onSummary).not.toHaveBeenCalled();
    expect(cancelled.stream.mock.calls[0]![2]).toMatchObject({ signal: controller.signal });
  });

  it("does not discard an oversized current request or replace a checkpoint when saving fails", async () => {
    const s = setup();
    await expect(s.transform([user("x".repeat(100000))] as AgentMessage[])).rejects.toThrow(/context budget/);
    expect(s.onSummary).not.toHaveBeenCalled();
    s.onSummary.mockRejectedValueOnce(new Error("Disk full"));
    await expect(s.transform(history())).rejects.toThrow("Disk full");
    await s.transform(history());
    expect(s.stream).toHaveBeenCalledTimes(2);
  });
});
