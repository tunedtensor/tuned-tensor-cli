import { describe, expect, it, vi } from "vitest";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import { trimAgentContext } from "../agent-context.js";

const user = (text: string) => ({ role: "user", content: text, timestamp: 1 });
const answer = (content: unknown[]) => ({ role: "assistant", content });
const toolResult = (text: string) => ({ role: "toolResult", toolCallId: "call", toolName: "get_local_spec", content: [{ type: "text", text }] });
const options = { contextWindow: 6000, maxTokens: 1000, tools: [] };
const messages = (items: unknown[]) => items as AgentMessage[];

describe("agent context budget", () => {
  it("leaves short conversations intact", () => {
    const input = messages([user("Create a spec"), answer([{ type: "text", text: "Prepared" }])]);
    expect(trimAgentContext(input, options)).toBe(input);
  });

  it("drops old tool exchanges together, retains instructions and current calls, and leaves saved data intact", () => {
    const input = messages([
      { role: "system", content: "System instructions", toolsAdded: [{ name: "get_local_spec" }] },
      user("Use Python and never start training"),
      answer([{ type: "thinking", thinking: "x".repeat(9000) }, { type: "toolCall", id: "call", name: "get_local_spec", arguments: {} }]),
      toolResult("x".repeat(9000)),
      answer([{ type: "text", text: "Created python-agent/tunedtensor.json" }]),
      user("Download its model"),
      answer([{ type: "toolCall", id: "current", name: "get_local_spec", arguments: { spec_path: "python-agent/tunedtensor.json" } }]),
      { ...toolResult("Current spec"), toolCallId: "current" },
    ]);
    const saved = JSON.stringify(input);
    const onTrim = vi.fn();
    const output = trimAgentContext(input, { ...options, onTrim });
    expect(output).toEqual([input[0], input[1], input[4], input[5], input[6], input[7]]);
    expect(JSON.stringify(input)).toBe(saved);
    expect(onTrim).toHaveBeenCalledOnce();
  });

  it("uses provider input usage to catch overhead missed by character estimates", () => {
    const input = messages([
      user("Create a spec"),
      answer([{ type: "toolCall", id: "call", name: "get_local_spec", arguments: {} }]),
      toolResult("x".repeat(3000)),
      { ...answer([{ type: "text", text: "Spec ready" }]), usage: { input: 2200, cacheRead: 0, cacheWrite: 0 } },
      user("Find a dataset"),
      answer([{ type: "toolCall", id: "search", name: "search_hugging_face", arguments: {} }]),
      { ...toolResult("y".repeat(2500)), toolCallId: "search" },
    ]);
    const onTrim = vi.fn();
    const output = trimAgentContext(input, { ...options, onTrim });
    expect(onTrim).toHaveBeenCalledOnce();
    expect(output).toEqual([input[0], input[3], input[4], input[5], input[6]]);
  });

  it("does not silently discard old user instructions when text alone exceeds the budget", () => {
    const input = messages([user("x".repeat(9000)), answer([{ type: "text", text: "Old answer" }]), user("Current request")]);
    expect(() => trimAgentContext(input, options)).toThrow(/context budget/);
  });

  it("reports an oversized current request instead of cutting its tool results", () => {
    const input = messages([user("Current request"), toolResult("x".repeat(10000))]);
    expect(() => trimAgentContext(input, options)).toThrow(/context budget/);
  });
});
