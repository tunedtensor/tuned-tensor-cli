import type { AgentMessage, AgentTool } from "@earendil-works/pi-agent-core";

/** Trim only the model's view; the persisted transcript and approvals stay intact. */
export function trimAgentContext(
  messages: AgentMessage[],
  options: { contextWindow: number; maxTokens: number; tools: AgentTool[]; onTrim?: () => void },
): AgentMessage[] {
  // Code and JSON tokenize more densely than prose. Reserve room for the answer,
  // protocol overhead, and tool declarations even when Pi stores them separately.
  const toolChars = JSON.stringify(options.tools.map(({ name, description, parameters }) => ({ name, description, parameters }))).length;
  const budget = Math.max(0, options.contextWindow - options.maxTokens - 2048);
  const estimate = (items: AgentMessage[]) => (toolChars + items.reduce((total, item) =>
    total + JSON.stringify("content" in item ? item.content : item).length, 0)) / 3;
  // Providers add protocol/template tokens that the character estimate misses.
  // Calibrate against the most recent actual input usage, never downward: that
  // response may itself have used a trimmed view of this full transcript.
  let overhead = 0;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]!;
    if (message.role !== "assistant" || !message.usage) continue;
    const input = message.usage.input + (message.usage.cacheRead ?? 0) + (message.usage.cacheWrite ?? 0);
    if (input > 0) {
      overhead = Math.max(0, input - estimate(messages.slice(0, index)));
      break;
    }
  }
  const size = (items: AgentMessage[]) => estimate(items) + overhead;
  if (size(messages) <= budget) return messages;

  let latestUser = messages.length - 1;
  while (latestUser >= 0 && messages[latestUser]!.role !== "user") latestUser -= 1;
  if (latestUser < 0) throw new Error("The conversation is too large. Use /new to start a fresh conversation.");
  // Keep user instructions and the assistant's explanations. Drop completed tool
  // exchanges together, never leave an orphan tool result or incomplete call.
  const earlier = messages.slice(0, latestUser).flatMap((message): AgentMessage[] => {
    if (message.role === "toolResult") return [];
    if (message.role !== "assistant") return [message];
    const content = message.content.filter((part) => part.type === "text");
    return content.length ? [{ ...message, content }] : [];
  });
  const current = messages.slice(latestUser);
  const trimmed = [...earlier, ...current];
  if (size(trimmed) > budget) {
    throw new Error("The conversation still exceeds the model's context budget after omitting older tool output. Use a smaller request or /new.");
  }
  options.onTrim?.();
  return trimmed;
}
