import { chatCompletion } from "./providers.js";
import type { ProviderConfig } from "./types.js";

interface JudgeResult {
  score: number;
  passed: boolean;
  reasoning: string;
}

export async function judgeResponse(opts: {
  input: string;
  output: string;
  expected: string | null;
  systemPrompt: string;
  guidelines: string[];
  constraints: string[];
  config: ProviderConfig;
}): Promise<JudgeResult> {
  const criteriaLines: string[] = [];

  if (opts.guidelines.length) {
    criteriaLines.push("Guidelines the response should follow:");
    for (const g of opts.guidelines) criteriaLines.push(`  - ${g}`);
  }
  if (opts.constraints.length) {
    criteriaLines.push("Constraints the response must satisfy:");
    for (const c of opts.constraints) criteriaLines.push(`  - ${c}`);
  }
  if (opts.expected) {
    criteriaLines.push(`Expected response (reference): ${opts.expected}`);
  }

  const judgePrompt = `You are an eval judge. Score an AI assistant's response on a scale of 0.0 to 1.0.

The assistant was given this system prompt:
"${opts.systemPrompt}"

${criteriaLines.join("\n")}

User input: "${opts.input}"
Assistant response: "${opts.output}"

Respond with ONLY a JSON object (no markdown, no explanation outside the JSON):
{"score": <0.0-1.0>, "passed": <true/false>, "reasoning": "<brief explanation>"}

A score >= 0.7 is a pass. Be strict but fair.`;

  const response = await chatCompletion(
    [{ role: "user", content: judgePrompt }],
    opts.config,
  );

  return parseJudgeResponse(response.content);
}

function parseJudgeResponse(content: string): JudgeResult {
  const jsonMatch = content.match(/\{[\s\S]*\}/);
  if (!jsonMatch) {
    return { score: 0, passed: false, reasoning: "Judge returned invalid format" };
  }

  try {
    const parsed = JSON.parse(jsonMatch[0]);
    const score = Math.max(0, Math.min(1, Number(parsed.score) || 0));
    return {
      score,
      passed: parsed.passed ?? score >= 0.7,
      reasoning: String(parsed.reasoning || "No reasoning provided"),
    };
  } catch {
    return { score: 0, passed: false, reasoning: "Failed to parse judge response" };
  }
}
