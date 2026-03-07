import type {
  EvalCase,
  EvalResult,
  EvalSummary,
  LocalSpec,
  ProviderConfig,
} from "./types.js";
import { validateSpec } from "./rules.js";
import { runAssertions, checkConstraints } from "./rules.js";
import { chatCompletion } from "./providers.js";
import { judgeResponse } from "./judge.js";

function buildEvalCases(spec: LocalSpec): EvalCase[] {
  if (spec.eval_cases?.length) return spec.eval_cases;

  return spec.examples.map((ex) => ({
    input: ex.input,
    expected: ex.output,
  }));
}

export async function runEvals(
  spec: LocalSpec,
  provider: ProviderConfig,
  onProgress?: (completed: number, total: number) => void,
): Promise<EvalSummary> {
  const specValidation = validateSpec(spec);
  const cases = buildEvalCases(spec);
  const results: EvalResult[] = [];

  for (const [i, evalCase] of cases.entries()) {
    const result = await runSingleEval(spec, evalCase, provider);
    results.push(result);
    onProgress?.(i + 1, cases.length);
  }

  const scores = results.map((r) => r.score).filter((s): s is number => s !== null);

  return {
    total: results.length,
    passed: results.filter((r) => r.passed).length,
    failed: results.filter((r) => !r.passed).length,
    mean_score: scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : null,
    pass_rate: results.length ? results.filter((r) => r.passed).length / results.length : 0,
    results,
    spec_validation: specValidation,
  };
}

async function runSingleEval(
  spec: LocalSpec,
  evalCase: EvalCase,
  provider: ProviderConfig,
): Promise<EvalResult> {
  try {
    const response = await chatCompletion(
      [
        { role: "system", content: spec.system_prompt },
        { role: "user", content: evalCase.input },
      ],
      provider,
    );

    const assertions = [
      ...runAssertions(response.content, evalCase.assert ?? []),
      ...checkConstraints(response.content, spec.constraints),
    ];

    let judgeResult = null;
    if (spec.guidelines.length || evalCase.expected) {
      judgeResult = await judgeResponse({
        input: evalCase.input,
        output: response.content,
        expected: evalCase.expected ?? null,
        systemPrompt: spec.system_prompt,
        guidelines: spec.guidelines,
        constraints: spec.constraints,
        config: provider,
      });
    }

    const rulesPassed = assertions.every((a) => a.passed);
    const judgePassed = judgeResult ? judgeResult.passed : true;

    return {
      input: evalCase.input,
      expected: evalCase.expected ?? null,
      actual: response.content,
      passed: rulesPassed && judgePassed,
      score: judgeResult?.score ?? (rulesPassed ? 1.0 : 0.0),
      reasoning: judgeResult?.reasoning ?? (rulesPassed ? "All checks passed" : "Rule check(s) failed"),
      latency_ms: response.latency_ms,
      assertions,
    };
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    return {
      input: evalCase.input,
      expected: evalCase.expected ?? null,
      actual: null,
      passed: false,
      score: 0,
      reasoning: `Model call failed: ${msg}`,
      latency_ms: null,
      assertions: [],
    };
  }
}
