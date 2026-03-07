import type {
  EvalCase,
  EvalResult,
  EvalSummary,
  LocalSpec,
} from "./types.js";
import { validateSpec } from "./rules.js";
import { runAssertions, checkConstraints } from "./rules.js";

function buildEvalCases(spec: LocalSpec): EvalCase[] {
  if (spec.eval_cases?.length) return spec.eval_cases;

  return spec.examples.map((ex) => ({
    input: ex.input,
    expected: ex.output,
  }));
}

export function runEvals(
  spec: LocalSpec,
  onProgress?: (completed: number, total: number) => void,
): EvalSummary {
  const specValidation = validateSpec(spec);
  const cases = buildEvalCases(spec);
  const results: EvalResult[] = [];

  for (const [i, evalCase] of cases.entries()) {
    const result = runSingleEval(spec, evalCase);
    results.push(result);
    onProgress?.(i + 1, cases.length);
  }

  return {
    total: results.length,
    passed: results.filter((r) => r.passed).length,
    failed: results.filter((r) => !r.passed).length,
    pass_rate: results.length ? results.filter((r) => r.passed).length / results.length : 0,
    results,
    spec_validation: specValidation,
  };
}

function runSingleEval(
  spec: LocalSpec,
  evalCase: EvalCase,
): EvalResult {
  const text = evalCase.expected ?? "";
  const assertions = [
    ...runAssertions(text, evalCase.assert ?? []),
    ...checkConstraints(text, spec.constraints),
  ];

  const allPassed = assertions.every((a) => a.passed);

  return {
    input: evalCase.input,
    expected: evalCase.expected ?? null,
    passed: allPassed,
    assertions,
  };
}
