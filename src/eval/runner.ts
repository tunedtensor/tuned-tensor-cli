import type {
  EvalCase,
  EvalResult,
  EvalSummary,
  LocalSpec,
  PlaygroundResponse,
} from "./types.js";
import type { ClientOpts } from "../client.js";
import { post } from "../client.js";
import { validateSpec } from "./rules.js";
import { runAssertions, checkConstraints } from "./rules.js";

function buildEvalCases(spec: LocalSpec): EvalCase[] {
  if (spec.eval_cases?.length) return spec.eval_cases;

  return spec.examples.map((ex) => ({
    input: ex.input,
    expected: ex.output,
  }));
}

export async function runEvals(
  spec: LocalSpec,
  opts?: {
    model?: string;
    clientOpts?: ClientOpts;
    onProgress?: (completed: number, total: number) => void;
  },
): Promise<EvalSummary> {
  const specValidation = validateSpec(spec);
  const cases = buildEvalCases(spec);
  const results: EvalResult[] = [];
  const model = opts?.model ?? null;

  for (const [i, evalCase] of cases.entries()) {
    const result = model
      ? await runModelEval(spec, evalCase, model, opts?.clientOpts)
      : runOfflineEval(spec, evalCase);
    results.push(result);
    opts?.onProgress?.(i + 1, cases.length);
  }

  return {
    total: results.length,
    passed: results.filter((r) => r.passed).length,
    failed: results.filter((r) => !r.passed).length,
    pass_rate: results.length ? results.filter((r) => r.passed).length / results.length : 0,
    model,
    results,
    spec_validation: specValidation,
  };
}

function runOfflineEval(
  spec: LocalSpec,
  evalCase: EvalCase,
): EvalResult {
  const text = evalCase.expected ?? "";
  const assertions = [
    ...runAssertions(text, evalCase.assert ?? []),
    ...checkConstraints(text, spec.constraints),
  ];

  return {
    input: evalCase.input,
    expected: evalCase.expected ?? null,
    actual: null,
    passed: assertions.every((a) => a.passed),
    latency_ms: null,
    assertions,
  };
}

async function runModelEval(
  spec: LocalSpec,
  evalCase: EvalCase,
  model: string,
  clientOpts?: ClientOpts,
): Promise<EvalResult> {
  try {
    const { data } = await post<PlaygroundResponse>(
      "/playground/completions",
      {
        model,
        messages: [
          { role: "system", content: spec.system_prompt },
          { role: "user", content: evalCase.input },
        ],
        temperature: 0,
        max_tokens: 2048,
      },
      clientOpts,
    );

    const assertions = [
      ...runAssertions(data.content, evalCase.assert ?? []),
      ...checkConstraints(data.content, spec.constraints),
    ];

    return {
      input: evalCase.input,
      expected: evalCase.expected ?? null,
      actual: data.content,
      passed: assertions.every((a) => a.passed),
      latency_ms: data.latency_ms,
      assertions,
    };
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    return {
      input: evalCase.input,
      expected: evalCase.expected ?? null,
      actual: null,
      passed: false,
      latency_ms: null,
      assertions: [{ assertion: "model-call", passed: false, message: msg }],
    };
  }
}
