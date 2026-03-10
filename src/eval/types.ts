export interface LocalSpec {
  id?: string;
  name: string;
  description?: string;
  base_model: string;
  system_prompt: string;
  guidelines: string[];
  constraints: string[];
  examples: Example[];
  eval_cases?: EvalCase[];
}

export interface Example {
  input: string;
  output: string;
}

export interface EvalCase {
  input: string;
  expected?: string | null;
  assert?: string[];
}

export interface AssertionResult {
  assertion: string;
  passed: boolean;
  message?: string;
}

export interface EvalResult {
  input: string;
  expected: string | null;
  actual: string | null;
  passed: boolean;
  latency_ms: number | null;
  assertions: AssertionResult[];
}

export interface EvalSummary {
  total: number;
  passed: number;
  failed: number;
  pass_rate: number;
  model: string;
  results: EvalResult[];
  spec_validation: ValidationResult;
}

export interface ValidationResult {
  valid: boolean;
  checks: ValidationCheck[];
}

export interface ValidationCheck {
  name: string;
  passed: boolean;
  message?: string;
}

export interface PlaygroundResponse {
  content: string;
  latency_ms: number;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
  };
}
