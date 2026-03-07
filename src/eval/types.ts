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
  score: number | null;
  reasoning: string | null;
  latency_ms: number | null;
  assertions: AssertionResult[];
}

export interface EvalSummary {
  total: number;
  passed: number;
  failed: number;
  mean_score: number | null;
  pass_rate: number;
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

export interface ProviderConfig {
  provider: "ollama" | "openai" | "custom";
  model: string;
  baseUrl?: string;
  apiKey?: string;
}
