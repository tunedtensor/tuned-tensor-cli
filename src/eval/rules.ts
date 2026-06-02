import type {
  AssertionResult,
  LocalSpec,
  ValidationCheck,
  ValidationResult,
} from "./types.js";

export function validateSpec(spec: LocalSpec): ValidationResult {
  const checks: ValidationCheck[] = [];

  checks.push({
    name: "Has name",
    passed: Boolean(spec.name?.trim()),
    message: spec.name?.trim() ? undefined : "Spec is missing a name",
  });

  checks.push({
    name: "Has system prompt",
    passed: Boolean(spec.system_prompt?.trim()),
    message: spec.system_prompt?.trim()
      ? undefined
      : "Spec is missing a system_prompt",
  });

  checks.push({
    name: "Has base model",
    passed: Boolean(spec.base_model?.trim()),
    message: spec.base_model?.trim()
      ? undefined
      : "Spec is missing a base_model",
  });

  const exCount = spec.examples?.length ?? 0;
  checks.push({
    name: "Has examples",
    passed: exCount > 0,
    message: exCount > 0 ? `${exCount} example(s)` : "Add at least one example",
  });

  checks.push({
    name: "Has guidelines",
    passed: (spec.guidelines?.length ?? 0) > 0,
    message:
      (spec.guidelines?.length ?? 0) > 0
        ? undefined
        : "Consider adding guidelines for better eval coverage",
  });

  if (spec.constraints?.length && spec.examples?.length) {
    const violations = checkExamplesAgainstConstraints(spec);
    checks.push({
      name: "Examples satisfy constraints",
      passed: violations.length === 0,
      message:
        violations.length === 0
          ? undefined
          : violations.join("; "),
    });
  }

  const evalCaseErrors = validateEvalCases(spec);
  checks.push({
    name: "Eval cases valid",
    passed: evalCaseErrors.length === 0,
    message:
      evalCaseErrors.length === 0
        ? spec.eval_cases?.length
          ? `${spec.eval_cases.length} executable eval case(s)`
          : undefined
        : evalCaseErrors.join("; "),
  });

  return {
    valid: checks.every((c) => c.passed),
    checks,
  };
}

export function validateEvalCases(spec: Pick<LocalSpec, "eval_cases">): string[] {
  const errors: string[] = [];
  const cases = spec.eval_cases ?? [];
  for (const [caseIndex, evalCase] of cases.entries()) {
    const label = `eval_cases[${caseIndex}]`;
    if (!evalCase || typeof evalCase !== "object") {
      errors.push(`${label} must be an object`);
      continue;
    }
    if (typeof evalCase.input !== "string" || evalCase.input.trim().length === 0) {
      errors.push(`${label}.input must be a non-empty string`);
    }
    if (evalCase.runtime !== "python") {
      errors.push(`${label}.runtime must be "python"`);
    }
    if (!Array.isArray(evalCase.tests) || evalCase.tests.length === 0) {
      errors.push(`${label}.tests must contain at least one test`);
      continue;
    }

    for (const [testIndex, test] of evalCase.tests.entries()) {
      const testLabel = `${label}.tests[${testIndex}]`;
      if (!test || typeof test !== "object") {
        errors.push(`${testLabel} must be an object`);
        continue;
      }
      if (test.name !== undefined && (typeof test.name !== "string" || test.name.trim().length === 0)) {
        errors.push(`${testLabel}.name must be a non-empty string when provided`);
      }
      if (test.args !== undefined && (!Array.isArray(test.args) || !test.args.every((arg) => typeof arg === "string"))) {
        errors.push(`${testLabel}.args must be an array of strings`);
      }
      if (test.stdin !== undefined && typeof test.stdin !== "string") {
        errors.push(`${testLabel}.stdin must be a string`);
      }
      if (
        test.expected_exit_code !== undefined
        && (
          typeof test.expected_exit_code !== "number"
          || !Number.isInteger(test.expected_exit_code)
          || test.expected_exit_code < 0
          || test.expected_exit_code > 255
        )
      ) {
        errors.push(`${testLabel}.expected_exit_code must be an integer between 0 and 255`);
      }
      if (test.expected_stdout !== undefined && typeof test.expected_stdout !== "string") {
        errors.push(`${testLabel}.expected_stdout must be a string`);
      }
      validateFiles(test.files, `${testLabel}.files`, errors);
      validateFiles(test.expected_files, `${testLabel}.expected_files`, errors);
    }
  }
  return errors;
}

function validateFiles(
  files: unknown,
  label: string,
  errors: string[],
): void {
  if (files === undefined) return;
  if (!Array.isArray(files)) {
    errors.push(`${label} must be an array`);
    return;
  }
  for (const [index, file] of files.entries()) {
    const fileLabel = `${label}[${index}]`;
    if (!file || typeof file !== "object") {
      errors.push(`${fileLabel} must be an object`);
      continue;
    }
    const row = file as { path?: unknown; content?: unknown };
    if (typeof row.path !== "string" || !isSafeRelativePath(row.path)) {
      errors.push(`${fileLabel}.path must be a safe relative path`);
    }
    if (row.content !== undefined && typeof row.content !== "string") {
      errors.push(`${fileLabel}.content must be a string`);
    }
  }
}

function isSafeRelativePath(value: string): boolean {
  if (value.length === 0 || value.length > 240) return false;
  if (value.includes("\0") || value.startsWith("/")) return false;
  const normalized = value.replace(/\\/g, "/");
  return !normalized.split("/").some((part) => part === "" || part === "." || part === "..");
}

function checkExamplesAgainstConstraints(spec: LocalSpec): string[] {
  const violations: string[] = [];
  for (const [i, ex] of spec.examples.entries()) {
    for (const constraint of spec.constraints) {
      const result = evaluateConstraint(ex.output, constraint);
      if (!result.passed) {
        violations.push(`Example ${i + 1} violates "${constraint}": ${result.message}`);
      }
    }
  }
  return violations;
}

function evaluateConstraint(
  text: string,
  constraint: string,
): { passed: boolean; message?: string } {
  const lower = constraint.toLowerCase();

  const neverMatch = lower.match(/^never (?:share|mention|include|reveal|use|say|output|return|give|provide|show|disclose|expose)\s+(.+)/);
  if (neverMatch) {
    const forbidden = neverMatch[1].replace(/['"]/g, "").trim();
    if (text.toLowerCase().includes(forbidden.toLowerCase())) {
      return { passed: false, message: `Output contains "${forbidden}"` };
    }
    return { passed: true };
  }

  const alwaysMatch = lower.match(/^always (?:include|use|start with|end with|contain|respond in|reply in)\s+(.+)/);
  if (alwaysMatch) {
    const required = alwaysMatch[1].replace(/['"]/g, "").trim();
    if (!text.toLowerCase().includes(required.toLowerCase())) {
      return { passed: false, message: `Output missing "${required}"` };
    }
    return { passed: true };
  }

  return {
    passed: true,
    message: `Constraint not enforceable by rules: "${constraint}" (will be checked by LLM judge if a provider is configured)`,
  };
}

export function runAssertions(
  text: string,
  assertions: string[],
): AssertionResult[] {
  return assertions.map((assertion) => runSingleAssertion(text, assertion));
}

function runSingleAssertion(text: string, assertion: string): AssertionResult {
  const [type, ...rest] = assertion.split(":");
  const value = rest.join(":");

  switch (type) {
    case "contains":
      return {
        assertion,
        passed: text.includes(value),
        message: text.includes(value)
          ? undefined
          : `Expected output to contain "${value}"`,
      };

    case "not-contains":
      return {
        assertion,
        passed: !text.includes(value),
        message: !text.includes(value)
          ? undefined
          : `Expected output to not contain "${value}"`,
      };

    case "matches": {
      const re = new RegExp(value);
      const match = re.test(text);
      return {
        assertion,
        passed: match,
        message: match ? undefined : `Output did not match pattern /${value}/`,
      };
    }

    case "max-length": {
      const max = Number(value);
      return {
        assertion,
        passed: text.length <= max,
        message:
          text.length <= max
            ? undefined
            : `Output length ${text.length} exceeds max ${max}`,
      };
    }

    case "min-length": {
      const min = Number(value);
      return {
        assertion,
        passed: text.length >= min,
        message:
          text.length >= min
            ? undefined
            : `Output length ${text.length} below min ${min}`,
      };
    }

    case "is-json": {
      let valid = false;
      try {
        JSON.parse(text);
        valid = true;
      } catch {}
      return {
        assertion,
        passed: valid,
        message: valid ? undefined : "Output is not valid JSON",
      };
    }

    default:
      return {
        assertion,
        passed: true,
        message: `Unknown assertion type: ${type} (skipped)`,
      };
  }
}

export function checkConstraints(
  text: string,
  constraints: string[],
): AssertionResult[] {
  return constraints.map((constraint) => {
    const result = evaluateConstraint(text, constraint);
    return {
      assertion: `constraint: ${constraint}`,
      passed: result.passed,
      message: result.message,
    };
  });
}
