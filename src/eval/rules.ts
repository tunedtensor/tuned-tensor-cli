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

  return {
    valid: checks.every((c) => c.passed),
    checks,
  };
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
