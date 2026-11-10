import { z } from "zod";
import type { EvalReport, RunReport } from "./contracts.js";

/** Acceptance criteria are agreed in the spec before spending GPU time. */
export const taskQualityPolicySchema = z.object({
  minScore: z.number().min(0).max(1),
  minPassRate: z.number().min(0).max(1),
  minExamples: z.number().int().min(1),
  minScoreDelta: z.number().min(0).max(1).default(0),
}).strict();
export type TaskQualityPolicy = z.infer<typeof taskQualityPolicySchema>;
export const taskQualityGateSchema = z.object({
  policy: taskQualityPolicySchema,
  passed: z.boolean(),
  failures: z.array(z.string()),
}).strict();

export function evaluateTaskQuality(
  baseline: EvalReport,
  candidate: EvalReport,
  policy?: TaskQualityPolicy,
): { passed: boolean; failures: string[] } {
  const failures: string[] = [];
  const tolerance = 1e-12;
  const minimum = policy?.minExamples ?? 1;
  if (candidate.total < minimum || candidate.eval_examples_used < minimum) {
    failures.push(`Task evaluation needs at least ${minimum} held-out examples; evaluated ${candidate.eval_examples_used}.`);
  }
  if (baseline.results.length !== candidate.results.length
    || baseline.results.length !== baseline.total || candidate.results.length !== candidate.total
    || baseline.results.some((row, i) => row.prompt !== candidate.results[i]?.prompt || row.expected !== candidate.results[i]?.expected)) {
    failures.push("Baseline and candidate must evaluate the same held-out prompts and references.");
  }
  if (policy && (!["spec_holdout", "prebuilt_test", "prebuilt_validation"].includes(candidate.eval_split ?? "")
    || baseline.inference_provider !== "transformers" || candidate.inference_provider !== "transformers")) {
    failures.push("Task acceptance requires recorded held-out inference evidence.");
  }
  if (candidate.eval_split === "spec_examples") failures.push("Task quality requires held-out evaluation, not the training examples.");
  if (baseline.inference_provider === "none" || candidate.inference_provider === "none") {
    failures.push("A dry-run has no measured model quality.");
  }
  if (candidate.avg_score + tolerance < (policy?.minScore ?? 0)) {
    failures.push(`Task score ${candidate.avg_score.toFixed(4)} is below ${policy!.minScore.toFixed(4)}.`);
  }
  if (candidate.pass_rate + tolerance < (policy?.minPassRate ?? 0)) {
    failures.push(`Task pass rate ${candidate.pass_rate.toFixed(4)} is below ${policy!.minPassRate.toFixed(4)}.`);
  }
  if (candidate.avg_score - baseline.avg_score + tolerance < (policy?.minScoreDelta ?? 0)) {
    failures.push(`Task score change ${(candidate.avg_score - baseline.avg_score).toFixed(4)} is below the required ${(policy?.minScoreDelta ?? 0).toFixed(4)}.`);
  }
  if (candidate.pass_rate + tolerance < baseline.pass_rate) {
    failures.push("Task pass rate regressed from the baseline.");
  }
  return { passed: failures.length === 0, failures };
}

/** Recompute, rather than trusting a persisted passed flag. */
export function taskQualityForReport(report: RunReport) {
  return evaluateTaskQuality(report.baseline, report.candidate, report.task_quality?.policy);
}
