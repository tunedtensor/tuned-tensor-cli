import { describe, expect, it } from "vitest";
import {
  canonicalPipeline,
  createExecutionPlan,
  validatePipeline,
  type Pipeline,
} from "../pipeline.js";

const valid: Pipeline = {
  version: 1,
  name: "hybrid",
  target: "local",
  steps: [
    { id: "baseline", uses: "evaluate", with: { model: "base" } },
    { id: "train", uses: "train", target: "cloud" },
    { id: "candidate", uses: "evaluate", with: { model: { from: "train.model" } } },
    { id: "compare", uses: "compare", with: {
      before: { from: "baseline.report" }, after: { from: "candidate.report" },
    } },
  ],
};

describe("composable pipeline contract", () => {
  it("accepts the ordered v1 recipe and resolves targets and transfers", () => {
    expect(validatePipeline(valid)).toEqual([]);
    expect(createExecutionPlan(valid)).toMatchObject({
      version: 1,
      steps: [
        { id: "baseline", target: "local", transfers: [] },
        { id: "train", target: "cloud", transfers: [] },
        { id: "candidate", target: "local", transfers: [{ from: "train.model", from_target: "cloud", to_target: "local" }] },
        { id: "compare", target: "local", transfers: [] },
      ],
    });
  });

  it("rejects duplicate IDs, non-backward references, and wrong output kinds", () => {
    expect(validatePipeline({ ...valid, steps: [...valid.steps, { id: "train", uses: "train" }] }))
      .toContainEqual(expect.stringMatching(/unique.*train/i));
    expect(validatePipeline({ ...valid, steps: [
      { id: "candidate", uses: "evaluate", with: { model: { from: "train.model" } } },
      { id: "train", uses: "train" },
    ] })).toContainEqual(expect.stringMatching(/candidate.*prior step/i));
    expect(validatePipeline({ ...valid, steps: [{ id: "x", uses: "evaluate", with: { model: { from: "missing.model" } } }] }))
      .toContainEqual(expect.stringMatching(/missing\.model.*prior step/i));
    expect(validatePipeline({ ...valid, steps: [{ id: "train", uses: "train" }, { id: "x", uses: "evaluate", with: { model: { from: "train.report" } } }] }))
      .toContainEqual(expect.stringMatching(/does not produce.*report/i));
  });

  it("requires evaluate model base or a prior model ref and distinct compare report refs", () => {
    const invalidModel = validatePipeline({ version: 1, target: "local", steps: [{ id: "eval", uses: "evaluate", with: { model: "candidate" } }] });
    expect(invalidModel.length).toBeGreaterThan(0);
    expect(invalidModel[0]).toMatch(/model/i);
    expect(validatePipeline({ version: 1, target: "local", steps: [{ id: "compare", uses: "compare", with: { before: { from: "x.report" }, after: { from: "x.report" } } }] }))
      .toContainEqual(expect.stringMatching(/distinct/i));
  });

  it("keeps --only and --skip safe by refusing omitted dependencies", () => {
    expect(() => createExecutionPlan(valid, { only: ["candidate"] })).toThrow(/dependency.*train/i);
    expect(() => createExecutionPlan(valid, { skip: ["train"] })).toThrow(/dependency.*train/i);
    expect(createExecutionPlan(valid, { only: ["baseline"] }).steps.map((step) => step.id)).toEqual(["baseline"]);
  });

  it("provides canonical local and cloud recipes", () => {
    expect(canonicalPipeline("local").steps.map((step) => step.id)).toEqual(["baseline", "train", "candidate", "compare"]);
    expect(createExecutionPlan(canonicalPipeline("local")).steps[0]).toMatchObject({ with: { evaluator: "behavior" } });
    expect(canonicalPipeline("cloud").steps.every((step) => step.target === "cloud")).toBe(true);
  });

  it("rejects ambiguous multiple training artifacts in v1", () => {
    expect(validatePipeline({
      version: 1,
      target: "local",
      steps: [
        { id: "train_a", uses: "train" },
        { id: "train_b", uses: "train" },
      ],
    })).toContainEqual(expect.stringMatching(/at most one train/i));
  });
});
