import { describe, expect, it } from "vitest";
import { localBehaviorSpecFileSchema } from "../local-runtime/contracts.js";
import {
  hasLocalOnlySpecFields,
  projectCloudSpec,
  projectLocalSpec,
  unknownProjectSpecKeys,
} from "../project-spec.js";

const sharedSpec = {
  id: "11111111-1111-4111-8111-111111111111",
  name: "Shared Spec",
  description: "Works in both execution modes.",
  base_model: "qwen/qwen3.5-2b-base",
  system_prompt: "Be concise.",
  guidelines: ["Answer directly."],
  constraints: [],
  examples: [
    { input: "One", output: "1" },
    { input: "Two", output: "2" },
  ],
  eval_cases: [
    {
      input: "Write a Python program.",
      runtime: "python",
      tests: [{ expected_stdout: "ok\n" }],
    },
  ],
  hyperparameters: { n_epochs: 2 },
  dataset_prebuilt: {
    training: "train.jsonl",
    validation: "validation.jsonl",
  },
  project_note: "keep this in the project file",
};

describe("project spec projections", () => {
  it("projects only hosted API fields for cloud operations", () => {
    const projection = projectCloudSpec(sharedSpec);

    expect(projection.body).toEqual({
      name: "Shared Spec",
      description: "Works in both execution modes.",
      base_model: "Qwen/Qwen3.5-2B",
      system_prompt: "Be concise.",
      guidelines: ["Answer directly."],
      constraints: [],
      examples: sharedSpec.examples,
      eval_cases: sharedSpec.eval_cases,
    });
    expect(projection.droppedKeys).toEqual([
      "id",
      "hyperparameters",
      "dataset_prebuilt",
      "project_note",
    ]);
  });

  it("projects only TT Local fields for local operations", () => {
    const projection = projectLocalSpec(sharedSpec);

    expect(projection.body).toEqual({
      id: sharedSpec.id,
      name: "Shared Spec",
      description: "Works in both execution modes.",
      base_model: "Qwen/Qwen3.5-2B",
      system_prompt: "Be concise.",
      guidelines: ["Answer directly."],
      constraints: [],
      examples: sharedSpec.examples,
      hyperparameters: sharedSpec.hyperparameters,
      dataset_prebuilt: sharedSpec.dataset_prebuilt,
    });
    expect(projection.droppedKeys).toEqual(["eval_cases", "project_note"]);
    expect(() => localBehaviorSpecFileSchema.parse(projection.body)).not.toThrow();
  });

  it("distinguishes local-only fields from unknown project metadata", () => {
    expect(hasLocalOnlySpecFields(sharedSpec)).toBe(true);
    expect(
      unknownProjectSpecKeys(sharedSpec),
    ).toEqual(["project_note"]);
    expect(hasLocalOnlySpecFields({ name: "Cloud only" })).toBe(false);
  });

  it("keeps foundation hyperparameters local-only and omits them from cloud projection", () => {
    const foundationSpec = {
      engine: "foundation",
      name: "Tiny GPT",
      system_prompt: "You are a helpful assistant.",
      guidelines: ["Answer directly."],
      constraints: [],
      examples: [
        { input: "Hello", output: "Hi there." },
        { input: "Thanks", output: "You're welcome." },
      ],
      foundation: {
        depth: 2,
        pretrain_steps: 2,
        finetune_steps: 2,
        rl_steps: 0,
        vocab_size: 256,
        max_chars: 20_000,
        sequence_length: 64,
        batch_size: 2,
        nproc_per_node: 1,
      },
      project_note: "keep this in the project file",
    };

    const cloud = projectCloudSpec(foundationSpec);
    expect(cloud.body.engine).toBeUndefined();
    expect(cloud.body.foundation).toBeUndefined();
    expect(cloud.body.base_model).toBeUndefined();
    expect(cloud.droppedKeys).toEqual(expect.arrayContaining(["engine", "foundation", "project_note"]));

    const local = projectLocalSpec(foundationSpec);
    expect(local.body).toMatchObject({
      engine: "foundation",
      name: "Tiny GPT",
      foundation: foundationSpec.foundation,
    });
    expect(() => localBehaviorSpecFileSchema.parse(local.body)).not.toThrow();
  });
});
