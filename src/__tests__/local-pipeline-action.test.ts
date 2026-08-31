import { describe, expect, it } from "vitest";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { prepareLocalPipelineAction } from "../local-pipeline-action.js";

describe("local pipeline agent actions", () => {
  it("derives the foundation pipeline from the sealed workspace spec", async () => {
    const workspace = mkdtempSync(join(tmpdir(), "tt-agent-foundation-pipeline-"));
    writeFileSync(join(workspace, "tunedtensor.json"), JSON.stringify({
      engine: "foundation",
      name: "Tiny support model",
      system_prompt: "Answer support questions.",
      guidelines: ["Be concise."],
      constraints: [],
      examples: [
        { input: "Can I return this?", output: "Yes, within 30 days." },
        { input: "When do you ship?", output: "Within two business days." },
      ],
      foundation: {
        depth: 3,
        pretrain_steps: 4,
        finetune_steps: 5,
        rl_steps: 0,
        vocab_size: 256,
        max_chars: 20_000,
        sequence_length: 64,
        batch_size: 2,
        nproc_per_node: 1,
      },
    }));

    try {
      const prepared = await prepareLocalPipelineAction({ workspaceRoot: workspace });
      expect(prepared).toMatchObject({
        engine: "foundation",
        specPath: "./tunedtensor.json",
        dryRun: true,
        pipeline: {
          runtime: { engine: "foundation" },
          steps: expect.arrayContaining([
            expect.objectContaining({
              id: "pretrain",
              uses: "pretrain",
              with: expect.objectContaining({ depth: 3, steps: 4 }),
            }),
          ]),
        },
      });
    } finally {
      rmSync(workspace, { recursive: true, force: true });
    }
  });

  it("keeps real adapter and foundation execution outside model-mediated approval", async () => {
    const workspace = mkdtempSync(join(tmpdir(), "tt-agent-real-pipeline-"));
    const adapter = {
      name: "Sentiment",
      base_model: "Qwen/Qwen3.5-2B",
      system_prompt: "Classify sentiment.",
      guidelines: ["Return one label."],
      examples: [
        { input: "Great", output: "positive" },
        { input: "Awful", output: "negative" },
      ],
    };
    const foundation = {
      engine: "foundation",
      name: "Tiny support model",
      system_prompt: "Answer support questions.",
      guidelines: ["Be concise."],
      constraints: [],
      examples: [
        { input: "Can I return this?", output: "Yes, within 30 days." },
        { input: "When do you ship?", output: "Within two business days." },
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
    };

    try {
      for (const spec of [adapter, foundation]) {
        writeFileSync(join(workspace, "tunedtensor.json"), JSON.stringify(spec));
        await expect(prepareLocalPipelineAction({
          workspaceRoot: workspace,
          dryRun: false,
        })).rejects.toThrow(/real pipeline execution requires the explicit direct tt pipeline run command/i);
      }
    } finally {
      rmSync(workspace, { recursive: true, force: true });
    }
  });

  it("rejects workspace escapes and cloud-targeted execution", async () => {
    const workspace = mkdtempSync(join(tmpdir(), "tt-agent-pipeline-guard-"));
    writeFileSync(join(workspace, "tunedtensor.json"), JSON.stringify({
      name: "Sentiment",
      base_model: "Qwen/Qwen3.5-2B",
      system_prompt: "Classify sentiment.",
      guidelines: ["Return one label."],
      examples: [
        { input: "Great", output: "positive" },
        { input: "Awful", output: "negative" },
      ],
    }));

    try {
      await expect(prepareLocalPipelineAction({
        workspaceRoot: workspace,
        specPath: "../tunedtensor.json",
      })).rejects.toThrow(/stay inside the current workspace/i);
      await expect(prepareLocalPipelineAction({
        workspaceRoot: workspace,
        pipeline: {
          version: 1,
          target: "cloud",
          steps: [
            { id: "baseline", uses: "evaluate", with: { model: "base", evaluator: "behavior" } },
          ],
        },
      })).rejects.toThrow(/targets cloud execution/i);
    } finally {
      rmSync(workspace, { recursive: true, force: true });
    }
  });
});
