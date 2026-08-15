import { describe, expect, it, vi } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Value } from "typebox/value";
import {
  boundedToolJson,
  createTunedTensorTools,
  type AgentToolApi,
} from "../agent-tools.js";
import type { AgentAction } from "../agent-client.js";
import { projectCloudSpec, projectLocalSpec } from "../project-spec.js";

const SPEC_ID = "251f122f-dd8e-4894-a0ab-99965e976e29";

function fakeApi(): AgentToolApi {
  return {
    get: vi.fn(async (path) => ({
      data: path === `/behavior-specs/${SPEC_ID}`
        ? { id: SPEC_ID, name: "Untrusted\u001b[2J", updated_at: "v1" }
        : [{ id: SPEC_ID, name: "Support" }],
    })),
    postRead: vi.fn(async () => ({
      data: { estimated_cost_cents: 125, estimated_training_tokens: 1000 },
    })),
    propose: vi.fn(async (action: AgentAction) => action),
  };
}

function tool(name: string, api: AgentToolApi, workspaceRoot?: string) {
  return createTunedTensorTools(api, workspaceRoot ? { workspaceRoot } : undefined)
    .find((candidate) => candidate.name === name)!;
}

describe("Tuned Tensor agent tools", () => {
  it("executes typed read tools immediately and bounds untrusted output", async () => {
    const api = fakeApi();
    const result = await tool("list_specs", api).execute(
      "call-1",
      { page: 1, per_page: 20 },
    );

    expect(api.get).toHaveBeenCalledWith("/behavior-specs", {
      page: 1,
      per_page: 20,
    });
    expect(result.content[0]).toMatchObject({ type: "text" });
    expect((result.content[0] as { text: string }).text.length).toBeLessThanOrEqual(32_000);
    expect(api.propose).not.toHaveBeenCalled();
  });

  it("prepares a spec update without mutating and seals its version", async () => {
    const api = fakeApi();
    const result = await tool("prepare_update_spec", api).execute(
      "call-2",
      { spec_id: SPEC_ID, changes: { description: "Safer" } },
    );

    expect(api.get).toHaveBeenCalledWith(`/behavior-specs/${SPEC_ID}`);
    expect(api.propose).toHaveBeenCalledWith(expect.objectContaining({
      operation: "update_spec",
      arguments: expect.objectContaining({
        spec_id: SPEC_ID,
        expected_spec_updated_at: "v1",
      }),
      status: "proposed",
    }));
    expect(api).not.toHaveProperty("postMutation");
    expect(result.details).toMatchObject({ operation: "update_spec" });
  });

  it("prepares a workspace-scoped local spec without calling the cloud API", async () => {
    const workspace = mkdtempSync(join(tmpdir(), "tt-agent-workspace-"));
    const api = fakeApi();
    const spec = {
      name: "Sentiment classifier",
      description: "Classify short product feedback.",
      base_model: "Qwen/Qwen3.5-2B",
      system_prompt: "Classify the sentiment and return only the label.",
      guidelines: ["Return positive, neutral, or negative."],
      constraints: ["Return one lowercase label."],
      examples: [
        { input: "This is excellent.", output: "positive" },
        { input: "This is disappointing.", output: "negative" },
      ],
    };

    try {
      const output = await tool("prepare_create_local_spec", api, workspace).execute(
        "call-local-spec",
        { directory: "sentiment-demo", spec },
      );

      expect(api.get).not.toHaveBeenCalled();
      expect(api.postRead).not.toHaveBeenCalled();
      expect(api.propose).toHaveBeenCalledWith(expect.objectContaining({
        operation: "create_local_spec",
        arguments: {
          directory: "sentiment-demo",
          spec,
          workspace_fingerprint: expect.stringMatching(/^[a-f0-9]{64}$/),
        },
        preview: {
          directory: "./sentiment-demo",
          spec_path: "./sentiment-demo/tunedtensor.json",
        },
      }));
      expect(output.details).toMatchObject({ operation: "create_local_spec" });
    } finally {
      rmSync(workspace, { recursive: true, force: true });
    }
  });

  it("validates and prepares pipeline plans without direct execution", async () => {
    const api = fakeApi();
    const pipeline = {
      version: 1,
      target: "local",
      steps: [{ id: "baseline", uses: "evaluate", with: { model: "base" } }],
    };
    const validated = await tool("validate_pipeline", api).execute("pipeline-validate", { pipeline });
    expect(validated.details).toMatchObject({ valid: true });

    await tool("prepare_pipeline_run", api).execute("pipeline-prepare", { pipeline, dry_run: true });
    expect(api.propose).toHaveBeenCalledWith(expect.objectContaining({
      operation: "run_pipeline",
      arguments: expect.objectContaining({ dry_run: true }),
    }));
  });

  it("exposes exactly the allowed hosted tool set", () => {
    expect(createTunedTensorTools(fakeApi()).map((candidate) => candidate.name)).toEqual([
      "list_specs", "get_spec", "list_runs", "get_run", "diagnose_run",
      "report_run", "estimate_run", "list_datasets", "get_dataset",
      "list_models", "get_model", "get_balance", "list_transactions",
      "describe_pipeline", "validate_pipeline",
      "prepare_create_spec", "prepare_update_spec", "prepare_pipeline_run",
    ]);
  });

  it("matches the public run-estimate contract", () => {
    const schema = tool("estimate_run", fakeApi()).parameters;
    expect(Value.Check(schema, {
      spec_id: SPEC_ID,
      run: {
        augment: false,
        parent_model_id: SPEC_ID,
        hyperparameters: { n_epochs: 20, lora_alpha: 1 },
      },
    })).toBe(true);
    expect(Value.Check(schema, {
      spec_id: SPEC_ID,
      run: { continuation_model_id: SPEC_ID },
    })).toBe(false);
    expect(Value.Check(schema, {
      spec_id: SPEC_ID,
      run: { hyperparameters: { n_epochs: 21 } },
    })).toBe(false);
    expect(Value.Check(schema, {
      spec_id: SPEC_ID,
      run: { split_ratios: { train: 0.8, validation: 0.2, test: 0.2 } },
    })).toBe(false);
  });

  it("does not advertise unsupported behavior-spec fields", () => {
    const schema = tool("prepare_update_spec", fakeApi()).parameters;
    expect(Value.Check(schema, {
      spec_id: SPEC_ID,
      changes: { eval_cases: [] },
    })).toBe(false);
  });

  it("keeps local-only models out of hosted spec boundaries", () => {
    const museSpec = {
      name: "Muse local",
      base_model: "meta-models/Muse-Glimmer-30B",
      examples: [{ input: "Hello", output: "Hi" }],
    };

    expect(projectLocalSpec(museSpec).body.base_model)
      .toBe("meta-models/Muse-Glimmer-30B");
    expect(() => projectCloudSpec(museSpec))
      .toThrow(/not supported for hosted training/i);
    expect(Value.Check(tool("prepare_create_spec", fakeApi()).parameters, {
      spec: museSpec,
    })).toBe(false);
  });

  it("handles serialization failures and hard-limits oversized results", () => {
    const circular: Record<string, unknown> = {};
    circular.self = circular;
    expect(() => JSON.parse((awaitableText(circular)))).not.toThrow();
    expect(awaitableText({ value: "x".repeat(100_000) }).length).toBeLessThanOrEqual(32_000);
  });

  it("also bounds persisted tool-result details", async () => {
    const api = fakeApi();
    vi.mocked(api.get).mockResolvedValue({ data: { value: "x".repeat(100_000) } });

    const output = await tool("list_specs", api).execute(
      "call-large",
      { page: 1, per_page: 20 },
    );

    expect(JSON.stringify(output.details).length).toBeLessThanOrEqual(32_000);
  });

  it("refuses oversized mutation proposals instead of hiding approval details", async () => {
    const api = fakeApi();

    await expect(tool("prepare_create_spec", api).execute(
      "call-large-action",
      {
        spec: {
          name: "Large spec",
          examples: [{ input: "x".repeat(13_000), output: "y".repeat(13_000) }],
        },
      },
    )).rejects.toThrow(/too large.*direct CLI/i);
    expect(api.propose).not.toHaveBeenCalled();
  });


});

function awaitableText(value: unknown): string {
  // Keep serialization behavior testable independently of a paid/model runtime.
  return boundedToolJson(value);
}
