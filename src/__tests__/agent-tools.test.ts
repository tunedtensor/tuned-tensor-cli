import { describe, expect, it, vi } from "vitest";
import { Value } from "typebox/value";
import {
  boundedToolJson,
  createTunedTensorTools,
  type AgentToolApi,
} from "../agent-tools.js";
import type { AgentAction } from "../agent-client.js";

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

function tool(name: string, api: AgentToolApi) {
  return createTunedTensorTools(api).find((candidate) => candidate.name === name)!;
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

  it("exposes exactly the allowed hosted tool set", () => {
    expect(createTunedTensorTools(fakeApi()).map((candidate) => candidate.name)).toEqual([
      "list_specs", "get_spec", "list_runs", "get_run", "diagnose_run",
      "report_run", "estimate_run", "list_datasets", "get_dataset",
      "list_models", "get_model", "get_balance", "list_transactions",
      "prepare_create_spec", "prepare_update_spec",
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
