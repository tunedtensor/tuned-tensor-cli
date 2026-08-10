import { describe, expect, it, vi } from "vitest";
import {
  approvePreparedAction,
  rejectPreparedAction,
  type AgentMutationApi,
} from "../agent-approval.js";
import type { AgentAction } from "../agent-client.js";

const SPEC_ID = "251f122f-dd8e-4894-a0ab-99965e976e29";
const ACTION_ID = "ed8e4bca-ab1c-4c9f-8b65-9f7997f76670";
const CAPABILITY = "local_agent_spec_mutation_guards_v1";

function api(): AgentMutationApi {
  return {
    get: vi.fn(async (path) => path === "/version"
      ? { data: { capabilities: [CAPABILITY] } }
      : { data: { id: SPEC_ID, updated_at: "v1" } }),
    post: vi.fn(async () => ({ data: { id: SPEC_ID } })),
    put: vi.fn(async () => ({ data: { id: SPEC_ID } })),
  };
}

function createAction(): AgentAction {
  return {
    id: ACTION_ID,
    operation: "create_spec",
    title: "Create spec",
    summary: "Create Support",
    risk: "Creates a remote spec",
    status: "proposed",
    arguments: { spec: { name: "Support", examples: [{ input: "a", output: "b" }] } },
  };
}

function updateAction(): AgentAction {
  return {
    id: ACTION_ID,
    operation: "update_spec",
    title: "Update spec",
    summary: "Update Support",
    risk: "Updates a remote spec",
    status: "proposed",
    arguments: {
      spec_id: SPEC_ID,
      changes: { description: "Updated" },
      expected_spec_updated_at: "v1",
    },
  };
}

describe("deterministic local approvals", () => {
  it("seals execution before one guarded create and cannot repeat", async () => {
    const client = api();
    const action = createAction();
    const transitions: string[] = [];
    await expect(approvePreparedAction(action, client, async (value) => {
      transitions.push(value.status ?? "");
    })).resolves.toEqual({ id: SPEC_ID });

    expect(transitions).toEqual(["executing", "completed"]);
    expect(client.post).toHaveBeenCalledWith("/behavior-specs", {
      name: "Support",
      examples: [{ input: "a", output: "b" }],
    }, {
      actionId: action.id,
      operation: "create_spec",
    });
    await expect(approvePreparedAction(action, client, async () => {})).rejects.toThrow(/cannot be approved/i);
    expect(client.post).toHaveBeenCalledTimes(1);
  });

  it("conditionally updates the exact approved spec version", async () => {
    const client = api();
    const action = updateAction();
    await approvePreparedAction(action, client, async () => {});
    expect(client.put).toHaveBeenCalledWith(
      `/behavior-specs/${SPEC_ID}`,
      { description: "Updated" },
      { actionId: ACTION_ID, operation: "update_spec", expectedUpdatedAt: "v1" },
    );
  });

  it("refuses mutation against a server without guard support", async () => {
    const client = api();
    vi.mocked(client.get).mockResolvedValue({ data: { capabilities: [] } });
    const action = createAction();
    await expect(approvePreparedAction(action, client, async () => {})).rejects.toThrow(
      /does not advertise.*mutation guard/i,
    );
    expect(client.post).not.toHaveBeenCalled();
    expect(action.status).toBe("failed");
  });

  it("rejects stale resources and rejection never calls a mutation", async () => {
    const client = api();
    vi.mocked(client.get).mockImplementation(async (path) => path === "/version"
      ? { data: { capabilities: [CAPABILITY] } }
      : { data: { id: SPEC_ID, updated_at: "v2" } });
    const stale = updateAction();
    await expect(approvePreparedAction(stale, client, async () => {})).rejects.toThrow(/changed.*prepare/i);
    expect(client.put).not.toHaveBeenCalled();

    const rejected = createAction();
    await rejectPreparedAction(rejected, async () => {});
    expect(rejected.status).toBe("rejected");
    expect(client.post).not.toHaveBeenCalled();
  });

  it("records an unknown outcome when dispatch may have reached the server", async () => {
    const client = api();
    vi.mocked(client.post).mockRejectedValueOnce(new Error("connection reset after upload"));
    const action = createAction();
    const transitions: string[] = [];

    await expect(approvePreparedAction(action, client, async (value) => {
      transitions.push(value.status ?? "");
    })).rejects.toThrow(/outcome is unknown.*cannot be retried/i);
    expect(transitions).toEqual(["executing", "outcome_unknown"]);
    expect(action.status).toBe("outcome_unknown");
  });

  it("records an unknown outcome when completed state cannot be persisted", async () => {
    const client = api();
    const action = createAction();
    const transitions: string[] = [];
    let calls = 0;

    await expect(approvePreparedAction(action, client, async (value) => {
      calls += 1;
      transitions.push(value.status ?? "");
      if (calls === 2) throw new Error("disk full");
    })).rejects.toThrow(/outcome is unknown.*cannot be retried/i);
    expect(client.post).toHaveBeenCalledTimes(1);
    expect(transitions).toEqual(["executing", "completed", "outcome_unknown"]);
    expect(action.status).toBe("outcome_unknown");
  });
});
