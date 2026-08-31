import { describe, expect, it, vi } from "vitest";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { writeFile as writeFileAsync } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  approvePreparedAction,
  rejectPreparedAction,
  type AgentMutationApi,
} from "../agent-approval.js";
import type { AgentAction } from "../agent-client.js";
import { prepareLocalSpecProject } from "../local-spec-workspace.js";
import { prepareLocalPipelineAction } from "../local-pipeline-action.js";

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
  it("executes a sealed local pipeline through the deterministic command runner", async () => {
    const workspace = mkdtempSync(join(tmpdir(), "tt-local-pipeline-approval-"));
    const specPath = join(workspace, "tunedtensor.json");
    writeFileSync(specPath, JSON.stringify({
      name: "Sentiment",
      base_model: "Qwen/Qwen3.5-2B",
      system_prompt: "Classify sentiment.",
      guidelines: ["Return one label."],
      examples: [
        { input: "Great", output: "positive" },
        { input: "Awful", output: "negative" },
      ],
    }));
    const prepared = await prepareLocalPipelineAction({
      workspaceRoot: workspace,
      dryRun: true,
    });
    const action: AgentAction = {
      id: ACTION_ID,
      operation: "run_local_pipeline",
      title: "Dry-run local pipeline",
      summary: "Preview the adapter pipeline.",
      risk: "medium",
      status: "proposed",
      arguments: {
        pipeline: prepared.pipeline,
        spec_path: prepared.specPath,
        spec_sha256: prepared.specSha256,
        dry_run: prepared.dryRun,
      },
    };
    const transitions: string[] = [];
    const runPipelineCommand = vi.fn(async (command: string[], options: { cwd: string }) => {
      expect(command.slice(0, 2)).toEqual(["pipeline", "run"]);
      expect(command).toContain("--dry-run");
      expect(options.cwd).toBe(workspace);
      expect(JSON.parse(readFileSync(command[command.indexOf("--file") + 1]!, "utf8")))
        .toEqual(prepared.pipeline);
      expect(JSON.parse(readFileSync(command[command.indexOf("--spec") + 1]!, "utf8")))
        .toMatchObject({ name: "Sentiment" });
      return { exitCode: 0, signal: null };
    });

    try {
      await expect(approvePreparedAction(
        action,
        api(),
        async (value) => { transitions.push(value.status ?? ""); },
        { workspaceRoot: workspace, runPipelineCommand },
      )).resolves.toMatchObject({
        completed: true,
        engine: "adapter",
        spec_path: "./tunedtensor.json",
        dry_run: true,
      });
      expect(transitions).toEqual(["executing", "completed"]);
      expect(runPipelineCommand).toHaveBeenCalledTimes(1);
    } finally {
      rmSync(workspace, { recursive: true, force: true });
    }
  });

  it("refuses an approved pipeline when its sealed spec changed", async () => {
    const workspace = mkdtempSync(join(tmpdir(), "tt-local-pipeline-stale-"));
    const specPath = join(workspace, "tunedtensor.json");
    const spec = {
      name: "Sentiment",
      base_model: "Qwen/Qwen3.5-2B",
      system_prompt: "Classify sentiment.",
      guidelines: ["Return one label."],
      examples: [
        { input: "Great", output: "positive" },
        { input: "Awful", output: "negative" },
      ],
    };
    writeFileSync(specPath, JSON.stringify(spec));
    const prepared = await prepareLocalPipelineAction({ workspaceRoot: workspace });
    const action: AgentAction = {
      id: ACTION_ID,
      operation: "run_local_pipeline",
      title: "Run local pipeline",
      summary: "Run the adapter pipeline.",
      risk: "medium",
      status: "proposed",
      arguments: {
        pipeline: prepared.pipeline,
        spec_path: prepared.specPath,
        spec_sha256: prepared.specSha256,
        dry_run: true,
      },
    };
    writeFileSync(specPath, JSON.stringify({ ...spec, system_prompt: "Changed after review." }));
    const runPipelineCommand = vi.fn();

    try {
      await expect(approvePreparedAction(
        action,
        api(),
        async () => {},
        { workspaceRoot: workspace, runPipelineCommand },
      )).rejects.toThrow(/changed after.*prepared/i);
      expect(action.status).toBe("failed");
      expect(runPipelineCommand).not.toHaveBeenCalled();
    } finally {
      rmSync(workspace, { recursive: true, force: true });
    }
  });

  it("creates an approved local spec without contacting the cloud API", async () => {
    const workspace = mkdtempSync(join(tmpdir(), "tt-local-spec-approval-"));
    const client = api();
    const spec = {
      name: "Sentiment classifier",
      base_model: "Qwen/Qwen3.5-2B" as const,
      system_prompt: "Classify sentiment and return only the label.",
      guidelines: ["Return positive, neutral, or negative."],
      constraints: ["Return one lowercase label."],
      examples: [
        { input: "Excellent work.", output: "positive" },
        { input: "This is disappointing.", output: "negative" },
      ],
    };

    try {
      const prepared = await prepareLocalSpecProject(workspace, "sentiment-demo", spec);
      const action: AgentAction = {
        id: ACTION_ID,
        operation: "create_local_spec",
        title: "Create local spec",
        summary: "Create ./sentiment-demo/tunedtensor.json.",
        risk: "Creates local files",
        status: "proposed",
        arguments: {
          directory: "sentiment-demo",
          spec,
          workspace_fingerprint: prepared.workspaceFingerprint,
        },
      };
      const transitions: string[] = [];

      await expect(approvePreparedAction(action, client, async (value) => {
        transitions.push(value.status ?? "");
      }, { workspaceRoot: workspace })).resolves.toEqual({
        created: true,
        directory: "./sentiment-demo",
        path: "./sentiment-demo/tunedtensor.json",
      });

      expect(transitions).toEqual(["executing", "completed"]);
      expect(client.get).not.toHaveBeenCalled();
      expect(client.post).not.toHaveBeenCalled();
      expect(client.put).not.toHaveBeenCalled();
      expect(JSON.parse(readFileSync(join(workspace, "sentiment-demo", "tunedtensor.json"), "utf8")))
        .toEqual(spec);
    } finally {
      rmSync(workspace, { recursive: true, force: true });
    }
  });

  it("records a known local precondition failure without overwriting the target", async () => {
    const workspace = mkdtempSync(join(tmpdir(), "tt-local-spec-race-"));
    const client = api();
    const spec = {
      name: "Sentiment classifier",
      base_model: "Qwen/Qwen3.5-2B" as const,
      system_prompt: "Classify sentiment and return only the label.",
      guidelines: ["Return positive, neutral, or negative."],
      examples: [
        { input: "Excellent work.", output: "positive" },
        { input: "This is disappointing.", output: "negative" },
      ],
    };

    try {
      const prepared = await prepareLocalSpecProject(workspace, "sentiment-demo", spec);
      const action: AgentAction = {
        id: ACTION_ID,
        operation: "create_local_spec",
        title: "Create local spec",
        summary: "Create ./sentiment-demo/tunedtensor.json.",
        risk: "Creates local files",
        status: "proposed",
        arguments: {
          directory: "sentiment-demo",
          spec,
          workspace_fingerprint: prepared.workspaceFingerprint,
        },
      };
      const transitions: string[] = [];
      mkdirSync(join(workspace, "sentiment-demo"));
      writeFileSync(join(workspace, "sentiment-demo", "sentinel.txt"), "do not replace");

      await expect(approvePreparedAction(action, client, async (value) => {
        transitions.push(value.status ?? "");
      }, { workspaceRoot: workspace })).rejects.toThrow(/refusing to overwrite/i);

      expect(action.status).toBe("failed");
      expect(transitions).toEqual(["executing", "failed"]);
      expect(readFileSync(join(workspace, "sentiment-demo", "sentinel.txt"), "utf8"))
        .toBe("do not replace");
      expect(client.post).not.toHaveBeenCalled();
      expect(client.put).not.toHaveBeenCalled();
    } finally {
      rmSync(workspace, { recursive: true, force: true });
    }
  });

  it("records an unknown outcome without deleting a racing local spec", async () => {
    const workspace = mkdtempSync(join(tmpdir(), "tt-local-spec-collision-"));
    const client = api();
    const transitions: string[] = [];
    const spec = {
      name: "Sentiment classifier",
      base_model: "Qwen/Qwen3.5-2B" as const,
      system_prompt: "Classify sentiment.",
      guidelines: ["Return one sentiment label."],
      examples: [
        { input: "Great.", output: "positive" },
        { input: "Awful.", output: "negative" },
      ],
    };
    const prepared = await prepareLocalSpecProject(workspace, "sentiment-demo", spec);
    const action: AgentAction = {
      id: ACTION_ID,
      operation: "create_local_spec",
      title: "Create local spec",
      summary: "Create ./sentiment-demo/tunedtensor.json.",
      risk: "Creates local files",
      status: "proposed",
      arguments: {
        directory: "sentiment-demo",
        spec,
        workspace_fingerprint: prepared.workspaceFingerprint,
      },
    };
    const specPath = join(workspace, "sentiment-demo", "tunedtensor.json");

    try {
      await expect(approvePreparedAction(
        action,
        client,
        async (next) => {
          if (next.status) transitions.push(next.status);
        },
        {
          workspaceRoot: workspace,
          localFileOperations: {
            writeFile: async (path, data, options) => {
              await writeFileAsync(path, "external content", options);
              await writeFileAsync(path, data, options);
            },
          },
        },
      )).rejects.toThrow(/outcome is unknown/i);

      expect(action.status).toBe("outcome_unknown");
      expect(transitions).toEqual(["executing", "outcome_unknown"]);
      expect(readFileSync(specPath, "utf8")).toBe("external content");
      expect(client.get).not.toHaveBeenCalled();
      expect(client.post).not.toHaveBeenCalled();
    } finally {
      rmSync(workspace, { recursive: true, force: true });
    }
  });

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
