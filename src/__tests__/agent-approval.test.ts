import { describe, expect, it, vi } from "vitest";
import { existsSync, mkdtempSync as createTemporaryDirectory, realpathSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { writeFile as writeFileAsync } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  approvePreparedAction,
  preflightPreparedAction,
  rejectPreparedAction,
} from "../agent-approval.js";
import type { AgentLocalServer } from "../agent-local-server.js";
import type { AgentAction } from "../agent-client.js";
import { prepareLocalSpecProject } from "../local-spec-workspace.js";
import { prepareLocalPipelineAction } from "../local-pipeline-action.js";

const ACTION_ID = "ed8e4bca-ab1c-4c9f-8b65-9f7997f76670";
const SERVER_ID = "251f122f-dd8e-4894-a0ab-99965e976e29";

// Windows CI may expose TEMP through an 8.3 path alias.
const mkdtempSync = (prefix: string) => realpathSync.native(createTemporaryDirectory(prefix));

function localServer(stop = vi.fn(async () => ({ stopped: true }))) {
  return { server: { stop, status: vi.fn(async () => ({ server_id: SERVER_ID })) } as unknown as AgentLocalServer, stop };
}

function stopAction(): AgentAction {
  return {
    id: ACTION_ID,
    operation: "stop_local_service",
    title: "Stop local model server",
    summary: "Release the GPU used by this session's server.",
    risk: "Stops a local process",
    status: "proposed",
    arguments: { server_id: SERVER_ID },
  };
}

describe("deterministic local approvals", () => {
  it("executes a sealed local pipeline through the deterministic command runner", async () => {
    const workspace = mkdtempSync(join(tmpdir(), "tt-local-pipeline-approval-"));
    const specPath = join(workspace, "tunedtensor.json");
    writeFileSync(join(workspace, "local-runner.json"), JSON.stringify({
      artifactRoot: "artifacts",
    }));
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
        workspace_fingerprint: prepared.workspaceFingerprint,
        config_path: prepared.configPath,
        config_sha256: prepared.configSha256,
        dry_run: prepared.dryRun,
      },
    };
    const transitions: string[] = [];
    const runPipelineCommand = vi.fn(async (command: string[], options: { cwd: string }) => {
      expect(command.slice(0, 2)).toEqual(["pipeline", "run"]);
      expect(command).toContain("--dry-run");
      expect(options.cwd).toBe(workspace);
      expect(command).not.toContain("--file");
      expect(command).not.toContain("--config");
      expect(JSON.parse(readFileSync(command[command.indexOf("--spec") + 1]!, "utf8")))
        .toMatchObject({ name: "Sentiment", pipeline: prepared.pipeline, runtime: { artifactRoot: join(workspace, "artifacts") } });
      return { exitCode: 0, signal: null };
    });

    try {
      await expect(approvePreparedAction(
        action,
        async (value) => { transitions.push(value.status ?? ""); },
        { workspaceRoot: workspace, runPipelineCommand },
      )).resolves.toMatchObject({
        completed: true,
        engine: "adapter",
        spec_path: "./tunedtensor.json",
        config_path: "./local-runner.json",
        dry_run: true,
        command: ["tt", "pipeline", "run", "--dry-run", "--spec", "./tunedtensor.json"],
      });
      expect(transitions).toEqual(["executing", "completed"]);
      expect(runPipelineCommand).toHaveBeenCalledTimes(1);
    } finally {
      rmSync(workspace, { recursive: true, force: true });
    }
  });

  it("rejects a persisted real pipeline action before claiming it", async () => {
    const workspace = mkdtempSync(join(tmpdir(), "tt-local-pipeline-real-action-"));
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
    const prepared = await prepareLocalPipelineAction({ workspaceRoot: workspace });
    const action: AgentAction = {
      id: ACTION_ID,
      operation: "run_local_pipeline",
      title: "Run local pipeline",
      summary: "Run the adapter pipeline.",
      risk: "high",
      status: "proposed",
      arguments: {
        pipeline: prepared.pipeline,
        spec_path: prepared.specPath,
        spec_sha256: prepared.specSha256,
        workspace_fingerprint: prepared.workspaceFingerprint,
        dry_run: false,
      },
    };
    const transitions: string[] = [];
    const runPipelineCommand = vi.fn(async () => ({ exitCode: 0, signal: null }));

    try {
      await expect(approvePreparedAction(
        action,
        async (value) => { transitions.push(value.status ?? ""); },
        {
          workspaceRoot: workspace,
          runPipelineCommand,
        },
      )).rejects.toThrow(/real pipeline execution requires the explicit direct tt pipeline run command/i);
      expect(action.status).toBe("failed");
      expect(transitions).toEqual(["failed"]);
      expect(runPipelineCommand).not.toHaveBeenCalled();
    } finally {
      rmSync(workspace, { recursive: true, force: true });
    }
  });

  it("refuses an approved pipeline in a different workspace with an identical spec", async () => {
    const preparedWorkspace = mkdtempSync(join(tmpdir(), "tt-local-pipeline-prepared-workspace-"));
    const approvalWorkspace = mkdtempSync(join(tmpdir(), "tt-local-pipeline-approval-workspace-"));
    const spec = JSON.stringify({
      name: "Sentiment",
      base_model: "Qwen/Qwen3.5-2B",
      system_prompt: "Classify sentiment.",
      guidelines: ["Return one label."],
      examples: [
        { input: "Great", output: "positive" },
        { input: "Awful", output: "negative" },
      ],
    });
    writeFileSync(join(preparedWorkspace, "tunedtensor.json"), spec);
    writeFileSync(join(approvalWorkspace, "tunedtensor.json"), spec);
    const prepared = await prepareLocalPipelineAction({ workspaceRoot: preparedWorkspace });
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
        workspace_fingerprint: prepared.workspaceFingerprint,
        dry_run: true,
      },
    };
    const runPipelineCommand = vi.fn(async () => ({ exitCode: 0, signal: null }));
    const transitions: string[] = [];

    try {
      await expect(approvePreparedAction(
        action,
        async (value) => { transitions.push(value.status ?? ""); },
        { workspaceRoot: approvalWorkspace, runPipelineCommand },
      )).rejects.toThrow(/workspace changed after.*prepared/i);
      expect(action.status).toBe("proposed");
      expect(transitions).toEqual([]);
      expect(runPipelineCommand).not.toHaveBeenCalled();

      const claimedAction: AgentAction = {
        ...action,
        id: "bc77f76c-28b2-44f3-acb3-57bb554259a7",
        status: "proposed",
      };
      const claimedTransitions: string[] = [];
      await expect(approvePreparedAction(
        claimedAction,
        async (value) => { claimedTransitions.push(value.status ?? ""); },
        {
          workspaceRoot: approvalWorkspace,
          runPipelineCommand,
          durableClaimed: true,
        },
      )).rejects.toThrow(/workspace changed after.*prepared/i);
      expect(claimedAction.status).toBe("failed");
      expect(claimedTransitions).toEqual(["failed"]);
      expect(runPipelineCommand).not.toHaveBeenCalled();
    } finally {
      rmSync(preparedWorkspace, { recursive: true, force: true });
      rmSync(approvalWorkspace, { recursive: true, force: true });
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
        workspace_fingerprint: prepared.workspaceFingerprint,
        dry_run: true,
      },
    };
    writeFileSync(specPath, JSON.stringify({ ...spec, system_prompt: "Changed after review." }));
    const runPipelineCommand = vi.fn();

    try {
      await expect(approvePreparedAction(
        action,
        async () => {},
        { workspaceRoot: workspace, runPipelineCommand },
      )).rejects.toThrow(/changed after.*prepared/i);
      expect(action.status).toBe("failed");
      expect(runPipelineCommand).not.toHaveBeenCalled();
    } finally {
      rmSync(workspace, { recursive: true, force: true });
    }
  });

  it("creates an approved local spec", async () => {
    const workspace = mkdtempSync(join(tmpdir(), "tt-local-spec-approval-"));
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
      const prepared = await prepareLocalSpecProject(workspace, spec);
      const action: AgentAction = {
        id: ACTION_ID,
        operation: "create_local_spec",
        title: "Create local spec",
        summary: "Create ./tunedtensor.json.",
        risk: "Creates local files",
        status: "proposed",
        arguments: {
          spec,
          workspace_fingerprint: prepared.workspaceFingerprint,
        },
      };
      const transitions: string[] = [];

      await expect(approvePreparedAction(action, async (value) => {
        transitions.push(value.status ?? "");
      }, { workspaceRoot: workspace })).resolves.toEqual({
        created: true,
        path: "./tunedtensor.json",
      });

      expect(transitions).toEqual(["executing", "completed"]);
      expect(JSON.parse(readFileSync(join(workspace, "tunedtensor.json"), "utf8")))
        .toEqual({ id: expect.stringMatching(/^[0-9a-f-]{36}$/), ...spec });
    } finally {
      rmSync(workspace, { recursive: true, force: true });
    }
  });

  it("refuses a create proposal that targets a subfolder from an older version", async () => {
    const workspace = mkdtempSync(join(tmpdir(), "tt-local-spec-legacy-"));
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
    const prepared = await prepareLocalSpecProject(workspace, spec);
    const action: AgentAction = {
      id: ACTION_ID,
      operation: "create_local_spec",
      title: "Create local spec",
      summary: "Create ./legacy/tunedtensor.json.",
      risk: "Creates local files",
      status: "proposed",
      arguments: { directory: "legacy", spec, workspace_fingerprint: prepared.workspaceFingerprint },
    };

    try {
      await expect(preflightPreparedAction(action, { workspaceRoot: workspace }))
        .rejects.toThrow(/older tt version/i);
      expect(action.status).toBe("proposed");
      await expect(approvePreparedAction(action, async () => {}, { workspaceRoot: workspace }))
        .rejects.toThrow(/older tt version/i);
      expect(action.status).toBe("failed");
      expect(existsSync(join(workspace, "tunedtensor.json"))).toBe(false);
    } finally {
      rmSync(workspace, { recursive: true, force: true });
    }
  });

  it("records a known local precondition failure without overwriting the target", async () => {
    const workspace = mkdtempSync(join(tmpdir(), "tt-local-spec-race-"));
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
      const prepared = await prepareLocalSpecProject(workspace, spec);
      const action: AgentAction = {
        id: ACTION_ID,
        operation: "create_local_spec",
        title: "Create local spec",
        summary: "Create ./tunedtensor.json.",
        risk: "Creates local files",
        status: "proposed",
        arguments: {
          spec,
          workspace_fingerprint: prepared.workspaceFingerprint,
        },
      };
      const transitions: string[] = [];
      writeFileSync(join(workspace, "tunedtensor.json"), "do not replace");

      await expect(approvePreparedAction(action, async (value) => {
        transitions.push(value.status ?? "");
      }, { workspaceRoot: workspace })).rejects.toThrow(/refusing to overwrite/i);

      expect(action.status).toBe("failed");
      expect(transitions).toEqual(["executing", "failed"]);
      expect(readFileSync(join(workspace, "tunedtensor.json"), "utf8"))
        .toBe("do not replace");
    } finally {
      rmSync(workspace, { recursive: true, force: true });
    }
  });

  it("records a failure without replacing a racing local spec", async () => {
    const workspace = mkdtempSync(join(tmpdir(), "tt-local-spec-collision-"));
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
    const prepared = await prepareLocalSpecProject(workspace, spec);
    const action: AgentAction = {
      id: ACTION_ID,
      operation: "create_local_spec",
      title: "Create local spec",
      summary: "Create ./tunedtensor.json.",
      risk: "Creates local files",
      status: "proposed",
      arguments: {
        spec,
        workspace_fingerprint: prepared.workspaceFingerprint,
      },
    };
    const specPath = join(workspace, "tunedtensor.json");

    try {
      await expect(approvePreparedAction(
        action,
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
      )).rejects.toThrow(/refusing to overwrite/i);

      expect(action.status).toBe("failed");
      expect(transitions).toEqual(["executing", "failed"]);
      expect(readFileSync(specPath, "utf8")).toBe("external content");
    } finally {
      rmSync(workspace, { recursive: true, force: true });
    }
  });

  it("seals execution before one approved action and cannot repeat", async () => {
    const { server, stop } = localServer();
    const action = stopAction();
    const transitions: string[] = [];
    await expect(approvePreparedAction(action, async (value) => {
      transitions.push(value.status ?? "");
    }, { localServer: server })).resolves.toEqual({ stopped: true });

    expect(transitions).toEqual(["executing", "completed"]);
    expect(stop).toHaveBeenCalledWith(SERVER_ID);
    await expect(approvePreparedAction(action, async () => {}, { localServer: server })).rejects.toThrow(/cannot be approved/i);
    expect(stop).toHaveBeenCalledTimes(1);
  });

  it("refuses retired hosted-account operations without dispatching", async () => {
    for (const operation of ["create_spec", "update_spec", "start_label_job", "download_label_job"]) {
      const action: AgentAction = { ...stopAction(), operation, arguments: {} };
      await expect(approvePreparedAction(action, async () => {})).rejects.toThrow(/unsupported prepared operation/i);
      expect(action.status).toBe("failed");
    }
  });

  it("rejection never executes an action", async () => {
    const { stop } = localServer();
    const rejected = stopAction();
    await rejectPreparedAction(rejected, async () => {});
    expect(rejected.status).toBe("rejected");
    expect(stop).not.toHaveBeenCalled();
  });

  it("records an unknown outcome when execution may have had an effect", async () => {
    const { server } = localServer(vi.fn(async () => { throw new Error("process did not confirm exit"); }));
    const action = stopAction();
    const transitions: string[] = [];

    await expect(approvePreparedAction(action, async (value) => {
      transitions.push(value.status ?? "");
    }, { localServer: server })).rejects.toThrow(/outcome is unknown.*cannot be retried/i);
    expect(transitions).toEqual(["executing", "outcome_unknown"]);
    expect(action.status).toBe("outcome_unknown");
  });

  it("records an unknown outcome when completed state cannot be persisted", async () => {
    const { server, stop } = localServer();
    const action = stopAction();
    const transitions: string[] = [];
    let calls = 0;

    await expect(approvePreparedAction(action, async (value) => {
      calls += 1;
      transitions.push(value.status ?? "");
      if (calls === 2) throw new Error("disk full");
    }, { localServer: server })).rejects.toThrow(/outcome is unknown.*cannot be retried/i);
    expect(stop).toHaveBeenCalledTimes(1);
    expect(transitions).toEqual(["executing", "completed", "outcome_unknown"]);
    expect(action.status).toBe("outcome_unknown");
  });
});
