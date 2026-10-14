import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { LocalAgentStore } from "../agent-store.js";
import { createLocalAgentClient } from "../local-agent-client.js";
import { createCloudActionContext, approvePreparedAction, preflightPreparedAction, PreparedActionWorkspaceMismatchError } from "../agent-approval.js";
import { createTunedTensorTools, type AgentToolApi } from "../agent-tools.js";
import { approvedResultsNote, summarizeLabelJob, prepareLabelDownload } from "../agent-labeling.js";
import { LabelingNotStartedError } from "../commands/label.js";
import { createAccountLabelingExecutor } from "../cli.js";
import { buildSystemMessage, examplesFromChatJsonl } from "../local-runtime/dataset.js";
import type { AgentAction } from "../agent-client.js";

const JOB_ID = "55555555-5555-4555-8555-555555555555";
const CLOUD_CONTEXT = createCloudActionContext("https://account-a.example", "tt_secret_token_value");
const SPEC = {
  name: "triage",
  system_prompt: "Triage the email.",
  guidelines: ["Answer as Category / Priority / Action."],
  constraints: [],
  base_model: "Qwen/Qwen3.5-2B",
  examples: [{ input: "Invoice overdue", output: "Billing / High / Reply" }],
};

let root: string;
beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "tt-agent-label-"));
  writeFileSync(join(root, "tunedtensor.json"), JSON.stringify(SPEC, null, 2));
  writeFileSync(join(root, "emails.jsonl"), '{"input":"Lunch?"}\n{"input":"Server down"}\n');
});
afterEach(() => {
  vi.unstubAllGlobals();
  rmSync(root, { recursive: true, force: true });
});

function fakeApi(get: AgentToolApi["get"] = vi.fn(async () => ({}))): AgentToolApi {
  return { get, postRead: vi.fn(), propose: async (action) => action };
}

function tool(api: AgentToolApi, name: string) {
  return createTunedTensorTools(api, { workspaceRoot: root }).find((candidate) => candidate.name === name)!;
}

describe("agent labeling tools", () => {
  it("are offered only with a workspace and a TT account", () => {
    const names = (options: Parameters<typeof createTunedTensorTools>[1]) =>
      createTunedTensorTools(fakeApi(), options).map((candidate) => candidate.name);
    const labeling = ["list_label_jobs", "get_label_job", "prepare_label_job", "prepare_label_download"];
    expect(names({ workspaceRoot: root })).toEqual(expect.arrayContaining(labeling));
    expect(names({ workspaceRoot: root, localOnly: true })).not.toContain("prepare_label_job");
    expect(names({})).not.toContain("prepare_label_job");
  });

  it("prepares a labeling job with the file and spec fingerprints", async () => {
    const action = (await tool(fakeApi(), "prepare_label_job").execute("p", { file: "emails.jsonl" })).details as AgentAction;
    expect(action).toMatchObject({
      operation: "start_label_job",
      status: "proposed",
      arguments: {
        file: "./emails.jsonl",
        file_sha256: expect.stringMatching(/^[a-f0-9]{64}$/),
        format: "jsonl",
        spec_path: "./tunedtensor.json",
        spec_sha256: expect.stringMatching(/^[a-f0-9]{64}$/),
      },
      preview: { rows: 2, spec_examples: 1, cost: expect.stringMatching(/TT credits/) },
    });
  });

  it("refuses unsafe paths, invalid files and specs without examples", async () => {
    const prepare = tool(fakeApi(), "prepare_label_job");
    await expect(prepare.execute("p", { file: "../emails.jsonl" })).rejects.toThrow(/inside the current workspace/);
    await expect(prepare.execute("p", { file: "/etc/passwd" })).rejects.toThrow(/inside the current workspace/);
    symlinkSync(join(root, "emails.jsonl"), join(root, "link.jsonl"));
    await expect(prepare.execute("p", { file: "link.jsonl" })).rejects.toThrow(/symlinks/);
    writeFileSync(join(root, "bad.jsonl"), '{"text":"no input"}\n');
    await expect(prepare.execute("p", { file: "bad.jsonl" })).rejects.toThrow(/string "input"/);
    writeFileSync(join(root, "tunedtensor.json"), JSON.stringify({ ...SPEC, examples: [] }));
    await expect(prepare.execute("p", { file: "emails.jsonl" })).rejects.toThrow(/Add a few labeled examples/);
  });

  it("reports progress and the next step", async () => {
    const get = vi.fn(async () => ({ data: {
      id: JOB_ID, name: "emails", status: "labeling", est_cost_cents: 40,
      counts: { total: 200, pending: 50, labeled: 150, accepted: 0, edited: 0, rejected: 0, failed: 0 },
    } }));
    const status = (await tool(fakeApi(get), "get_label_job").execute("s", { id: JOB_ID })).details as Record<string, unknown>;
    expect(get).toHaveBeenCalledWith(`/labeling-jobs/${JOB_ID}`);
    expect(status).toMatchObject({
      status: "labeling",
      progress: { labeled_or_failed: 150, total: 200, percent: 75 },
      next: expect.stringMatching(/Check again later/),
    });
    expect(summarizeLabelJob({ data: { status: "awaiting_review" } }).next).toMatch(/prepare_label_download/);
  });

  it("prepares a download only for finished jobs with enough rows", async () => {
    const job = (fields: Record<string, unknown>) => vi.fn(async () => ({ data: {
      id: JOB_ID, name: "emails", status: "awaiting_review",
      counts: { total: 10, pending: 0, labeled: 10, accepted: 0, edited: 0, rejected: 0, failed: 0 },
      ...fields,
    } }));
    await expect(tool(fakeApi(job({ status: "labeling" })), "prepare_label_download").execute("d", { job_id: JOB_ID }))
      .rejects.toThrow(/once labeling finishes/);
    await expect(tool(fakeApi(job({})), "prepare_label_download").execute("d", { job_id: JOB_ID }))
      .rejects.toThrow(/accept rows .* or to include unreviewed drafts/);

    const action = (await tool(fakeApi(job({})), "prepare_label_download")
      .execute("d", { job_id: JOB_ID, include_unreviewed: true })).details as AgentAction;
    expect(action).toMatchObject({
      operation: "download_label_job",
      arguments: { job_id: JOB_ID, include_unreviewed: true, validation_ratio: 0.1, overwrite: false },
      preview: { output: ["./datasets/emails/training.jsonl", "./datasets/emails/validation.jsonl", "./datasets/emails/export.json"] },
    });

    mkdirSync(join(root, "datasets", "emails"), { recursive: true });
    writeFileSync(join(root, "datasets", "emails", "training.jsonl"), "");
    await expect(tool(fakeApi(job({})), "prepare_label_download").execute("d", { job_id: JOB_ID, include_unreviewed: true }))
      .rejects.toThrow(/whether to overwrite/);
  });
});

describe("labeling approval", () => {
  async function preparedDownload(): Promise<AgentAction> {
    const prepared = await prepareLabelDownload(root, {
      name: "emails", status: "awaiting_review", counts: { accepted: 10 },
    }, { job_id: JOB_ID });
    return { id: "download-1", operation: "download_label_job", title: "Download", summary: "", risk: "medium", status: "proposed", arguments: prepared.args };
  }
  async function prepared(): Promise<AgentAction> {
    return (await tool(fakeApi(), "prepare_label_job").execute("p", { file: "emails.jsonl", name: "Inbox" })).details as AgentAction;
  }

  it("starts the job with workspace paths resolved", async () => {
    const action = await prepared();
    const labeling = { start: vi.fn(async () => ({ job_id: JOB_ID })), download: vi.fn() };
    const output = await approvePreparedAction(action, { get: vi.fn(), post: vi.fn(), put: vi.fn() }, async () => {}, {
      workspaceRoot: root, labeling,
    });
    expect(output).toEqual({ job_id: JOB_ID });
    expect(action.status).toBe("completed");
    expect(labeling.start).toHaveBeenCalledWith(expect.objectContaining({
      file: expect.stringMatching(/emails\.jsonl$/),
      format: "jsonl",
      name: "Inbox",
      specPath: expect.stringMatching(/tunedtensor\.json$/),
    }));
  });

  it("never uploads a file that changed after preparation", async () => {
    const action = await prepared();
    writeFileSync(join(root, "emails.jsonl"), '{"input":"Different"}\n');
    const labeling = { start: vi.fn(), download: vi.fn() };
    await expect(approvePreparedAction(action, { get: vi.fn(), post: vi.fn(), put: vi.fn() }, async () => {}, {
      workspaceRoot: root, labeling,
    })).rejects.toThrow(/changed after this labeling job was prepared/);
    expect(labeling.start).not.toHaveBeenCalled();
    expect(action.status).toBe("failed");
  });

  it("marks a refused start as failed, but a lost response as unknown", async () => {
    const refused = await prepared();
    await expect(approvePreparedAction(refused, { get: vi.fn(), post: vi.fn(), put: vi.fn() }, async () => {}, {
      workspaceRoot: root,
      labeling: { start: vi.fn(async () => { throw new LabelingNotStartedError("Insufficient credits"); }), download: vi.fn() },
    })).rejects.toThrow("Insufficient credits");
    expect(refused.status).toBe("failed");

    const lost = await prepared();
    await expect(approvePreparedAction(lost, { get: vi.fn(), post: vi.fn(), put: vi.fn() }, async () => {}, {
      workspaceRoot: root,
      labeling: { start: vi.fn(async () => { throw new TypeError("fetch failed"); }), download: vi.fn() },
    })).rejects.toThrow(/outcome is unknown.*labeling jobs \(tt cloud label list\)/);
    expect(lost.status).toBe("outcome_unknown");
  });

  it("marks a failed download as safely failed rather than unknown", async () => {
    const action = await preparedDownload();
    const labeling = { start: vi.fn(), download: vi.fn(async () => { throw new Error("network down"); }) };
    await expect(approvePreparedAction(action, { get: vi.fn(), post: vi.fn(), put: vi.fn() }, async () => {}, {
      workspaceRoot: root, labeling,
    })).rejects.toThrow("network down");
    expect(action.status).toBe("failed");
  });

  it("binds both actions to the workspace even when another workspace has identical files", async () => {
    const other = mkdtempSync(join(tmpdir(), "tt-label-other-"));
    try {
      for (const file of ["tunedtensor.json", "emails.jsonl"]) {
        writeFileSync(join(other, file), readFileSync(join(root, file)));
      }
      for (const action of [await prepared(), await preparedDownload()]) {
        await expect(preflightPreparedAction(action, { workspaceRoot: other }))
          .rejects.toBeInstanceOf(PreparedActionWorkspaceMismatchError);
        expect(action.status).toBe("proposed");
      }
    } finally { rmSync(other, { recursive: true, force: true }); }
  });

  it("refuses a changed download spec and paths replaced with symlinks after preparation", async () => {
    const action = await preparedDownload();
    const labeling = { start: vi.fn(), download: vi.fn() };
    writeFileSync(join(root, "tunedtensor.json"), JSON.stringify({ ...SPEC, system_prompt: "Different task" }));
    await expect(approvePreparedAction(action, { get: vi.fn(), post: vi.fn(), put: vi.fn() }, async () => {}, {
      workspaceRoot: root, labeling,
    })).rejects.toThrow(/spec changed/);
    expect(labeling.download).not.toHaveBeenCalled();

    const next = await preparedDownload();
    symlinkSync(root, join(root, "datasets"));
    await expect(approvePreparedAction(next, { get: vi.fn(), post: vi.fn(), put: vi.fn() }, async () => {}, {
      workspaceRoot: root, labeling,
    })).rejects.toThrow(/symlinks/);
    expect(labeling.download).not.toHaveBeenCalled();
  });

  it("passes the approved bytes even if the file changes before the upload", async () => {
    const action = await prepared();
    const original = readFileSync(join(root, "emails.jsonl"));
    const start = vi.fn(async (args) => {
      writeFileSync(args.file, '{"input":"changed during cloud push"}\n');
      expect(args.fileBytes).toEqual(original);
    });
    await approvePreparedAction(action, { get: vi.fn(), post: vi.fn(), put: vi.fn() }, async () => {}, {
      workspaceRoot: root, labeling: { start, download: vi.fn() },
    });
    expect(start).toHaveBeenCalledOnce();
  });
});

describe("labeling through the agent client", () => {
  it("runs approved upload and download through the production account executor", async () => {
    const original = readFileSync(join(root, "emails.jsonl"), "utf8");
    const cloudSpecId = "66666666-6666-4666-8666-666666666666";
    const job = { id: JOB_ID, name: "emails", status: "awaiting_review", counts: { accepted: 12 } };
    const seen: string[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url, init) => {
      const path = new URL(String(url)).pathname;
      seen.push(`${init?.method ?? "GET"} ${path}`);
      if (path === "/api/v1/behavior-specs") {
        writeFileSync(join(root, "emails.jsonl"), '{"input":"changed during spec push"}\n');
        return Response.json({ data: { id: cloudSpecId, name: SPEC.name } });
      }
      if (path === "/api/v1/labeling-jobs/upload-url") {
        expect(JSON.parse(init.body).size).toBe(Buffer.byteLength(original));
        return Response.json({ data: { path: "s3://b/source.jsonl", upload_url: "https://upload.example/source", method: "PUT" } });
      }
      if (path === "/source") {
        expect(await init.body.text()).toBe(original);
        return new Response(null, { status: 200 });
      }
      if (path === "/api/v1/labeling-jobs") {
        expect(JSON.parse(init.body).behavior_spec_id).toBe(cloudSpecId);
        return Response.json({ data: { ...job, status: "preparing", est_cost_cents: 12 } });
      }
      if (path === `/api/v1/labeling-jobs/${JOB_ID}`) {
        // A rename after preparation must not redirect the approved output path.
        return Response.json({ data: { ...job, name: "renamed", system_message: buildSystemMessage(SPEC) } });
      }
      if (path.endsWith("/rows")) {
        return Response.json({ data: Array.from({ length: 12 }, (_, i) => ({
          id: `row-${i}`, row_index: i, input: `email ${i}`, teacher_output: `category ${i}`,
          final_output: null, status: "accepted", sanitization_status: "clean",
        })), meta: { total: 12, page: 1, per_page: 100 } });
      }
      throw new Error(`Unexpected request: ${path}`);
    }));
    const labeling = createAccountLabelingExecutor({ baseUrl: "https://account.example", apiKey: "test-account-key" });
    const mutations = { get: vi.fn(), post: vi.fn(), put: vi.fn() };
    const start = (await tool(fakeApi(), "prepare_label_job").execute("p", { file: "emails.jsonl" })).details as AgentAction;
    await expect(approvePreparedAction(start, mutations, async () => {}, { workspaceRoot: root, labeling }))
      .resolves.toMatchObject({ job_id: JOB_ID, cloud_spec_id: cloudSpecId });
    const download = (await tool(fakeApi(async () => ({ data: job })), "prepare_label_download")
      .execute("d", { job_id: JOB_ID })).details as AgentAction;
    const output = await approvePreparedAction(download, mutations, async () => {}, { workspaceRoot: root, labeling });
    expect(output).toMatchObject({ dataset_prebuilt: {
      training: "./datasets/emails/training.jsonl", validation: "./datasets/emails/validation.jsonl",
    }, manifest: { teacher_instructions_match_spec: true } });
    const training = await examplesFromChatJsonl(join(root, "datasets/emails/training.jsonl"));
    const validation = await examplesFromChatJsonl(join(root, "datasets/emails/validation.jsonl"));
    expect(training.length + validation.length).toBe(12);
    expect(seen).toEqual([
      "POST /api/v1/behavior-specs", "POST /api/v1/labeling-jobs/upload-url", "PUT /source",
      "POST /api/v1/labeling-jobs", `GET /api/v1/labeling-jobs/${JOB_ID}`, `GET /api/v1/labeling-jobs/${JOB_ID}/rows`,
    ]);
  });

  it("keeps the preview, binds approval to the account, and tells the model the job ID next turn", async () => {
    const prompts: string[] = [];
    const labeling = {
      start: vi.fn(async () => ({ job_id: JOB_ID, name: "emails", status: "preparing", estimated_cost_cents: 12, cloud_spec_id: "spec-1" })),
      download: vi.fn(),
    };
    const client = createLocalAgentClient({
      store: new LocalAgentStore(join(root, ".agent")),
      workspaceRoot: root,
      selection: { provider: "openai", model: "gpt-5.2", thinking: "off" },
      modelRuntime: {
        getProviders: () => [{ id: "openai" }],
        getModels: () => [{ id: "gpt-5.2", provider: "openai", reasoning: true }],
        getModel: () => ({ id: "gpt-5.2", provider: "openai", reasoning: true }),
        hasConfiguredAuth: () => true,
      },
      cloudEnabled: true,
      cloudContext: CLOUD_CONTEXT,
      toolApi: fakeApi(),
      mutationApi: { get: vi.fn(), post: vi.fn(), put: vi.fn() },
      labeling,
      createAgent: (options) => ({
        state: { messages: [] },
        subscribe: () => () => {},
        abort: () => {},
        prompt: async (text: string) => {
          prompts.push(text);
          if (prompts.length === 1) {
            await options.tools.find((candidate) => candidate.name === "prepare_label_job")!
              .execute("p", { file: "emails.jsonl" });
          }
        },
      }),
    });
    const thread = await client.createThread();
    const turn = await client.runTurn(thread.id, "Label emails.jsonl with the teacher", () => {});
    const action = turn.actions[0]!;
    expect(action.cloud_context).toEqual(CLOUD_CONTEXT);
    expect(action.preview).toMatchObject({ rows: 2, api_origin: CLOUD_CONTEXT.origin });

    const events: unknown[] = [];
    await client.approveAction(action.id, (event) => events.push(event));
    expect(labeling.start).toHaveBeenCalledTimes(1);
    expect(events).toContainEqual({ type: "action_result", payload: expect.objectContaining({
      status: "completed",
      output: expect.objectContaining({ job_id: JOB_ID }),
    }) });

    await client.runTurn(thread.id, "How is it going?", () => {});
    expect(prompts[1]).toContain("[Action outcomes, recorded by tt]");
    expect(prompts[1]).toContain(JOB_ID);
    expect(prompts[1]).toMatch(/How is it going\?$/);
    await client.runTurn(thread.id, "And now?", () => {});
    expect(prompts[2]).toBe("And now?");
  });
});

describe("approvedResultsNote", () => {
  it("carries durable outcomes even when an action has no model result payload", () => {
    const actions = ["completed", "failed", "rejected", "outcome_unknown", "proposed", "executing"].map((status) => ({
      id: status, operation: "update_local_spec", status, title: "", summary: "", risk: "medium",
      arguments: { spec_path: "./tunedtensor.json", changes: { large: "do not repeat this diff" } },
    })) as AgentAction[];
    const note = approvedResultsNote(actions)!;
    for (const status of ["completed", "failed", "rejected", "outcome_unknown"]) {
      expect(note).toContain(`"status":"${status}"`);
    }
    expect(note).toContain('"spec_path":"./tunedtensor.json"');
    expect(note).not.toMatch(/"proposed"|"executing"|do not repeat this diff/);
    expect(approvedResultsNote(actions)).toBeUndefined();
    expect(actions[4]?.result_reported).toBeUndefined();
  });

  it("reports each result once", () => {
    const actions = [{ id: "a", title: "", summary: "", risk: "medium", result: { job_id: JOB_ID } }] as AgentAction[];
    expect(approvedResultsNote(actions)).toContain(JOB_ID);
    expect(approvedResultsNote(actions)).toBeUndefined();
  });
});
