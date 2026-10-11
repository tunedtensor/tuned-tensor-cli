import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { LocalAgentStore } from "../agent-store.js";
import { createLocalAgentClient } from "../local-agent-client.js";
import { createCloudActionContext, approvePreparedAction } from "../agent-approval.js";
import { createTunedTensorTools, type AgentToolApi } from "../agent-tools.js";
import { approvedResultsNote, summarizeLabelJob } from "../agent-labeling.js";
import { LabelingNotStartedError } from "../commands/label.js";
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
afterEach(() => rmSync(root, { recursive: true, force: true }));

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
    const action: AgentAction = {
      id: "download-1", operation: "download_label_job", title: "Download", summary: "", risk: "medium", status: "proposed",
      arguments: { job_id: JOB_ID, spec_path: "./tunedtensor.json", include_unreviewed: false, validation_ratio: 0.1, overwrite: false },
    };
    const labeling = { start: vi.fn(), download: vi.fn(async () => { throw new Error("network down"); }) };
    await expect(approvePreparedAction(action, { get: vi.fn(), post: vi.fn(), put: vi.fn() }, async () => {}, {
      workspaceRoot: root, labeling,
    })).rejects.toThrow("network down");
    expect(action.status).toBe("failed");
  });
});

describe("labeling through the agent client", () => {
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
    expect(prompts[1]).toContain("[Approved action results, recorded by tt]");
    expect(prompts[1]).toContain(JOB_ID);
    expect(prompts[1]).toMatch(/How is it going\?$/);
    await client.runTurn(thread.id, "And now?", () => {});
    expect(prompts[2]).toBe("And now?");
  });
});

describe("approvedResultsNote", () => {
  it("reports each result once", () => {
    const actions = [{ id: "a", title: "", summary: "", risk: "medium", result: { job_id: JOB_ID } }] as AgentAction[];
    expect(approvedResultsNote(actions)).toContain(JOB_ID);
    expect(approvedResultsNote(actions)).toBeUndefined();
  });
});
