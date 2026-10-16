import { execFile } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import type { StreamFn } from "@earendil-works/pi-agent-core";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TunedTensorAgentSession } from "../../agent.js";
import { createHash } from "node:crypto";
import { inspectLocalSpec, readSpecHistory } from "../../spec-workspace.js";
import { LocalAgentStore } from "../../agent-store.js";
import { createLocalAgentClient } from "../../local-agent-client.js";
import type { LocalPipelineCommandRunner } from "../../local-pipeline-action.js";
import { parquetWriteBuffer } from "hyparquet-writer";
import { recordAssetDownload } from "../../local-runtime/assets.js";
import type { BackgroundRunExecutor } from "../../local-background-run-action.js";
import { createBackgroundRunStore } from "../../local-runtime/background-runs.js";
import { hostname } from "node:os";

const execFileAsync = promisify(execFile);
type Context = Parameters<StreamFn>[1];
type Message = Awaited<ReturnType<Awaited<ReturnType<StreamFn>>["result"]>>;
type ToolArgs = Extract<Message["content"][number], { type: "toolCall" }>["arguments"];
type Reply = string | { tool: string; args: ToolArgs } | { error: string };

const spec = {
  name: "Feedback labels",
  base_model: "Qwen/Qwen3.5-2B",
  system_prompt: "Classify feedback as positive or negative.",
  guidelines: ["Return one lowercase label."],
  examples: [
    { input: "I love it", output: "positive" },
    { input: "It broke", output: "negative" },
  ],
};

let root: string;
beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "tt-conversation-workflow-"));
  vi.stubEnv("TUNED_TENSOR_HOME", join(root, "home"));
});
afterEach(() => {
  vi.unstubAllEnvs();
  rmSync(root, { recursive: true, force: true });
});

// Only the provider stream is scripted. Pi still validates arguments, executes
// the real tools, feeds their results back to the model, and enforces its budget.
function conversation(
  replies: Reply[] | ((context: Context, call: number) => Reply),
  runCommand?: LocalPipelineCommandRunner,
  backgroundRuns?: BackgroundRunExecutor,
) {
  const contexts: Context[] = [];
  const streamSimple = vi.fn((_model: unknown, context: Context) => {
    const call = contexts.length;
    contexts.push({ ...context, messages: structuredClone(context.messages) });
    const reply = typeof replies === "function" ? replies(context, call) : replies[call];
    if (reply === undefined) throw new Error(`Unexpected model call ${call + 1}`);
    const message: Message = {
      role: "assistant",
      api: "openai-completions",
      provider: "fixture",
      model: "scripted",
      timestamp: Date.now(),
      usage: {
        input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
      content: typeof reply === "string"
        ? [{ type: "text", text: reply }]
        : "tool" in reply
          ? [{ type: "toolCall", id: `call-${call}`, name: reply.tool, arguments: reply.args }]
          : [],
      stopReason: typeof reply === "string" ? "stop" : "tool" in reply ? "toolUse" : "error",
      ...(typeof reply !== "string" && "error" in reply ? { errorMessage: reply.error } : {}),
    };
    return {
      async *[Symbol.asyncIterator]() {
        yield { type: "start", partial: message };
        if (typeof reply === "string") {
          yield { type: "text_delta", contentIndex: 0, delta: reply, partial: message };
        }
        yield message.stopReason === "error"
          ? { type: "error", reason: "error", error: message }
          : { type: "done", reason: message.stopReason, message };
      },
      result: async () => message,
    };
  });
  const forbidden = vi.fn(async (): Promise<never> => { throw new Error("Unexpected cloud API call"); });
  const store = new LocalAgentStore(join(root, "agent"));
  const newClient = (cloudEnabled = false) => createLocalAgentClient({
    store,
    cloudEnabled,
    workspaceRoot: root,
    selection: { provider: "fixture", model: "scripted", thinking: "off" },
    modelRuntime: {
      getProviders: () => [{ id: "fixture" }],
      getModels: () => [{ id: "scripted", provider: "fixture" }],
      getModel: () => ({ id: "scripted", provider: "fixture" }),
      hasConfiguredAuth: () => true,
      streamSimple,
    },
    toolApi: { get: forbidden, postRead: forbidden, propose: async (action) => action },
    mutationApi: { get: forbidden, post: forbidden, put: forbidden },
    runPipelineCommand: runCommand,
    backgroundRuns,
  });
  const client = newClient();
  const output: string[] = [];
  const errors: string[] = [];
  const newSession = () => new TunedTensorAgentSession({
    client,
    io: { write: (text) => output.push(text), writeError: (text) => errors.push(text), clear: () => {} },
  });
  return { client, newClient, store, session: newSession(), newSession, output, errors, contexts, streamSimple, forbidden };
}

describe("conversation workflow contracts (scripted provider, real agent and tools)", () => {
  it("splits local data after approval, survives resume, and attaches the recorded files to a training preview", async () => {
    const source = JSON.stringify(spec, null, 2) + "\n";
    writeFileSync(join(root, "tunedtensor.json"), source);
    const data = Array.from({ length: 20 }, (_, i) => JSON.stringify({
      messages: [{ role: "user", content: `Feedback ${i}` }, { role: "assistant", content: i % 2 ? "positive" : "negative" }],
      metadata: { category: i % 2 ? "positive" : "negative" },
    })).join("\n") + "\n";
    writeFileSync(join(root, "data.jsonl"), data);
    const dataset = { training: "./split/training.jsonl", validation: "./split/validation.jsonl", format: "chat_jsonl" };
    const c = conversation([
      { tool: "inspect_dataset_file", args: { file: "data.jsonl" } },
      { tool: "prepare_dataset_split", args: { file: "data.jsonl", output: "split", train: .8, validation: .2, test: 0, seed: "123", stratify_by: "metadata.category" } },
      "Review the 80/20 split with seed 123.",
      { tool: "get_local_spec", args: {} },
      { tool: "prepare_update_local_spec", args: { expected_sha256: createHash("sha256").update(source).digest("hex"), changes: { dataset_prebuilt: dataset } } },
      "Review the dataset paths in your spec.",
      { tool: "prepare_pipeline_run", args: {} },
      "The saved spec is ready for a dry-run.",
    ]);
    await c.session.handleLine("Split data.jsonl 80/20 for train and validation with seed 123 and balanced categories.");
    expect(c.errors).toEqual([]);
    expect(c.session.snapshot().pendingActions).toMatchObject([{ operation: "split_local_dataset" }]);
    expect(existsSync(join(root, "split"))).toBe(false);
    expect(c.output.join(" ")).not.toContain("workspace_fingerprint");
    await c.session.handleLine("/approve");
    expect(JSON.parse(readFileSync(join(root, "split/split-manifest.json"), "utf8"))).toMatchObject({ counts: { training: 16, validation: 4, test: 0 }, dataset_prebuilt: dataset });
    expect(readFileSync(join(root, "data.jsonl"), "utf8")).toBe(data);
    expect(readFileSync(join(root, "tunedtensor.json"), "utf8")).toBe(source);
    const resumed = c.newSession();
    await resumed.handleLine(`/resume ${c.session.snapshot().thread!.id}`);
    await resumed.handleLine("Use the files you just created in my spec.");
    expect(JSON.stringify(c.contexts[3])).toContain("split/training.jsonl");
    await resumed.handleLine("/approve");
    expect(JSON.parse(readFileSync(join(root, "tunedtensor.json"), "utf8"))).toMatchObject({ ...spec, dataset_prebuilt: dataset });
    await resumed.handleLine("Preview training.");
    expect(c.errors).toEqual([]);
    expect(resumed.snapshot().pendingActions).toMatchObject([{ operation: "run_local_pipeline", arguments: { dry_run: true } }]);
    expect(c.forbidden).not.toHaveBeenCalled();
  });

  it("discovers cached Parquet data, inspects both splits and attaches it only after approval", async () => {
    const repo = "org/downloaded-coding";
    const revision = "c".repeat(40);
    const snapshot = join(root, "hub", "datasets--org--downloaded-coding", "snapshots", revision);
    mkdirSync(join(snapshot, "data"), { recursive: true });
    for (const split of ["train", "test"]) writeFileSync(join(snapshot, `data/${split}.parquet`), new Uint8Array(parquetWriteBuffer({
      columnData: [{ name: "prompt", data: [`${split} prompt`], type: "STRING" }, { name: "completion", data: ["result"], type: "STRING" }],
    })));
    await recordAssetDownload({ kind: "dataset", repo, revision, snapshotPath: snapshot });
    const source = JSON.stringify(spec, null, 2) + "\n";
    writeFileSync(join(root, "tunedtensor.json"), source);
    const dataset = { huggingface: { repo, revision, columns: { input: "prompt", output: "completion" } },
      training: "data/train.parquet", test: "data/test.parquet" };
    const c = conversation([
      { tool: "get_local_spec", args: {} },
      { tool: "list_local_datasets", args: {} },
      { tool: "inspect_local_dataset", args: { repo, revision, file: dataset.training } },
      { tool: "inspect_local_dataset", args: { repo, revision, file: dataset.test } },
      { tool: "prepare_update_local_spec", args: {
        expected_sha256: createHash("sha256").update(source).digest("hex"), changes: { dataset_prebuilt: dataset },
      } },
      "The downloaded training and test files are ready to attach for approval.",
      { tool: "get_local_spec", args: {} },
      { tool: "prepare_pipeline_run", args: {} },
      "The dry-run uses the saved dataset reference.",
    ]);
    await c.session.handleLine("Use our downloaded dataset for fine-tuning; preserve the rest of my spec.");
    expect(c.errors).toEqual([]);
    expect(c.session.snapshot().pendingActions).toHaveLength(1);
    expect(readFileSync(join(root, "tunedtensor.json"), "utf8")).toBe(source);
    expect(JSON.stringify(c.contexts[4])).toContain('parquet_metadata');
    expect(JSON.stringify(c.contexts[4])).toContain('completion');
    await c.session.handleLine("/approve");
    expect(JSON.parse(readFileSync(join(root, "tunedtensor.json"), "utf8"))).toEqual({ ...spec, id: expect.any(String), dataset_prebuilt: dataset });
    const resumed = c.newSession();
    await resumed.handleLine(`/resume ${c.session.snapshot().thread!.id}`);
    await resumed.handleLine("Preview the fine-tuning workflow without starting training.");
    expect(c.errors).toEqual([]);
    expect(resumed.snapshot().pendingActions).toMatchObject([{ operation: "run_local_pipeline", arguments: { dry_run: true } }]);
    expect(JSON.stringify(c.contexts[7])).toContain('data/train.parquet');
    expect(c.forbidden).not.toHaveBeenCalled();
  });

  it("reports a failed summary visibly and retains the conversation for retry", async () => {
    const c = conversation([{ error: "Summary provider unavailable" }]);
    const thread = await c.client.createThread();
    const state = await c.store.load(thread.id);
    state.messages = [{ role: "user", content: "Past discussion. ".repeat(3000), timestamp: 1 }];
    await c.store.save(state);
    const errors: string[] = [];
    const result = await c.client.runTurn(thread.id, "Continue", (event) => {
      if (event.type === "error") errors.push(String(event.payload.message));
    });
    expect(result.status).toBe("failed");
    expect(errors.join("\n")).toContain("Summary provider unavailable");
    const saved = await c.store.load(thread.id);
    expect(saved.summary).toBeUndefined();
    expect(saved.messages[0]).toEqual(state.messages[0]);
  });

  it("automatically summarizes long history, resumes it, and preserves pending approvals", async () => {
    const c = conversation([
      "Goal: Python coding agent. Muse is downloaded and verified. Never start training.",
      "Yes, Muse is already verified in cache.",
      "We are building a Python coding agent. Training remains unstarted.",
    ]);
    const thread = await c.client.createThread();
    const state = await c.store.load(thread.id);
    state.messages = [
      { role: "user", content: "Python coding agent. Never train. " + "Past discussion. ".repeat(3000), timestamp: 1 },
      { role: "assistant", content: [{ type: "text", text: "Muse verified in cache." }], timestamp: 2 },
    ];
    state.actions = [{ id: "pending-spec", title: "Edit spec", summary: "Awaiting approval", risk: "low", status: "proposed" }];
    await c.store.save(state);
    const events: string[] = [];
    const result = await c.client.runTurn(thread.id, "Do we have Muse already?", (event) => { events.push(JSON.stringify(event)); });
    expect(result.status).toBe("completed");
    expect(result.response).toContain("verified in cache");
    const saved = await c.store.load(thread.id);
    expect(saved.summary).toMatchObject({ text: expect.stringContaining("Never start training"), messageCount: 2 });
    expect(saved.messages.slice(0, 2)).toEqual(state.messages);
    expect(saved.actions).toEqual(state.actions);
    expect(events.join("\n")).toContain("Conversation summary saved");
    const followup = await c.newClient().runTurn(thread.id, "What was my goal?", () => {});
    expect(followup.status).toBe("completed");
    expect(c.streamSimple).toHaveBeenCalledTimes(3);
    expect(JSON.stringify(c.contexts[0])).toContain("context summarization assistant");
    expect(JSON.stringify(c.contexts[2])).toContain("Earlier conversation summary");
    expect(JSON.stringify(c.contexts[2])).not.toContain("Past discussion.");
    expect((await c.store.load(thread.id)).actions).toEqual(state.actions);
  });

  it.each(["dataset", "base_model"] as const)("resumes and approves a %s download exactly once", async (kind) => {
    writeFileSync(join(root, "tunedtensor.json"), JSON.stringify({
      ...spec,
      dataset_prebuilt: {
        huggingface: { repo: "org/dataset", columns: { input: "text", output: "label" } },
        training: "train.parquet", test: "test.parquet",
      },
    }));
    const runner = vi.fn<LocalPipelineCommandRunner>(async (args) => {
      expect(args.slice(0, 2)).toEqual([kind === "dataset" ? "datasets" : "models", "prefetch"]);
      expect(JSON.parse(readFileSync(args[2]!, "utf8")).base_model).toBe(spec.base_model);
      return { exitCode: 0, signal: null };
    });
    const c = conversation([
      { tool: "prepare_download", args: { kind } },
      "The download is ready for approval.",
    ], runner);
    await c.session.handleLine(`Download the ${kind} in my spec.`);
    expect(c.errors).toEqual([]);
    expect(runner).not.toHaveBeenCalled();
    const resumed = c.newSession();
    await resumed.handleLine(`/resume ${c.session.snapshot().thread!.id}`);
    expect(resumed.snapshot().pendingActions[0].operation).toBe("prefetch_local");
    await resumed.handleLine("/approve");
    expect(c.errors).toEqual([]);
    expect(c.output.join("")).toContain("Download finished.");
    expect((await c.store.list())[0].actions[0].status).toBe("completed");
    expect(resumed.snapshot().pendingActions).toEqual([]);
    await resumed.handleLine("/approve");
    expect(runner).toHaveBeenCalledTimes(1);
    expect(c.streamSimple).toHaveBeenCalledTimes(2);
    expect(c.forbidden).not.toHaveBeenCalled();
  });

  it("creates a reviewed project, resumes the conversation, and approves the same plan through the real CLI", async () => {
    const plans: Array<{ dry_run: boolean; steps: Array<{ id: string; uses: string }> }> = [];
    const runCommand = vi.fn<LocalPipelineCommandRunner>(async (args, options) => {
      const { stdout } = await execFileAsync(process.execPath, [
        "--import", import.meta.resolve("tsx"), "--input-type=module", "--eval",
        `import { createProgram } from ${JSON.stringify(new URL("../../cli.ts", import.meta.url).href)};
         await createProgram("workflow-test").parseAsync(process.argv.slice(1), { from: "user" });`,
        "--", "--json", ...args,
      ], { cwd: options.cwd, env: process.env, signal: options.signal, timeout: 15_000 });
      plans.push(JSON.parse(stdout));
      return { exitCode: 0, signal: null };
    });
    const c = conversation([
      { tool: "prepare_create_local_spec", args: { directory: "feedback", spec } },
      "The project is ready for review. Use /approve to create it.",
      { tool: "prepare_pipeline_run", args: { spec_path: "feedback/tunedtensor.json" } },
      "The dry-run plan is ready for /approve.",
    ], runCommand);

    await c.session.handleLine("Create a feedback classifier project with positive and negative labels.");
    expect(c.errors).toEqual([]);
    expect(existsSync(join(root, "feedback"))).toBe(false);
    expect(c.session.snapshot().pendingActions).toHaveLength(1);
    await c.session.handleLine("/approve");
    expect(JSON.parse(readFileSync(join(root, "feedback/tunedtensor.json"), "utf8"))).toMatchObject(spec);

    // A fresh UI session reads durable conversation state, including tool results.
    const resumed = c.newSession();
    const threadId = c.session.snapshot().thread!.id;
    await resumed.handleLine(`/resume ${threadId}`);
    await resumed.handleLine("Preview fine-tuning and compare it with the base model; do not train yet.");
    expect(runCommand).not.toHaveBeenCalled();
    expect(resumed.snapshot().pendingActions[0]).toMatchObject({
      operation: "run_local_pipeline", arguments: { dry_run: true, spec_path: "./feedback/tunedtensor.json" },
    });
    expect(c.contexts[2].messages.some((message) => message.role === "toolResult")).toBe(true);
    await resumed.handleLine("/approve");
    expect(runCommand).toHaveBeenCalledTimes(1);
    expect(plans).toEqual([expect.objectContaining({
      dry_run: true,
      steps: [
        expect.objectContaining({ id: "baseline", uses: "evaluate" }),
        expect.objectContaining({ id: "train", uses: "train" }),
        expect.objectContaining({ id: "candidate", uses: "evaluate" }),
        expect.objectContaining({ id: "compare", uses: "compare" }),
      ],
    })]);
    expect((await c.store.load(threadId)).actions.map((action) => action.status)).toEqual(["completed", "completed"]);
    expect(resumed.snapshot().pendingActions).toEqual([]);
    expect(c.errors).toEqual([]);
    await resumed.handleLine("/approve");
    expect(runCommand).toHaveBeenCalledTimes(1);
    expect(c.streamSimple).toHaveBeenCalledTimes(4); // Approvals never call the model.
    expect(c.errors.join("")).toContain("There is no pending action");
    expect(c.forbidden).not.toHaveBeenCalled();
  }, 20_000);

  it.each(["reject", "spec changed", "config changed"])("does not execute a prepared run after %s", async (reason) => {
    writeFileSync(join(root, "tunedtensor.json"), JSON.stringify(spec));
    const runCommand = vi.fn<LocalPipelineCommandRunner>();
    const c = conversation([
      { tool: "prepare_pipeline_run", args: {} },
      "Ready for review.",
    ], runCommand);
    await c.session.handleLine("Fine-tune the current project.");
    expect(c.session.snapshot().pendingActions).toHaveLength(1);
    if (reason === "spec changed") {
      writeFileSync(join(root, "tunedtensor.json"), JSON.stringify({ ...spec, system_prompt: "Changed task" }));
    } else if (reason === "config changed") {
      writeFileSync(join(root, "local-runner.json"), JSON.stringify({ dryRun: true }));
    }
    await c.session.handleLine(reason === "reject" ? "/reject" : "/approve");
    expect(runCommand).not.toHaveBeenCalled();
    const state = await c.store.load(c.session.snapshot().thread!.id);
    expect(state.actions[0].status).toBe(reason === "reject" ? "rejected" : "failed");
    expect(c.session.snapshot().pendingActions).toEqual([]);
    if (reason !== "reject") expect(c.errors.join("")).toMatch(/changed.*prepare it again/);
  });

  it("refuses a persisted proposal changed to request real training", async () => {
    writeFileSync(join(root, "tunedtensor.json"), JSON.stringify(spec));
    const runCommand = vi.fn<LocalPipelineCommandRunner>();
    const c = conversation([
      { tool: "prepare_pipeline_run", args: {} }, "The dry-run is ready for review.",
    ], runCommand);
    await c.session.handleLine("Preview fine-tuning this project.");
    const state = await c.store.load(c.session.snapshot().thread!.id);
    expect(state.actions[0]).toMatchObject({ arguments: { dry_run: true } });
    (state.actions[0].arguments as Record<string, unknown>).dry_run = false;
    await c.store.save(state);
    await c.session.handleLine("/approve");
    expect(runCommand).not.toHaveBeenCalled();
    expect(c.errors.join("")).toMatch(/dry-run only|explicit direct/);
    expect((await c.store.load(state.thread.id)).actions[0].status).toBe("failed");
    expect(c.session.snapshot().pendingActions).toEqual([]);
  });

  it("records a dry-run execution failure without claiming completion or retrying the action", async () => {
    writeFileSync(join(root, "tunedtensor.json"), JSON.stringify(spec));
    const runCommand = vi.fn<LocalPipelineCommandRunner>(async () => ({ exitCode: 1, signal: null }));
    const c = conversation([
      { tool: "prepare_pipeline_run", args: {} }, "Ready for review.",
    ], runCommand);
    await c.session.handleLine("Fine-tune this project.");
    await c.session.handleLine("/approve");
    expect((await c.store.load(c.session.snapshot().thread!.id)).actions[0].status).toBe("failed");
    expect(c.errors.join("")).toContain("Approved pipeline dry-run exited with code 1");
    expect(c.session.snapshot().pendingActions).toEqual([]);
    await c.session.handleLine("/approve");
    expect(runCommand).toHaveBeenCalledTimes(1);
    expect(c.streamSimple).toHaveBeenCalledTimes(2);
  });

  it("surfaces provider failure as a failed turn with no executable action", async () => {
    const c = conversation([{ error: "Provider unavailable" }]);
    const thread = await c.client.createThread();
    const errors: unknown[] = [];
    const result = await c.client.runTurn(thread.id, "Fine-tune this project.", (event) => {
      if (event.type === "error") errors.push(event.payload.message);
    });
    expect(errors).toEqual(["Provider unavailable"]);
    expect(result).toMatchObject({ status: "failed", actions: [] });
    expect((await c.store.load(thread.id)).actions).toEqual([]);
    expect(c.streamSimple).toHaveBeenCalledTimes(1);
  });

  it("starts every turn from the current system prompt and stores only the conversation", async () => {
    const c = conversation(["First answer.", "Second answer."]);
    const thread = await c.client.createThread();
    await c.client.runTurn(thread.id, "Hello", () => {});
    const saved = await c.store.load(thread.id);
    expect(saved.messages.map((message) => (message as { role: string }).role)).toEqual(["user", "assistant"]);

    // Threads saved with a system message must not pin an outdated prompt.
    saved.messages.unshift({ role: "system", content: "STALE PROMPT", toolsAdded: [], timestamp: 0 });
    await c.store.save(saved);
    await c.client.runTurn(thread.id, "Again", () => {});
    const sent = JSON.stringify(c.contexts[1].messages);
    expect(sent).not.toContain("STALE PROMPT");
    expect(sent).toContain("local Tuned Tensor assistant");
    expect(c.contexts[1].messages.filter((message) => message.role === "system")).toHaveLength(1);
    expect((await c.store.load(thread.id)).messages.map((message) => (message as { role: string }).role))
      .toEqual(["user", "assistant", "user", "assistant"]);
  });

  it("refreshes account instructions and tool declarations when a thread resumes after logout", async () => {
    const c = conversation(["Signed in.", "Signed out."]);
    const signedIn = c.newClient(true);
    const thread = await signedIn.createThread();
    await signedIn.runTurn(thread.id, "Hello", () => {});
    const initialSystem = c.contexts[0].messages.find((message) => message.role === "system")!;
    expect(initialSystem.content).toContain("Account tools inspect cloud");
    expect(initialSystem.toolsAdded?.some((tool) => tool.name === "list_specs")).toBe(true);

    // Include both leading and mid-conversation system messages from Pi 1.x.
    const saved = await c.store.load(thread.id);
    saved.messages.unshift(initialSystem);
    saved.messages.push({ ...initialSystem, content: "STALE ACCOUNT INSTRUCTIONS" });
    await c.store.save(saved);
    await c.newClient(false).runTurn(thread.id, "Again", () => {});
    const systems = c.contexts[1].messages.filter((message) => message.role === "system");
    expect(systems).toHaveLength(1);
    expect(systems[0].content).toContain("No TT access token is configured");
    expect(systems[0].toolsAdded?.map((tool) => tool.name)).toContain("get_local_spec");
    expect(systems[0].toolsAdded?.map((tool) => tool.name)).not.toContain("list_specs");
    expect(JSON.stringify(c.contexts[1].messages)).not.toContain("STALE ACCOUNT INSTRUCTIONS");
    expect((await c.store.load(thread.id)).messages.map((message) => (message as { role: string }).role))
      .toEqual(["user", "assistant", "user", "assistant"]);
  });

  it("stops a model that repeatedly calls tools at the production turn budget", async () => {
    const c = conversation(() => ({ tool: "describe_pipeline", args: { engine: "adapter" } }));
    const thread = await c.client.createThread();
    const statuses: unknown[] = [];
    await c.client.runTurn(thread.id, "Explain the adapter workflow.", (event) => {
      if (event.type === "tool_result") statuses.push(event.payload.status);
    });
    expect(statuses.filter((status) => status === "success")).toHaveLength(12);
    expect(statuses.at(-1)).toBe("error");
    expect(c.streamSimple).toHaveBeenCalledTimes(13);
    expect((await c.store.load(thread.id)).actions).toEqual([]);
  });
});


describe("spec-centered conversation", () => {
  it("reads, proposes, reviews, approves, resumes and previews the edited spec through the real tools", async () => {
    const source = JSON.stringify({ ...spec, hyperparameters: { n_epochs: 3 }, evaluation: { inference: { maxNewTokens: 8, device: "cuda" } }, runtime: { artifactRoot: "artifacts" } });
    writeFileSync(join(root, "tunedtensor.json"), source);
    const sha256 = createHash("sha256").update(source).digest("hex");
    const command = vi.fn<LocalPipelineCommandRunner>(async (args) => {
      const sealedSpec = JSON.parse(readFileSync(args[args.indexOf("--spec") + 1], "utf8"));
      expect(sealedSpec.system_prompt).toBe("Classify feedback. Do not add commentary.");
      expect(sealedSpec.hyperparameters.n_epochs).toBe(3);
      expect(sealedSpec.examples).toEqual(spec.examples);
      expect(sealedSpec.evaluation.inference).toMatchObject({ maxNewTokens: 16, device: "cuda" });
      expect(sealedSpec.runtime.artifactRoot).toBe(join(root, "artifacts"));
      expect(args).toContain("--dry-run");
      return { exitCode: 0 };
    });
    const c = conversation([
      { tool: "get_local_spec", args: {} },
      { tool: "prepare_update_local_spec", args: { expected_sha256: sha256, changes: { system_prompt: "Classify feedback. Do not add commentary.", evaluation: { inference: { maxNewTokens: 16 } } } } },
      "Review the proposed spec change before approving.",
      { tool: "prepare_pipeline_run", args: {} },
      "The pipeline preview uses the approved spec.",
    ], command);
    await c.session.handleLine("Make the model avoid commentary and increase evaluation generation to 16 tokens while keeping my training settings.");
    expect(c.errors).toEqual([]);
    expect(readFileSync(join(root, "tunedtensor.json"), "utf8")).toBe(source);
    expect(c.output.join("\n")).toContain("@@ system_prompt @@");
    expect(c.session.snapshot().pendingActions[0].operation).toBe("update_local_spec");
    await c.session.handleLine("/approve");
    expect((await readSpecHistory(root))[0].status).toBe("applied");
    const resumed = c.newSession();
    await resumed.handleLine(`/resume ${c.session.snapshot().thread!.id}`);
    await resumed.handleLine("Preview the pipeline for my current spec.");
    expect(resumed.snapshot().pendingActions[0].arguments).toMatchObject({ spec_sha256: (await inspectLocalSpec(root)).sha256 });
    await resumed.handleLine("/approve");
    expect(command).toHaveBeenCalledTimes(1);
    expect(c.forbidden).not.toHaveBeenCalled();
    expect(c.errors).toEqual([]);
  });

  it.each(["reject", "stale"])("does not save an edit after %s", async reason => {
    const source = JSON.stringify(spec); writeFileSync(join(root, "tunedtensor.json"), source);
    const c = conversation([
      { tool: "prepare_update_local_spec", args: { expected_sha256: createHash("sha256").update(source).digest("hex"), changes: { name: "Proposed name" } } },
      "Ready for review.",
    ]);
    await c.session.handleLine("Rename my behavior spec.");
    if (reason === "stale") writeFileSync(join(root, "tunedtensor.json"), JSON.stringify({ ...spec, name: "External edit" }));
    await c.session.handleLine(reason === "reject" ? "/reject" : "/approve");
    expect((await inspectLocalSpec(root)).document).toMatchObject({ name: reason === "reject" ? spec.name : "External edit" });
    expect(await readSpecHistory(root)).toEqual([]);
    const state = await c.store.load(c.session.snapshot().thread!.id);
    expect(state.actions[0].status).toBe(reason === "reject" ? "rejected" : "failed");
  });
});

describe("background training runs (scripted provider, real agent and tools)", () => {
  function executor() {
    const store = createBackgroundRunStore(join(root, "background-runs"));
    const launch = vi.fn(async (run: Parameters<BackgroundRunExecutor["launch"]>[0]) =>
      // Stand in for the detached supervisor: this live process "owns" the run.
      await store.update(run.id, (current) => { current.process = { host: hostname(), supervisor_pid: process.pid }; }));
    const usage = {
      sampled_at: "2026-10-16T10:00:00.000Z",
      cpu: { model: "Fixture CPU", cores: 8, utilization_percent: 12, load_average: [0.5, 0.4, 0.3] as [number, number, number] },
      memory: { used_bytes: 8e9, total_bytes: 32e9 },
      gpus: [{ index: 0, name: "Fixture GPU", vendor: "nvidia" as const, unified_memory: false, utilization_percent: 3, memory_used_bytes: 1e9, memory_total_bytes: 24e9 }],
    };
    return { store, launch, value: { store, launch, maxParallel: () => 1, sampleUsage: async () => usage } satisfies BackgroundRunExecutor };
  }
  const reason = "Measure whether the two new examples fix negation.";

  it("refuses a real run until a dry-run of the same spec bytes completed", async () => {
    writeFileSync(join(root, "tunedtensor.json"), JSON.stringify(spec));
    const runs = executor();
    const c = conversation([
      { tool: "prepare_background_run", args: { reason } }, "I need a dry-run first.",
    ], undefined, runs.value);
    await c.session.handleLine("Train it for real.");
    expect(JSON.stringify(c.contexts[1]!.messages.at(-1))).toMatch(/completed dry-run of this exact spec/);
    expect(c.session.snapshot().pendingActions).toEqual([]);
    expect(runs.launch).not.toHaveBeenCalled();
  });

  it("starts after approval, keeps the audit record, reports status, and tells the model when it finishes", async () => {
    writeFileSync(join(root, "tunedtensor.json"), JSON.stringify(spec));
    const runs = executor();
    const runCommand = vi.fn<LocalPipelineCommandRunner>(async () => ({ exitCode: 0 }));
    let backgroundId = "";
    const c = conversation((_context, call) => [
      { tool: "prepare_pipeline_run", args: {} }, "Dry-run ready.",
      { tool: "prepare_background_run", args: { reason } }, "Ready to start.",
      { tool: "get_background_run", args: { id: backgroundId.slice(0, 8) } }, "It is running.",
      "It finished.",
    ][call] as Reply, runCommand, runs.value);

    await c.session.handleLine("Preview training.");
    await c.session.handleLine("/approve");
    expect(runCommand).toHaveBeenCalledTimes(1);

    await c.session.handleLine("Looks good, start the real training.");
    const pending = c.session.snapshot().pendingActions;
    expect(pending).toHaveLength(1);
    expect(pending[0]).toMatchObject({ operation: "start_local_pipeline", risk: "high", arguments: { reason } });
    expect(pending[0]!.preview).toMatchObject({ target: "local", max_parallel_runs: 1, resources: { gpus: [{ name: "Fixture GPU" }] } });
    expect(runs.launch).not.toHaveBeenCalled();

    await c.session.handleLine("/approve");
    expect(runs.launch).toHaveBeenCalledTimes(1);
    expect(c.errors.join("")).toBe("");
    const [record] = await runs.store.list();
    backgroundId = record!.id;
    const thread = await c.store.load(c.session.snapshot().thread!.id);
    const dryRunAction = thread.actions.find((action) => action.operation === "run_local_pipeline")!;
    const startAction = thread.actions.find((action) => action.operation === "start_local_pipeline")!;
    expect(startAction.status).toBe("completed");
    expect(record).toMatchObject({
      status: "starting",
      reason,
      requested_by: { via: "agent", thread_id: thread.thread.id, action_id: startAction.id, dry_run_action_id: dryRunAction.id },
      choices: { engine: "adapter", spec_path: "./tunedtensor.json", spec_sha256: createHash("sha256").update(JSON.stringify(spec)).digest("hex"), target: "local" },
      resources_at_launch: { cpu_cores: 8 },
    });
    expect(JSON.parse(readFileSync(runs.store.paths(record!.id).spec, "utf8"))).toMatchObject({ name: spec.name, pipeline: { version: 1 } });

    // The conversation continues while the run is active, and a second run is refused.
    await runs.store.update(record!.id, (run) => { run.status = "running"; run.started_at = new Date().toISOString(); });
    writeFileSync(runs.store.paths(record!.id).log, "baseline eval 4/10\n");
    await c.session.handleLine("How is it going?");
    expect(JSON.stringify(c.contexts[5]!.messages.at(-1))).toMatch(/baseline eval 4\/10/);
    await expect(runs.value.store.create({ ...record!, sealed_spec: {} }, { maxParallel: 1 })).rejects.toThrow(/already active/);

    await runs.store.update(record!.id, (run) => {
      run.status = "succeeded";
      run.finished_at = new Date().toISOString();
      run.outcome = { exit_code: 0, metrics: { comparison: { pass_rate_delta: 0.25 } } };
    });
    await c.session.handleLine("Anything new?");
    expect(JSON.stringify(c.contexts[6]!.messages.at(-1))).toMatch(/Background run outcomes.*succeeded.*pass_rate_delta/);
    expect((await runs.store.get(record!.id)).reported_at).toBeDefined();
  });

  it("refuses to start when the spec changed after the proposal", async () => {
    writeFileSync(join(root, "tunedtensor.json"), JSON.stringify(spec));
    const runs = executor();
    const c = conversation([
      { tool: "prepare_pipeline_run", args: {} }, "Dry-run ready.",
      { tool: "prepare_background_run", args: { reason } }, "Ready to start.",
    ], vi.fn<LocalPipelineCommandRunner>(async () => ({ exitCode: 0 })), runs.value);
    await c.session.handleLine("Preview training.");
    await c.session.handleLine("/approve");
    await c.session.handleLine("Start it.");
    writeFileSync(join(root, "tunedtensor.json"), JSON.stringify({ ...spec, guidelines: ["Changed after review."] }));
    await c.session.handleLine("/approve");
    expect(c.errors.join("")).toMatch(/spec changed after this pipeline action was prepared/);
    expect(runs.launch).not.toHaveBeenCalled();
    expect(await runs.store.list()).toEqual([]);
  });

  it("cancels an active run only after approval and records who and why", async () => {
    writeFileSync(join(root, "tunedtensor.json"), JSON.stringify(spec));
    const runs = executor();
    const created = await runs.store.create({
      reason,
      requested_by: { user: "someone", host: hostname(), via: "cli" },
      choices: { engine: "adapter", spec_path: "./tunedtensor.json", spec_sha256: "b".repeat(64), config_path: null, config_sha256: null, steps: [], target: "local", workspace_root: root },
      sealed_spec: {},
    });
    // This live process stands in for the supervisor, so the run is not reconciled to lost.
    await runs.store.update(created.id, (run) => { run.status = "running"; run.process = { host: hostname(), supervisor_pid: process.pid }; });
    const kill = vi.spyOn(process, "kill").mockImplementation(() => true);
    try {
      const c = conversation([
        { tool: "prepare_cancel_background_run", args: { id: created.id.slice(0, 8), reason: "Started on the wrong spec" } }, "Ready to stop it.",
      ], undefined, runs.value);
      await c.session.handleLine("Stop that run.");
      expect((await runs.store.get(created.id)).status).toBe("running");
      await c.session.handleLine("/approve");
      expect(await runs.store.get(created.id)).toMatchObject({
        status: "cancel_requested",
        cancel: { reason: "Started on the wrong spec", requested_by: { via: "agent" } },
      });
      expect(kill).toHaveBeenCalledWith(process.pid, "SIGTERM");
    } finally {
      kill.mockRestore();
    }
  });
});
