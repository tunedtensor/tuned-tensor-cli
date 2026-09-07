import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createServer, type Server } from "node:http";
import { once } from "node:events";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createPiModelRuntime, getAgentAuthPath, getAgentModelsPath, resolveAgentModel } from "../agent-model.js";
import { loginAgentProvider, setAgentModel } from "../agent-control.js";
import { getAgentSelection, readConfig, writeConfig } from "../config.js";

const token = `tt_${"a".repeat(48)}`;
const originalHome = process.env.TUNED_TENSOR_HOME;
let root: string;
let server: Server | undefined;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "tt-managed-agent-"));
  process.env.TUNED_TENSOR_HOME = root;
});

afterEach(async () => {
  if (server) {
    const closed = once(server, "close");
    server.close();
    server.closeAllConnections();
    await closed;
    server = undefined;
  }
  if (originalHome === undefined) delete process.env.TUNED_TENSOR_HOME;
  else process.env.TUNED_TENSOR_HOME = originalHome;
  rmSync(root, { recursive: true, force: true });
});

async function inferenceServer() {
  const requests: Array<{ path: string; authorization: string | undefined; body: Record<string, unknown> }> = [];
  server = createServer(async (request, response) => {
    let body = "";
    for await (const chunk of request) body += chunk.toString();
    requests.push({ path: request.url!, authorization: request.headers.authorization, body: JSON.parse(body) });
    response.writeHead(200, { "content-type": "text/event-stream" });
    response.write(`data: ${JSON.stringify({
      id: "completion-1", object: "chat.completion.chunk", created: 1, model: "server-model",
      choices: [{ index: 0, delta: { role: "assistant", content: "Hello" }, finish_reason: null }],
    })}\n\n`);
    response.write(`data: ${JSON.stringify({
      id: "completion-1", object: "chat.completion.chunk", created: 1, model: "server-model",
      choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
      usage: { prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 },
    })}\n\n`);
    response.end("data: [DONE]\n\n");
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("Missing test address");
  return { url: `http://127.0.0.1:${address.port}`, requests };
}

describe("managed inference and BYO routing", () => {
  it("streams through the TT proxy with one account token and rejects managed model overrides", async () => {
    const { url, requests } = await inferenceServer();
    writeConfig({ api_key: token, base_url: url });
    mkdirSync(join(root, "agent"));
    writeFileSync(getAgentModelsPath(), JSON.stringify({ providers: {
      tunedtensor: { baseUrl: "http://untrusted.invalid", modelOverrides: { managed: { baseUrl: "http://untrusted.invalid" } } },
      openrouter: { headers: { "X-Custom": "preserved" } },
    } }));
    const runtime = await createPiModelRuntime();
    const selection = getAgentSelection({})!;
    const { model } = resolveAgentModel(runtime, selection);
    expect(model).toMatchObject({ id: "managed", provider: "tunedtensor", maxTokens: 4096 });
    const result = await runtime.completeSimple(model as never, {
      messages: [{ role: "user", content: "Hi", timestamp: 1 }],
    });
    expect(result.stopReason).toBe("stop");
    expect(result.content).toContainEqual(expect.objectContaining({ type: "text", text: "Hello" }));
    expect(requests).toHaveLength(1);
    expect(requests[0]).toMatchObject({
      path: "/api/v1/agent/chat/completions",
      authorization: `Bearer ${token}`,
      body: { model: "managed", stream: true, max_tokens: 4096 },
    });
    expect(readFileSync(getAgentModelsPath(), "utf8")).not.toContain(token);
    expect(JSON.parse(readFileSync(getAgentModelsPath(), "utf8")).providers.tunedtensor).toBeUndefined();
    expect(!existsSync(getAgentAuthPath()) || !readFileSync(getAgentAuthPath(), "utf8").includes(token)).toBe(true);
    expect(() => resolveAgentModel(runtime, { ...selection, model: "user-picked-model" })).toThrow(/Unknown model/);
  });

  it("saves /login tunedtensor in account config and activates managed inference without provider credentials", async () => {
    const runtime = await createPiModelRuntime();
    expect(runtime.hasConfiguredAuth("tunedtensor")).toBe(false);
    await loginAgentProvider(runtime, "tunedtensor", token, { baseUrl: "https://staging.example" });
    expect(readConfig().api_key).toBe(token);
    expect(readConfig().base_url).toBe("https://staging.example");
    expect(getAgentSelection({})).toMatchObject({ provider: "tunedtensor", model: "managed" });
    expect(runtime.hasConfiguredAuth("tunedtensor")).toBe(true);
    const auth = await runtime.getAuth(runtime.getModel("tunedtensor", "managed")!);
    expect(auth?.auth.apiKey).toBe(token);
    expect(!existsSync(getAgentAuthPath()) || !readFileSync(getAgentAuthPath(), "utf8").includes(token)).toBe(true);
    writeConfig({});
    expect(runtime.hasConfiguredAuth("tunedtensor")).toBe(false);
    expect(() => resolveAgentModel(runtime, { provider: "tunedtensor", model: "managed", thinking: "off" })).toThrow(/tt auth login/);
  });

  it("sends arbitrary BYO OpenRouter model IDs and only the user's provider key to their endpoint", async () => {
    const { url, requests } = await inferenceServer();
    writeConfig({ api_key: token });
    mkdirSync(join(root, "agent"));
    writeFileSync(getAgentModelsPath(), JSON.stringify({ providers: {
      openrouter: { baseUrl: `${url}/openrouter/v1` },
    } }));
    const runtime = await createPiModelRuntime();
    await runtime.setRuntimeApiKey("openrouter", "byo-openrouter-key");
    const { selection } = setAgentModel(runtime, {}, "openrouter", "vendor/model-newer-than-the-catalog", { thinking: "off" });
    expect(getAgentSelection({})).toEqual(selection);
    const { model } = resolveAgentModel(runtime, selection);
    const result = await runtime.completeSimple(model as never, {
      messages: [{ role: "user", content: "Hi", timestamp: 1 }],
    });
    expect(result.stopReason).toBe("stop");
    expect(requests).toHaveLength(1);
    expect(requests[0]).toMatchObject({
      path: "/openrouter/v1/chat/completions",
      authorization: "Bearer byo-openrouter-key",
      body: { model: "vendor/model-newer-than-the-catalog", stream: true },
    });
    expect(JSON.stringify(requests)).not.toContain(token);
    const restored = await createPiModelRuntime();
    await restored.setRuntimeApiKey("openrouter", "byo-openrouter-key");
    expect(resolveAgentModel(restored, getAgentSelection({})!).model.id).toBe(selection.model);
  });

  it("replaces a saved BYO selection on TT login and routes through TT after restart", async () => {
    const { url, requests } = await inferenceServer();
    writeConfig({
      base_url: url,
      agent: { provider: "openrouter", model: "~deepseek/deepseek-v4-flash-latest", thinking: "high" },
    });
    const runtime = await createPiModelRuntime();
    await loginAgentProvider(runtime, "tunedtensor", token);
    const managed = { provider: "tunedtensor", model: "managed", thinking: "off" };
    expect(readConfig().agent).toEqual(managed);
    expect(getAgentSelection({})).toEqual(managed);
    expect(resolveAgentModel(runtime, getAgentSelection({})!).model.provider).toBe("tunedtensor");

    const restored = await createPiModelRuntime();
    const { model } = resolveAgentModel(restored, getAgentSelection({})!);
    const result = await restored.completeSimple(model as never, {
      messages: [{ role: "user", content: "Hi", timestamp: 1 }],
    });
    expect(result.stopReason).toBe("stop");
    expect(requests).toHaveLength(1);
    expect(requests[0]).toMatchObject({
      path: "/api/v1/agent/chat/completions",
      authorization: `Bearer ${token}`,
      body: { model: "managed" },
    });
  });
});
