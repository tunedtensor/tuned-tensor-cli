import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Command } from "commander";
import { inspectCheckpoint, readInferencePayload, registerEndpointCommands, uploadWeights } from "../../commands/endpoints.js";
import { setJsonMode } from "../../output.js";
import * as client from "../../client.js";
vi.mock("../../client.js", () => ({ get: vi.fn(), post: vi.fn(), del: vi.fn() }));
let dir: string;
const id = "11111111-1111-4111-8111-111111111111";
const requestId = "22222222-2222-4222-8222-222222222222";
const endpoint = { id, name: "test", status: "deploying", price_cents: 1, invocation_path: `/api/v1/endpoints/${id}/invoke` };
const question = { type: "choice", instructions: "Route the input", criteria: { yes: "yes", no: "no" } };
function model() {
  writeFileSync(join(dir, "rl_agent_config.json"), JSON.stringify({ encoder: "answerdotai/ModernBERT-large", head_layers: 2, max_len: 512, head_max_len: 192, tuned_tensor: {}, temperature: [1, 1, 1] }));
  const header = Buffer.from(JSON.stringify(Object.fromEntries(["encoder.x", "type_emb.x", "scorer.x", "act_head.x"].map(key => [key, {}]))));
  const length = Buffer.alloc(8); length.writeBigUInt64LE(BigInt(header.length));
  writeFileSync(join(dir, "model.safetensors"), Buffer.concat([length, header, Buffer.alloc(16)]));
}
function program() {
  const p = new Command().option("--api-key <token>").option("--base-url <url>");
  registerEndpointCommands(p); p.exitOverride(); return p;
}
beforeEach(() => {
  vi.clearAllMocks(); dir = mkdtempSync(join(tmpdir(), "tt-endpoint-test-"));
  vi.spyOn(console, "log").mockImplementation(() => {});
  vi.spyOn(console, "error").mockImplementation(() => {});
  setJsonMode(true);
});
afterEach(() => { rmSync(dir, { recursive: true, force: true }); vi.restoreAllMocks(); vi.unstubAllGlobals(); });
describe("checkpoint hosting", () => {
  it("streams a complete safetensors checkpoint and identifies TT fine-tunes", async () => {
    model();
    const result = await inspectCheckpoint(dir);
    expect(result.checkpoint_kind).toBe("tuned");
    expect(result.weights_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(result.weights_bytes).toBeGreaterThan(8);
  });
  it("rejects unsupported models and malformed tensor headers locally", async () => {
    model(); writeFileSync(join(dir, "model.safetensors"), Buffer.alloc(9, 255));
    await expect(inspectCheckpoint(dir)).rejects.toThrow("header");
    writeFileSync(join(dir, "rl_agent_config.json"), JSON.stringify({ encoder: "gpu-model" }));
    await expect(inspectCheckpoint(dir)).rejects.toThrow("only the pinned English Laya");
    expect(client.post).not.toHaveBeenCalled();
  });
  it("reads a complete model payload without a hosting-specific question schema", () => {
    const file = join(dir, "request.json");
    const payload = { input: "hello", question, options: { temperature: 0 } };
    writeFileSync(file, JSON.stringify(payload)); expect(readInferencePayload(file)).toEqual(payload);
    for (const invalid of [[], null, "text", {}]) {
      writeFileSync(file, JSON.stringify(invalid)); expect(() => readInferencePayload(file)).toThrow();
    }
    writeFileSync(file, JSON.stringify({ input: "x".repeat(131072) }));
    expect(() => readInferencePayload(file)).toThrow("128 KiB");
  });
  it("uploads without forwarding the account token and then starts deployment", async () => {
    model();
    const upload = { url: "https://test.s3.eu-west-1.amazonaws.com/weights", headers: { "content-type": "application/octet-stream" }, expires_in: 900 };
    vi.mocked(client.post).mockResolvedValueOnce({ data: { ...endpoint, inference_key: "tte_secret", upload } }).mockResolvedValueOnce({ data: endpoint });
    const fetch = vi.fn(async () => new Response(null, { status: 200 })); vi.stubGlobal("fetch", fetch);
    await program().parseAsync(["node", "tt", "--api-key", "tt_secret", "endpoints", "create", "--model", dir, "--name", "test"]);
    expect(vi.mocked(client.post).mock.calls[0][1]).not.toHaveProperty("question");
    expect(fetch).toHaveBeenCalledWith(expect.any(URL), expect.objectContaining({ method: "PUT", headers: upload.headers, redirect: "error" }));
    expect(client.post).toHaveBeenLastCalledWith(`/endpoints/${id}/deploy`, {}, expect.objectContaining({ apiKey: "tt_secret" }));
    expect(console.log).toHaveBeenCalledWith(expect.stringContaining("tte_secret"));
  });
  it("rejects insecure upload locations before sending weights", async () => {
    await expect(uploadWeights({ ...endpoint, upload: { url: "http://evil.invalid", headers: {}, expires_in: 900 } }, "unused")).rejects.toThrow("invalid model upload URL");
  });
  it("preserves a user-provided request ID and reports it after transport failure", async () => {
    const file = join(dir, "request.json"); const payload = { input: "hello", question };
    writeFileSync(file, JSON.stringify(payload));
    vi.mocked(client.post).mockRejectedValue(new Error("connection lost"));
    await expect(program().parseAsync(["node", "tt", "endpoints", "invoke", id, "--payload", file, "--request-id", requestId])).rejects.toThrow(`--request-id ${requestId}`);
    expect(client.post).toHaveBeenCalledWith(`/endpoints/${id}/invoke`, payload, expect.anything(), { "Idempotency-Key": requestId });
  });
});
