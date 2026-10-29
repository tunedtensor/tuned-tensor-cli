import { Command } from "commander";
import { createHash, randomUUID } from "node:crypto";
import { createReadStream, readFileSync, statSync, openSync, readSync, closeSync } from "node:fs";
import { open } from "node:fs/promises";
import { finished } from "node:stream/promises";
import { resolve, join } from "node:path";
import { z } from "zod";
import { get, post, del, type ClientOpts } from "../client.js";
import { getBaseUrl } from "../config.js";
import { isJsonMode, printJson, printDetail } from "../output.js";

const description = z.string().min(1).max(300);
const questionSchema = z.discriminatedUnion("type", [
  z.object({ type: z.literal("choice"), instructions: z.string().max(2000),
    criteria: z.record(z.string().min(1).max(100), description).refine(v => Object.keys(v).length >= 2 && Object.keys(v).length <= 32) }).strict(),
  z.object({ type: z.literal("score"), instructions: z.string().max(2000), criteria: z.array(description).min(2).max(10) }).strict(),
  z.object({ type: z.literal("noul"), instructions: z.string().max(2000),
    criteria: z.object({ true: description, false: description }).strict().optional() }).strict(),
]);

interface Endpoint {
  id: string; name: string; status: string; price_cents: number; invocation_path: string;
  inference_key?: string;
}
interface UploadEndpoint extends Endpoint { upload: { url: string; headers: Record<string, string>; expires_in: number } }

/** Validate locally before allocating cloud resources; stream the ~1.7 GB weights. */
export async function inspectCheckpoint(modelPath: string) {
  const directory = resolve(modelPath);
  const config = JSON.parse(readFileSync(join(directory, "rl_agent_config.json"), "utf8"));
  if (config.encoder !== "answerdotai/ModernBERT-large" || config.head_layers !== 2 || config.max_len !== 512 || config.head_max_len !== 192) {
    throw new Error("Hosting currently supports only the pinned English Laya checkpoint and TT decision fine-tunes.");
  }
  const kind = config.tuned_tensor ? "tuned" : "base";
  if (kind === "tuned" && (JSON.stringify(config.temperature) !== "[1,1,1]" || config.temperature_by_options)) {
    throw new Error("Custom calibration is not supported by hosted decision inference.");
  }
  const file = join(directory, "model.safetensors");
  const size = statSync(file).size;
  if (size <= 8 || size > 2_000_000_000) throw new Error("Weights must be a safetensors file smaller than 2 GB. GPU and adapter models are not supported.");
  const fd = openSync(file, "r");
  try {
    const length = Buffer.alloc(8);
    if (readSync(fd, length, 0, 8, 0) !== 8) throw new Error("Invalid safetensors header.");
    const headerSize = Number(length.readBigUInt64LE());
    if (!Number.isSafeInteger(headerSize) || headerSize < 2 || headerSize > 1_000_000 || headerSize + 8 >= size) throw new Error("Invalid safetensors header.");
    const header = Buffer.alloc(headerSize);
    if (readSync(fd, header, 0, headerSize, 8) !== headerSize) throw new Error("Truncated safetensors header.");
    const tensors = JSON.parse(header.toString("utf8"));
    for (const prefix of ["encoder.", "type_emb.", "scorer.", "act_head."]) {
      if (!Object.keys(tensors).some(key => key.startsWith(prefix))) throw new Error("Not a complete Laya checkpoint.");
    }
  } finally { closeSync(fd); }
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return { file, weights_bytes: size, weights_sha256: hash.digest("hex"), checkpoint_kind: kind as "base" | "tuned" };
}

export function readQuestion(file: string) {
  const data = JSON.parse(readFileSync(resolve(file), "utf8"));
  // A decision run's resolved-workflow.json records the exact training question.
  return questionSchema.parse(data.question ?? data);
}

export async function uploadWeights(endpoint: UploadEndpoint, file: string) {
  const url = new URL(endpoint.upload.url);
  if (url.protocol !== "https:" || url.username || url.password ||
      !/\.s3\.[a-z0-9-]+\.amazonaws\.com$/.test(url.hostname)) {
    throw new Error("The server returned an invalid model upload URL.");
  }
  const handle = await open(file, "r");
  const stream = handle.createReadStream();
  const completion = finished(stream);
  void completion.catch(() => {});
  try {
    const response = await fetch(url, { method: "PUT", headers: endpoint.upload.headers,
      body: stream, duplex: "half", redirect: "error",
      signal: AbortSignal.timeout(60 * 60_000) } as unknown as Parameters<typeof fetch>[1]);
    if (!response.ok) throw new Error(`Model upload failed (${response.status}). Delete endpoint ${endpoint.id} and retry.`);
  } finally { stream.destroy(); await completion.catch(() => {}); }
}

function show(endpoint: Endpoint, opts: ClientOpts) {
  const result = { ...endpoint, invocation_url: new URL(endpoint.invocation_path, getBaseUrl(opts)).toString() };
  if (isJsonMode()) return printJson(result);
  printDetail([["Endpoint", result.id], ["Name", result.name], ["Status", result.status],
    ["Inference URL", result.invocation_url], ["Price per successful request", `${result.price_cents} cents`],
    ...(result.inference_key ? [["Inference key (save securely)", result.inference_key] as [string, string]] : [])]);
}

export function registerEndpointCommands(parent: Command) {
  const endpoints = parent.command("endpoints").description("Host small CPU decision models using prepaid TT credits");
  endpoints.command("create").description("Upload Laya weights and deploy a serverless endpoint")
    .requiredOption("--model <directory>", "Local Laya checkpoint (e.g. a decision run's train/model)")
    .requiredOption("--question <file>", "Question JSON or the run's resolved-workflow.json")
    .requiredOption("--name <name>", "Endpoint display name")
    .action(async (options: { model: string; question: string; name: string }) => {
      const opts = parent.optsWithGlobals() as ClientOpts;
      const question = readQuestion(options.question);
      const { file, ...artifact } = await inspectCheckpoint(options.model);
      const { data } = await post<UploadEndpoint>("/endpoints", { name: options.name, format: "laya-safetensors-v1", question, ...artifact }, opts);
      if (!isJsonMode()) console.error(`Endpoint ${data.id} created. Uploading ${(artifact.weights_bytes / 1_000_000_000).toFixed(2)} GB of weights...`);
      try {
        await uploadWeights(data, file);
        const { data: deployed } = await post<Endpoint>(`/endpoints/${data.id}/deploy`, {}, opts);
        show({ ...deployed, inference_key: data.inference_key }, opts);
      } catch (error) {
        // Preserve the original error and ID; cleanup is durable even if the CLI exits.
        throw new Error(`Endpoint ${data.id}: ${error instanceof Error ? error.message : "upload/deployment failed"}. Use tt cloud endpoints get ${data.id} or delete ${data.id}. Rotate its key if the create response was lost.`, { cause: error });
      }
    });
  endpoints.command("list").action(async () => {
    const opts = parent.optsWithGlobals() as ClientOpts;
    const { data } = await get<Endpoint[]>("/endpoints", undefined, opts);
    if (isJsonMode()) return printJson(data);
    for (const row of data) show(row, opts);
  });
  endpoints.command("get <id>").description("Check deployment or deletion status").action(async (id: string) => {
    z.uuid().parse(id); const opts = parent.optsWithGlobals() as ClientOpts;
    show((await get<Endpoint>(`/endpoints/${id}`, undefined, opts)).data, opts);
  });
  endpoints.command("delete <id>").description("Stop serving and delete cloud weights and resources").action(async (id: string) => {
    z.uuid().parse(id); const opts = parent.optsWithGlobals() as ClientOpts;
    show((await del<Endpoint>(`/endpoints/${id}`, opts)).data, opts);
  });
  endpoints.command("rotate-key <id>").description("Revoke the old inference key and return a new key").action(async (id: string) => {
    z.uuid().parse(id);
    const { data } = await post<{ inference_key: string }>(`/endpoints/${id}/key`, {}, parent.optsWithGlobals() as ClientOpts);
    if (isJsonMode()) return printJson(data);
    printDetail([["Inference key (save securely)", data.inference_key]]);
  });
  endpoints.command("invoke <id>").requiredOption("--input <text>", "Decision input (up to 8192 characters)")
    .option("--request-id <uuid>", "Reuse this ID to retry without a second charge")
    .action(async (id: string, options: { input: string; requestId?: string }) => {
      z.uuid().parse(id);
      const requestId = options.requestId ?? randomUUID();
      z.uuid().parse(requestId);
      if (!isJsonMode()) console.error(`Request ID: ${requestId} (reuse with --request-id when retrying)`);
      try {
        const { data } = await post(`/endpoints/${id}/invoke`, { input: options.input }, parent.optsWithGlobals() as ClientOpts, { "Idempotency-Key": requestId });
        printJson({ request_id: requestId, data });
      } catch (error) {
        throw new Error(`Request ${requestId}: ${error instanceof Error ? error.message : "inference failed"}. Retry with --request-id ${requestId}.`, { cause: error });
      }
    });
}
