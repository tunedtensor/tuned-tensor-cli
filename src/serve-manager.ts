import { spawn, type ChildProcess } from "node:child_process";
import { randomUUID } from "node:crypto";
import { appendFileSync, mkdirSync } from "node:fs";
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { createServer as createNetServer } from "node:net";
import { dirname } from "node:path";

type SpawnModelServer = (
  command: string,
  args: string[],
  options: { env: NodeJS.ProcessEnv; stdio: "inherit" },
) => ChildProcess;

export interface ManagedServeOptions {
  publicHost: string;
  publicPort: number;
  python: string;
  args: string[];
  env: Record<string, string>;
  modelName: string;
  modelPath: string;
  idleTimeoutSeconds: number;
  restartAfterRequests: number;
  gateField: string;
  logFile?: string;
  spawnModelServer?: SpawnModelServer;
  logRecord?: (record: ManagedServeLogRecord) => void;
}

export interface ManagedServeLogRecord {
  timestamp: string;
  request_id: string;
  route: string;
  model_target: { name: string; path: string };
  request_bytes: number;
  response_status: number;
  latency_ms: number;
  queued_ms: number;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
  schema_validity: boolean | null;
  gate_field: string;
  gate_result: unknown;
  restart_count: number;
  error_summary: string | null;
}

interface ForwardResult {
  status: number;
  headers: Headers;
  body: string;
  parsed: unknown;
}

const COMPLETION_ROUTES = new Set(["/v1/chat/completions", "/chat/completions"]);

function parsePositiveInteger(value: number, fallback: number): number {
  return Number.isInteger(value) && value >= 0 ? value : fallback;
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function readRequestBody(req: IncomingMessage): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

function sendJson(res: ServerResponse, status: number, payload: unknown) {
  const body = Buffer.from(JSON.stringify(payload), "utf8");
  res.writeHead(status, {
    "content-type": "application/json",
    "content-length": String(body.byteLength),
    "access-control-allow-origin": "*",
    "access-control-allow-headers": "authorization, content-type",
    "access-control-allow-methods": "GET, POST, OPTIONS",
  });
  res.end(body);
}

function copyResponseHeaders(headers: Headers): Record<string, string> {
  const copied: Record<string, string> = {
    "access-control-allow-origin": "*",
    "access-control-allow-headers": "authorization, content-type",
    "access-control-allow-methods": "GET, POST, OPTIONS",
  };
  const contentType = headers.get("content-type");
  if (contentType) copied["content-type"] = contentType;
  return copied;
}

function tryJsonParse(value: string): unknown {
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

function readPath(value: unknown, path: string): unknown {
  if (!path) return null;
  let current = value;
  for (const part of path.split(".")) {
    if (!part) return null;
    if (!current || typeof current !== "object" || !(part in current)) return null;
    current = (current as Record<string, unknown>)[part];
  }
  return current ?? null;
}

function extractAssistantContent(response: unknown): string | null {
  if (!response || typeof response !== "object") return null;
  const choices = (response as { choices?: unknown }).choices;
  if (!Array.isArray(choices) || choices.length === 0) return null;
  const first = choices[0];
  if (!first || typeof first !== "object") return null;
  const message = (first as { message?: unknown }).message;
  if (!message || typeof message !== "object") return null;
  const content = (message as { content?: unknown }).content;
  return typeof content === "string" ? content : null;
}

function usageNumber(response: unknown, key: string): number | null {
  if (!response || typeof response !== "object") return null;
  const usage = (response as { usage?: unknown }).usage;
  if (!usage || typeof usage !== "object") return null;
  const value = (usage as Record<string, unknown>)[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function errorSummary(response: unknown, fallback: string | null): string | null {
  if (!response || typeof response !== "object") return fallback;
  const error = (response as { error?: unknown }).error;
  if (!error || typeof error !== "object") return fallback;
  const message = (error as { message?: unknown }).message;
  return typeof message === "string" ? message : fallback;
}

async function allocatePort(host: string): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = createNetServer();
    server.on("error", reject);
    server.listen(0, host, () => {
      const address = server.address();
      if (!address || typeof address === "string") {
        server.close(() => reject(new Error("Could not allocate an internal port")));
        return;
      }
      const port = address.port;
      server.close(() => resolve(port));
    });
  });
}

export class ManagedModelServer {
  private readonly spawnModelServer: SpawnModelServer;
  private readonly idleTimeoutMs: number;
  private readonly restartAfterRequests: number;
  private readonly logRecordOverride?: (record: ManagedServeLogRecord) => void;
  private server: Server | null = null;
  private child: ChildProcess | null = null;
  private internalPort: number | null = null;
  private queue: Promise<void> = Promise.resolve();
  private idleTimer: NodeJS.Timeout | null = null;
  private activeRequests = 0;
  private queuedRequests = 0;
  private requestsSinceStart = 0;
  private restartCount = 0;
  private stoppingChild = false;

  constructor(private readonly options: ManagedServeOptions) {
    this.spawnModelServer = options.spawnModelServer ?? spawn;
    this.idleTimeoutMs = parsePositiveInteger(options.idleTimeoutSeconds, 300) * 1000;
    this.restartAfterRequests = parsePositiveInteger(options.restartAfterRequests, 100);
    this.logRecordOverride = options.logRecord;
  }

  get isModelRunning(): boolean {
    return this.child != null;
  }

  get restarts(): number {
    return this.restartCount;
  }

  async start(): Promise<void> {
    if (this.server) return;
    this.internalPort = await allocatePort("127.0.0.1");
    this.server = createServer((req, res) => {
      this.handle(req, res).catch((err) => {
        sendJson(res, 500, { error: { message: err instanceof Error ? err.message : String(err) } });
      });
    });

    await new Promise<void>((resolve, reject) => {
      this.server?.once("error", reject);
      this.server?.listen(this.options.publicPort, this.options.publicHost, () => {
        this.server?.off("error", reject);
        resolve();
      });
    });
  }

  async stop(): Promise<void> {
    this.clearIdleTimer();
    await this.stopChild();
    if (!this.server) return;
    const server = this.server;
    this.server = null;
    await new Promise<void>((resolve, reject) => {
      server.close((err) => (err ? reject(err) : resolve()));
    });
  }

  async run(): Promise<void> {
    await this.start();
    await new Promise<void>((resolve, reject) => {
      this.server?.once("error", reject);
      this.server?.once("close", resolve);
    });
  }

  private async handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const route = new URL(req.url || "/", "http://localhost").pathname;
    if (req.method === "OPTIONS") {
      sendJson(res, 204, {});
      return;
    }

    if (req.method === "GET" && route === "/health") {
      await this.handleHealth(res);
      return;
    }

    if (req.method === "POST" && COMPLETION_ROUTES.has(route)) {
      const receivedAt = Date.now();
      const body = await readRequestBody(req);
      this.queuedRequests += 1;
      const work = this.queue.then(() => this.processGeneration(route, body, res, receivedAt));
      this.queue = work.catch(() => undefined);
      try {
        await work;
      } finally {
        this.queuedRequests -= 1;
        this.scheduleIdleStop();
      }
      return;
    }

    sendJson(res, 404, { error: { message: "Not found" } });
  }

  private async handleHealth(res: ServerResponse): Promise<void> {
    if (!this.child) {
      sendJson(res, 200, {
        status: "ok",
        wrapper: "ok",
        model_status: "cold",
        restart_count: this.restartCount,
      });
      return;
    }

    if (this.activeRequests > 0 || this.queuedRequests > 0) {
      sendJson(res, 200, {
        status: "ok",
        wrapper: "ok",
        model_status: "busy",
        restart_count: this.restartCount,
      });
      return;
    }

    const health = await this.checkHealth();
    if (health.ok) {
      sendJson(res, 200, {
        status: "ok",
        wrapper: "ok",
        model_status: "warm",
        restart_count: this.restartCount,
        model: health.body,
      });
      return;
    }

    await this.restartChild();
    sendJson(res, 200, {
      status: "ok",
      wrapper: "ok",
      model_status: "restarted",
      restart_count: this.restartCount,
    });
  }

  private async processGeneration(
    route: string,
    body: Buffer,
    res: ServerResponse,
    receivedAt: number,
  ): Promise<void> {
    const requestId = `ttreq-${randomUUID()}`;
    const startedAt = Date.now();
    const queuedMs = startedAt - receivedAt;
    this.activeRequests += 1;
    this.clearIdleTimer();

    let status = 500;
    let parsed: unknown = null;
    let failure: string | null = null;

    try {
      await this.ensureModelReady();
      const forwarded = await this.forwardGeneration(route, body);
      status = forwarded.status;
      parsed = forwarded.parsed;
      res.writeHead(status, {
        ...copyResponseHeaders(forwarded.headers),
        "content-length": String(Buffer.byteLength(forwarded.body)),
      });
      res.end(forwarded.body);
      this.requestsSinceStart += 1;
      failure = errorSummary(parsed, null);
    } catch (err) {
      failure = err instanceof Error ? err.message : String(err);
      status = 502;
      sendJson(res, status, { error: { message: failure } });
    } finally {
      const finishedAt = Date.now();
      this.activeRequests -= 1;
      this.writeLog({
        timestamp: new Date(finishedAt).toISOString(),
        request_id: requestId,
        route,
        model_target: {
          name: this.options.modelName,
          path: this.options.modelPath,
        },
        request_bytes: body.byteLength,
        response_status: status,
        latency_ms: finishedAt - receivedAt,
        queued_ms: queuedMs,
        prompt_tokens: usageNumber(parsed, "prompt_tokens"),
        completion_tokens: usageNumber(parsed, "completion_tokens"),
        total_tokens: usageNumber(parsed, "total_tokens"),
        schema_validity: status < 400 ? true : status === 422 ? false : null,
        gate_field: this.options.gateField,
        gate_result: this.extractGateResult(parsed),
        restart_count: this.restartCount,
        error_summary: failure,
      });
    }
  }

  private async ensureModelReady(): Promise<void> {
    if (this.child && this.restartAfterRequests > 0 && this.requestsSinceStart >= this.restartAfterRequests) {
      await this.restartChild();
      return;
    }

    if (this.child) {
      const health = await this.checkHealth();
      if (health.ok) return;
      await this.restartChild();
      return;
    }

    await this.startChild();
  }

  private async startChild(): Promise<void> {
    if (this.internalPort == null) {
      this.internalPort = await allocatePort("127.0.0.1");
    }

    const env = {
      ...process.env,
      ...this.options.env,
      TT_HOST: "127.0.0.1",
      TT_PORT: String(this.internalPort),
    };
    this.child = this.spawnModelServer(this.options.python, this.options.args, {
      stdio: "inherit",
      env,
    });
    this.child.once("exit", () => {
      if (!this.stoppingChild) {
        this.child = null;
        this.requestsSinceStart = 0;
      }
    });
    try {
      await this.waitForHealth();
      this.scheduleIdleStop();
    } catch (err) {
      await this.stopChild();
      throw err;
    }
  }

  private async restartChild(): Promise<void> {
    await this.stopChild();
    this.restartCount += 1;
    this.requestsSinceStart = 0;
    await this.startChild();
  }

  private async stopChild(): Promise<void> {
    if (!this.child) return;
    const child = this.child;
    this.child = null;
    this.requestsSinceStart = 0;
    this.stoppingChild = true;
    try {
      child.kill("SIGTERM");
      await Promise.race([
        new Promise<void>((resolve) => child.once("exit", () => resolve())),
        sleep(5000).then(() => {
          if (child.exitCode == null && child.signalCode == null) child.kill("SIGKILL");
        }),
      ]);
    } finally {
      this.stoppingChild = false;
    }
  }

  private async checkHealth(): Promise<{ ok: boolean; body: unknown }> {
    if (this.internalPort == null) return { ok: false, body: null };
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 2000);
    try {
      const response = await fetch(`http://127.0.0.1:${this.internalPort}/health`, {
        signal: controller.signal,
      });
      const text = await response.text();
      return { ok: response.ok, body: tryJsonParse(text) };
    } catch {
      return { ok: false, body: null };
    } finally {
      clearTimeout(timeout);
    }
  }

  private async waitForHealth(): Promise<void> {
    const deadline = Date.now() + 120_000;
    while (Date.now() < deadline) {
      const health = await this.checkHealth();
      if (health.ok) return;
      if (!this.child) break;
      await sleep(50);
    }
    throw new Error("Model server did not become healthy within 120 seconds");
  }

  private async forwardGeneration(route: string, body: Buffer): Promise<ForwardResult> {
    if (this.internalPort == null) throw new Error("Internal server port is not allocated");
    const response = await fetch(`http://127.0.0.1:${this.internalPort}${route}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body,
    });
    const text = await response.text();
    return {
      status: response.status,
      headers: response.headers,
      body: text,
      parsed: tryJsonParse(text),
    };
  }

  private extractGateResult(response: unknown): unknown {
    const content = extractAssistantContent(response);
    if (!content) return null;
    const parsedContent = tryJsonParse(content);
    return readPath(parsedContent, this.options.gateField);
  }

  private scheduleIdleStop() {
    this.clearIdleTimer();
    if (!this.child || this.activeRequests > 0 || this.queuedRequests > 0) return;
    this.idleTimer = setTimeout(() => {
      this.stopChild().catch((err) => {
        this.writeDiagnosticError(err);
      });
    }, this.idleTimeoutMs);
  }

  private clearIdleTimer() {
    if (!this.idleTimer) return;
    clearTimeout(this.idleTimer);
    this.idleTimer = null;
  }

  private writeLog(record: ManagedServeLogRecord) {
    if (this.logRecordOverride) {
      this.logRecordOverride(record);
      return;
    }
    const line = `${JSON.stringify(record)}\n`;
    if (this.options.logFile) {
      mkdirSync(dirname(this.options.logFile), { recursive: true });
      appendFileSync(this.options.logFile, line, "utf8");
      return;
    }
    process.stderr.write(line);
  }

  private writeDiagnosticError(err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    process.stderr.write(`${JSON.stringify({ timestamp: new Date().toISOString(), error: message })}\n`);
  }
}

export async function startManagedModelServer(options: ManagedServeOptions): Promise<void> {
  const server = new ManagedModelServer(options);
  const stop = () => {
    server.stop().catch(() => undefined);
  };
  process.once("SIGINT", stop);
  process.once("SIGTERM", stop);
  try {
    await server.run();
  } finally {
    process.off("SIGINT", stop);
    process.off("SIGTERM", stop);
    await server.stop();
  }
}
