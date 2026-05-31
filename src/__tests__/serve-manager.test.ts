import { describe, it, expect, vi, afterEach } from "vitest";
import { EventEmitter } from "node:events";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { createServer as createNetServer } from "node:net";
import type { ChildProcess } from "node:child_process";
import { ManagedModelServer, type ManagedServeLogRecord } from "../serve-manager.js";

async function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = createNetServer();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (!address || typeof address === "string") {
        server.close(() => reject(new Error("No port allocated")));
        return;
      }
      server.close(() => resolve(address.port));
    });
  });
}

function jsonResponse(res: ServerResponse, status: number, payload: unknown) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    "content-type": "application/json",
    "content-length": String(Buffer.byteLength(body)),
  });
  res.end(body);
}

async function readBody(req: IncomingMessage): Promise<Buffer> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(Buffer.from(chunk));
  }
  return Buffer.concat(chunks);
}

interface FakeSpawnOptions {
  onCompletion?: (body: Buffer, spawnIndex: number) => Promise<{ status: number; payload: unknown }>;
  healthOk?: (spawnIndex: number) => boolean;
}

function createFakeSpawn(options: FakeSpawnOptions = {}) {
  const servers: ReturnType<typeof createServer>[] = [];
  const spawn = vi.fn((_command, _args, spawnOpts: { env: NodeJS.ProcessEnv }) => {
    const spawnIndex = servers.length + 1;
    const server = createServer(async (req, res) => {
      const route = new URL(req.url || "/", "http://localhost").pathname;
      if (req.method === "GET" && route === "/health") {
        if (options.healthOk && !options.healthOk(spawnIndex)) {
          jsonResponse(res, 500, { status: "failed" });
          return;
        }
        jsonResponse(res, 200, { status: "ok", spawn_index: spawnIndex });
        return;
      }
      if (req.method === "POST" && route === "/v1/chat/completions") {
        const body = await readBody(req);
        const completion = options.onCompletion
          ? await options.onCompletion(body, spawnIndex)
          : {
              status: 200,
              payload: {
                choices: [{ message: { content: "{\"should_process\":true}" } }],
                usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
              },
            };
        jsonResponse(res, completion.status, completion.payload);
        return;
      }
      jsonResponse(res, 404, { error: { message: "Not found" } });
    });
    servers.push(server);
    server.listen(Number(spawnOpts.env.TT_PORT), spawnOpts.env.TT_HOST);

    const child = new EventEmitter() as ChildProcess;
    child.kill = vi.fn(() => {
      server.close(() => child.emit("exit", 0, null));
      return true;
    });
    Object.defineProperty(child, "exitCode", { get: () => null });
    Object.defineProperty(child, "signalCode", { get: () => null });
    return child;
  });

  return { spawn, servers };
}

async function createManagedServer(
  overrides: Partial<ConstructorParameters<typeof ManagedModelServer>[0]> = {},
) {
  const port = await freePort();
  const logs: ManagedServeLogRecord[] = [];
  const fake = createFakeSpawn();
  const server = new ManagedModelServer({
    publicHost: "127.0.0.1",
    publicPort: port,
    python: "python3",
    args: ["server.py"],
    env: {},
    modelName: "test-model",
    modelPath: "/models/test-model",
    idleTimeoutSeconds: 300,
    restartAfterRequests: 100,
    gateField: "should_process",
    spawnModelServer: fake.spawn,
    logRecord: (record) => logs.push(record),
    ...overrides,
  });
  await server.start();
  return { server, port, logs, fake };
}

afterEach(() => {
  vi.useRealTimers();
});

describe("ManagedModelServer", () => {
  it("starts the model on demand and logs request metrics", async () => {
    const { server, port, logs, fake } = await createManagedServer();
    try {
      expect(fake.spawn).not.toHaveBeenCalled();

      const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
        method: "POST",
        body: JSON.stringify({ messages: [{ role: "user", content: "hi" }] }),
      });

      expect(response.status).toBe(200);
      expect(fake.spawn).toHaveBeenCalledTimes(1);
      expect(logs).toHaveLength(1);
      expect(logs[0]).toMatchObject({
        route: "/v1/chat/completions",
        response_status: 200,
        prompt_tokens: 3,
        completion_tokens: 2,
        total_tokens: 5,
        schema_validity: true,
        gate_field: "should_process",
        gate_result: true,
        restart_count: 0,
      });
      expect(logs[0].request_bytes).toBeGreaterThan(0);
    } finally {
      await server.stop();
    }
  });

  it("serializes generation requests through one internal request at a time", async () => {
    let active = 0;
    let maxActive = 0;
    let releaseFirst!: () => void;
    const firstBlock = new Promise<void>((resolve) => {
      releaseFirst = resolve;
    });
    const fake = createFakeSpawn({
      async onCompletion() {
        active += 1;
        maxActive = Math.max(maxActive, active);
        if (maxActive === 1) await firstBlock;
        active -= 1;
        return {
          status: 200,
          payload: { choices: [{ message: { content: "{\"should_process\":true}" } }] },
        };
      },
    });
    const port = await freePort();
    const server = new ManagedModelServer({
      publicHost: "127.0.0.1",
      publicPort: port,
      python: "python3",
      args: ["server.py"],
      env: {},
      modelName: "test-model",
      modelPath: "/models/test-model",
      idleTimeoutSeconds: 300,
      restartAfterRequests: 100,
      gateField: "should_process",
      spawnModelServer: fake.spawn,
      logRecord: () => undefined,
    });
    await server.start();
    try {
      const first = fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
        method: "POST",
        body: "{}",
      });
      const second = fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
        method: "POST",
        body: "{}",
      });

      await new Promise((resolve) => setTimeout(resolve, 50));
      expect(maxActive).toBe(1);
      releaseFirst();
      await Promise.all([first, second]);
      expect(maxActive).toBe(1);
    } finally {
      await server.stop();
    }
  });

  it("restarts before the next request after the configured request threshold", async () => {
    const { server, port, fake } = await createManagedServer({ restartAfterRequests: 1 });
    try {
      await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, { method: "POST", body: "{}" });
      await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, { method: "POST", body: "{}" });

      expect(fake.spawn).toHaveBeenCalledTimes(2);
      expect(server.restarts).toBe(1);
    } finally {
      await server.stop();
    }
  });

  it("restarts when an internal health check fails", async () => {
    let firstSpawnHealthy = true;
    const fake = createFakeSpawn({
      healthOk(spawnIndex) {
        return spawnIndex === 1 ? firstSpawnHealthy : true;
      },
    });
    const port = await freePort();
    const server = new ManagedModelServer({
      publicHost: "127.0.0.1",
      publicPort: port,
      python: "python3",
      args: ["server.py"],
      env: {},
      modelName: "test-model",
      modelPath: "/models/test-model",
      idleTimeoutSeconds: 300,
      restartAfterRequests: 100,
      gateField: "should_process",
      spawnModelServer: fake.spawn,
      logRecord: () => undefined,
    });
    await server.start();
    try {
      await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, { method: "POST", body: "{}" });
      firstSpawnHealthy = false;
      await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, { method: "POST", body: "{}" });

      expect(fake.spawn).toHaveBeenCalledTimes(2);
      expect(server.restarts).toBe(1);
    } finally {
      await server.stop();
    }
  });

  it("stops the model after the idle timeout", async () => {
    const { server, port, fake } = await createManagedServer({ idleTimeoutSeconds: 0 });
    try {
      await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, { method: "POST", body: "{}" });
      await new Promise((resolve) => setTimeout(resolve, 50));

      expect(server.isModelRunning).toBe(false);
      const child = fake.spawn.mock.results[0].value as ChildProcess;
      expect(child.kill).toHaveBeenCalledWith("SIGTERM");
    } finally {
      await server.stop();
    }
  });

  it("logs schema validity failures and missing gate values", async () => {
    const fake = createFakeSpawn({
      async onCompletion() {
        return {
          status: 422,
          payload: { error: { message: "Model did not produce valid JSON" } },
        };
      },
    });
    const port = await freePort();
    const logs: ManagedServeLogRecord[] = [];
    const server = new ManagedModelServer({
      publicHost: "127.0.0.1",
      publicPort: port,
      python: "python3",
      args: ["server.py"],
      env: {},
      modelName: "test-model",
      modelPath: "/models/test-model",
      idleTimeoutSeconds: 300,
      restartAfterRequests: 100,
      gateField: "should_process",
      spawnModelServer: fake.spawn,
      logRecord: (record) => logs.push(record),
    });
    await server.start();
    try {
      const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
        method: "POST",
        body: "{}",
      });

      expect(response.status).toBe(422);
      expect(logs[0]).toMatchObject({
        response_status: 422,
        schema_validity: false,
        gate_result: null,
        error_summary: "Model did not produce valid JSON",
      });
    } finally {
      await server.stop();
    }
  });
});
