import { createServer } from "node:net";
import { mkdtemp, rm, symlink, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, expect, it, vi } from "vitest";
import { AgentLocalServer } from "../agent-local-server.js";
vi.mock("../local-runtime/model-server.js", () => ({ verifyLocalModelServerLaunch: vi.fn(async () => {}) }));
const roots: string[] = [];
const servers: AgentLocalServer[] = [];
afterEach(async () => {
  for (const service of servers.splice(0)) { const status = await service.status(); if (status.server_id) await service.stop(status.server_id); }
  for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true });
});
async function freePort() {
  const socket = createServer();
  await new Promise<void>(resolve => socket.listen(0, "127.0.0.1", resolve));
  const port = (socket.address() as { port: number }).port;
  await new Promise<void>(resolve => socket.close(() => resolve()));
  return port;
}
it.skipIf(process.platform !== "linux")("reports loading before readiness, verifies served identity, then stops the owned process", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-agent-server-")); roots.push(root);
  const port = await freePort();
  const service = new AgentLocalServer(); servers.push(service);
  const code = `require('http').createServer((req,res)=>{res.setHeader('content-type','application/json');res.end(JSON.stringify(req.url==='/v1/models'?{data:[{id:'local-reviewed-model'}]}:{ok:true}));}).listen(${port},'127.0.0.1');`;
  const launch = { command: process.execPath, commandArgs: ["-e", code], displayCommand: [], env: process.env, url: `http://127.0.0.1:${port}`, modelName: "local-reviewed-model" };
  const started = await service.start(launch, join(root, "serve.log"));
  expect(started.state).toBe("starting");
  await expect(service.start(launch, join(root, "serve.log"))).rejects.toThrow(/already owns/);
  await vi.waitFor(async () => expect((await service.status()).state).toBe("ready"), { timeout: 5000 });
  const cancelTurn = vi.fn();
  process.on("SIGINT", cancelTurn);
  try {
    process.emit("SIGINT");
    expect(cancelTurn).toHaveBeenCalledTimes(1);
    expect((await service.status()).state).toBe("ready");
  } finally { process.off("SIGINT", cancelTurn); }
  await expect(service.stop("wrong-id")).rejects.toThrow(/selection changed/);
  expect((await service.stop(started.server_id!)).state).toBe("stopped");
  await expect(fetch(`${launch.url}/health`)).rejects.toThrow();
});
it.skipIf(process.platform !== "linux")("does not announce readiness for the wrong model identity", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-agent-server-")); roots.push(root);
  const port = await freePort();
  const service = new AgentLocalServer(); servers.push(service);
  await service.start({ command: process.execPath, commandArgs: ["-e", `require('http').createServer((q,r)=>r.end(JSON.stringify({data:[{id:'wrong'}]}))).listen(${port},'127.0.0.1')`], displayCommand: [], env: process.env, url: `http://127.0.0.1:${port}`, modelName: "expected" }, join(root, "serve.log"));
  await vi.waitFor(async () => expect((await fetch(`http://127.0.0.1:${port}/health`)).ok).toBe(true));
  expect((await service.status()).state).toBe("starting");
});

it.skipIf(process.platform !== "linux")("refuses an occupied port and symlink log, and reports a failed child honestly", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-agent-server-")); roots.push(root);
  const socket = createServer();
  await new Promise<void>(resolve => socket.listen(0, "127.0.0.1", resolve));
  const port = (socket.address() as { port: number }).port;
  const service = new AgentLocalServer(); servers.push(service);
  const launch = { command: process.execPath, commandArgs: ["-e", "process.exit(7)"], displayCommand: [], env: process.env, url: `http://127.0.0.1:${port}`, modelName: "expected" };
  try { await expect(service.start(launch, join(root, "serve.log"))).rejects.toThrow(/EADDRINUSE/); }
  finally { await new Promise<void>(resolve => socket.close(() => resolve())); }
  expect(await service.status()).toMatchObject({ state: "stopped", server_id: null });
  await writeFile(join(root, "target"), "unchanged");
  await symlink(join(root, "target"), join(root, "link.log"));
  await expect(service.start(launch, join(root, "link.log"))).rejects.toThrow();
  expect(await readFile(join(root, "target"), "utf8")).toBe("unchanged");
  await service.start(launch, join(root, "serve.log"));
  await vi.waitFor(async () => expect(await service.status()).toMatchObject({ state: "failed", error: "Server exited (7)." }));
});

it.skipIf(process.platform !== "linux")("health probes never follow redirects away from the owned endpoint", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-agent-server-")); roots.push(root);
  const port = await freePort();
  const service = new AgentLocalServer(); servers.push(service);
  const code = `require('http').createServer((q,r)=>{if(q.url==='/v1/models'){r.writeHead(302,{location:'/redirected'});r.end();}else{require('fs').writeFileSync(${JSON.stringify(join(root, "followed"))},'unsafe');r.end(JSON.stringify({data:[{id:'expected'}]}));}}).listen(${port},'127.0.0.1')`;
  await service.start({ command: process.execPath, commandArgs: ["-e", code], displayCommand: [], env: process.env, url: `http://127.0.0.1:${port}`, modelName: "expected" }, join(root, "serve.log"));
  await vi.waitFor(async () => expect((await fetch(`http://127.0.0.1:${port}/v1/models`, { redirect: "manual" })).status).toBe(302));
  expect((await service.status()).state).toBe("starting");
  await expect(readFile(join(root, "followed"))).rejects.toMatchObject({ code: "ENOENT" });
});
