import { spawn, type ChildProcess } from "node:child_process";
import { randomUUID } from "node:crypto";
import { constants } from "node:fs";
import { open } from "node:fs/promises";
import { createServer } from "node:net";
import type { LocalModelServerLaunch } from "./local-runtime/model-server.js";
import { verifyLocalModelServerLaunch } from "./local-runtime/model-server.js";

/** Owns only the child started by this agent session; never adopts an arbitrary PID. */
export class AgentLocalServer {
  private child?: ChildProcess;
  private record?: { server_id: string; model_id: string; url: string; log_path: string; state: "starting" | "ready" | "failed" | "stopped"; error?: string };
  private starting = false;
  private readonly onExit = () => this.kill("SIGKILL");
  private readonly signalHandlers = new Map<NodeJS.Signals, () => void>();
  private attachCleanup() {
    process.once("exit", this.onExit);
    for (const signal of ["SIGTERM", "SIGHUP", "SIGINT"] as const) {
      const handler = () => {
        // A UI may use Ctrl-C to cancel only the current turn. Its exit path
        // triggers onExit; do not stop serving when that owner keeps running.
        if (process.listenerCount(signal) > 0) {
          process.once(signal, handler);
          return;
        }
        this.kill("SIGKILL");
        process.kill(process.pid, signal);
      };
      this.signalHandlers.set(signal, handler);
      process.once(signal, handler);
    }
  }
  private detachCleanup() {
    process.off("exit", this.onExit);
    for (const [signal, handler] of this.signalHandlers) process.off(signal, handler);
    this.signalHandlers.clear();
  }
  private kill(signal: NodeJS.Signals) {
    const child = this.child;
    if (!child || child.exitCode !== null || child.signalCode !== null) return;
    try { if (child.pid) process.kill(-child.pid, signal); } catch { child.kill(signal); }
  }
  async start(launch: LocalModelServerLaunch, logPath: string) {
    if (process.platform !== "linux") throw new Error("Agent serving requires Linux and CUDA.");
    if (this.child || this.starting) throw new Error("This conversation already owns a model server. Stop it before starting another.");
    this.starting = true;
    try {
      await verifyLocalModelServerLaunch(launch);
      const address = new URL(launch.url);
      if (address.hostname !== "127.0.0.1") throw new Error("Agent model servers bind to loopback only.");
      await new Promise<void>((resolve, reject) => {
        const probe = createServer();
        probe.once("error", reject);
        probe.listen(Number(address.port), "127.0.0.1", () => probe.close(error => error ? reject(error) : resolve()));
      });
      const log = await open(logPath, constants.O_WRONLY | constants.O_APPEND | constants.O_CREAT | constants.O_NOFOLLOW, 0o600);
      const record = { server_id: randomUUID(), model_id: launch.modelName, url: launch.url, log_path: logPath, state: "starting" as const };
      this.record = record;
      try {
        const child = spawn(launch.command, launch.commandArgs, { env: launch.env, stdio: ["ignore", log.fd, log.fd], detached: true });
        this.child = child;
        this.attachCleanup();
        const finished = (error?: string) => {
          // uv or the serving parent may exit while GPU worker descendants remain.
          try { if (child.pid) process.kill(-child.pid, "SIGKILL"); } catch { /* Group already exited. */ }
          if (this.child === child) this.child = undefined;
          if (this.record?.server_id === record.server_id && this.record.state !== "stopped") {
            this.record.state = "failed"; this.record.error = error ?? "Server exited; inspect the log before retrying.";
          }
          this.detachCleanup();
        };
        child.once("error", error => finished(error.message));
        child.once("exit", (code, signal) => finished(`Server exited (${signal ?? code}).`));
        await new Promise<void>((resolve, reject) => { child.once("spawn", resolve); child.once("error", reject); });
        child.unref();
      } finally { await log.close(); }
      return { ...this.record, note: "Model loading has started. Use get_local_service to confirm health and served model identity before sending requests. Exiting this tt process stops its server." };
    } finally { this.starting = false; }
  }
  async status() {
    if (!this.record) return { state: "stopped", server_id: null };
    const record = this.record;
    const child = this.child;
    if (child) {
      try {
        const response = await fetch(`${record.url}/v1/models`, { signal: AbortSignal.timeout(2000), redirect: "error" });
        const body = await response.json() as { data?: { id?: string }[] };
        const health = await fetch(`${record.url}/health`, { signal: AbortSignal.timeout(2000), redirect: "error" });
        if (this.child === child && response.ok && health.ok && body.data?.some(model => model.id === record.model_id)) record.state = "ready";
        else if (this.child === child) record.state = "starting";
      } catch { if (this.child === child) record.state = "starting"; }
    }
    return { ...this.record };
  }
  async stop(serverId: string) {
    if (!this.record || this.record.server_id !== serverId) throw new Error("Server selection changed; inspect the current service before stopping it.");
    const child = this.child;
    if (!child) return { ...this.record };
    this.record.state = "stopped";
    await new Promise<void>(resolve => {
      let timer: NodeJS.Timeout;
      const done = () => { clearTimeout(timer); resolve(); };
      child.once("exit", done);
      this.kill("SIGTERM");
      timer = setTimeout(() => { this.kill("SIGKILL"); }, 5000);
    });
    return { ...this.record };
  }
}
