import {
  chmod,
  lstat,
  mkdir,
  readFile,
  readdir,
  rename,
  stat,
  unlink,
  writeFile,
} from "node:fs/promises";
import { randomUUID } from "node:crypto";
import { join } from "node:path";
import type { AgentAction, AgentThread } from "./agent-client.js";

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const DEFAULT_MAX_THREAD_BYTES = 1_000_000;
const DEFAULT_MAX_MESSAGES = 200;
const DEFAULT_MAX_ACTIONS = 200;
const DEFAULT_MAX_THREADS = 100;
const DEFAULT_MAX_CLAIMS = 1_000;

export interface StoredAgentThread {
  thread: AgentThread;
  messages: unknown[];
  actions: AgentAction[];
}

export interface LocalAgentStoreOptions {
  secretValues?: readonly string[];
  /** Re-read on each persist so /login keys saved later are still redacted. */
  secretValueProvider?: () => readonly string[];
  maxThreadBytes?: number;
  maxMessages?: number;
  maxActions?: number;
  maxThreads?: number;
  maxClaims?: number;
}

function assertThreadId(id: string): void {
  if (!UUID.test(id)) throw new Error("Invalid local agent thread ID.");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function scrub(value: unknown, secrets: readonly string[]): unknown {
  if (typeof value === "string") {
    return secrets.reduce(
      (text, secret) => secret ? text.replaceAll(secret, "[REDACTED]") : text,
      value,
    );
  }
  if (Array.isArray(value)) return value.map((entry) => scrub(entry, secrets));
  if (isRecord(value)) {
    return Object.fromEntries(
      Object.entries(value).map(([key, entry]) => [key, scrub(entry, secrets)]),
    );
  }
  return value;
}

export class LocalAgentStore {
  private readonly threadsDir: string;
  private readonly claimsDir: string;
  private readonly secrets: readonly string[];
  private readonly secretValueProvider: (() => readonly string[]) | undefined;
  private readonly maxThreadBytes: number;
  private readonly maxMessages: number;
  private readonly maxActions: number;
  private readonly maxThreads: number;
  private readonly maxClaims: number;

  constructor(
    private readonly rootDir: string,
    options: LocalAgentStoreOptions = {},
  ) {
    this.threadsDir = join(rootDir, "threads");
    this.claimsDir = join(rootDir, "action-claims");
    this.secrets = options.secretValues?.filter(Boolean) ?? [];
    this.secretValueProvider = options.secretValueProvider;
    this.maxThreadBytes = options.maxThreadBytes ?? DEFAULT_MAX_THREAD_BYTES;
    this.maxMessages = options.maxMessages ?? DEFAULT_MAX_MESSAGES;
    this.maxActions = options.maxActions ?? DEFAULT_MAX_ACTIONS;
    this.maxThreads = options.maxThreads ?? DEFAULT_MAX_THREADS;
    this.maxClaims = options.maxClaims ?? DEFAULT_MAX_CLAIMS;
  }

  redact<T>(value: T): T {
    let extra: readonly string[] = [];
    try {
      extra = this.secretValueProvider?.() ?? [];
    } catch {
      extra = [];
    }
    return scrub(value, [...this.secrets, ...extra]) as T;
  }

  private async ensureDirectories(): Promise<void> {
    await mkdir(this.rootDir, { recursive: true, mode: 0o700 });
    await chmod(this.rootDir, 0o700);
    await mkdir(this.threadsDir, { recursive: true, mode: 0o700 });
    await chmod(this.threadsDir, 0o700);
    await mkdir(this.claimsDir, { recursive: true, mode: 0o700 });
    await chmod(this.claimsDir, 0o700);
  }

  private path(id: string): string {
    assertThreadId(id);
    return join(this.threadsDir, `${id}.json`);
  }

  private serializeWithinQuota(state: StoredAgentThread): string {
    const active = state.actions.filter((action) =>
      action.status === "proposed" || action.status === "executing" || action.status === "outcome_unknown"
    );
    const terminal = state.actions.filter((action) => !active.includes(action));
    const compact: StoredAgentThread = this.redact({
      thread: state.thread,
      messages: state.messages.slice(-this.maxMessages),
      actions: [...active, ...terminal.slice(-Math.max(0, this.maxActions - active.length))],
    });
    let serialized = `${JSON.stringify(compact, null, 2)}\n`;
    while (Buffer.byteLength(serialized) > this.maxThreadBytes && compact.messages.length > 0) {
      compact.messages.shift();
      serialized = `${JSON.stringify(compact, null, 2)}\n`;
    }
    if (Buffer.byteLength(serialized) > this.maxThreadBytes) {
      throw new Error("Local agent state exceeds its bounded storage quota.");
    }
    return serialized;
  }

  private async pruneDirectory(
    directory: string,
    suffix: RegExp,
    maximum: number,
    preserve?: string,
  ): Promise<void> {
    const entries = await Promise.all((await readdir(directory))
      .filter((name) => suffix.test(name) && name !== preserve)
      .map(async (name) => ({ name, modified: (await stat(join(directory, name))).mtimeMs })));
    entries.sort((a, b) => a.modified - b.modified);
    const keepOther = Math.max(0, maximum - (preserve ? 1 : 0));
    for (const entry of entries.slice(0, Math.max(0, entries.length - keepOther))) {
      await unlink(join(directory, entry.name));
    }
  }

  private async pruneThreads(preserve: string): Promise<void> {
    const names = (await readdir(this.threadsDir))
      .filter((name) => /^[0-9a-f-]{36}\.json$/i.test(name));
    const protectedNames = new Set([preserve]);
    for (const name of names) {
      if (name === preserve) continue;
      try {
        const state = await this.load(name.slice(0, -5));
        if (state.actions.some((action) =>
          action.status === "proposed" ||
          action.status === "executing" ||
          action.status === "outcome_unknown"
        )) {
          protectedNames.add(name);
        }
      } catch {
        // Corrupt files are not safety records and remain eligible for pruning.
      }
    }
    const candidates = await Promise.all(names
      .filter((name) => !protectedNames.has(name))
      .map(async (name) => ({
        name,
        modified: (await stat(join(this.threadsDir, name))).mtimeMs,
      })));
    candidates.sort((a, b) => a.modified - b.modified);
    const keep = Math.max(0, this.maxThreads - protectedNames.size);
    for (const entry of candidates.slice(0, Math.max(0, candidates.length - keep))) {
      await unlink(join(this.threadsDir, entry.name));
    }
  }

  async save(state: StoredAgentThread): Promise<void> {
    assertThreadId(state.thread.id);
    await this.ensureDirectories();
    const destination = this.path(state.thread.id);
    const temporary = join(
      this.threadsDir,
      `.${state.thread.id}.${randomUUID()}.tmp`,
    );
    let serialized: string;
    try {
      serialized = this.serializeWithinQuota(state);
    } catch (error) {
      if (error instanceof Error) throw error;
      throw new Error("Local agent conversation could not be serialized.");
    }
    await writeFile(temporary, serialized, { mode: 0o600, flag: "wx" });
    await chmod(temporary, 0o600);
    await rename(temporary, destination);
    await chmod(destination, 0o600);
    await this.pruneThreads(`${state.thread.id}.json`);
  }

  async claimAction(actionId: string): Promise<void> {
    assertThreadId(actionId);
    await this.ensureDirectories();
    const path = join(this.claimsDir, `${actionId}.claim`);
    try {
      await writeFile(path, `${new Date().toISOString()}\n`, {
        mode: 0o600,
        flag: "wx",
      });
      await chmod(path, 0o600);
      await this.pruneDirectory(
        this.claimsDir,
        /^[0-9a-f-]{36}\.claim$/i,
        this.maxClaims,
        `${actionId}.claim`,
      );
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "EEXIST") {
        throw new Error(`Action ${actionId} is already being or has been settled.`);
      }
      throw error;
    }
  }

  async load(id: string): Promise<StoredAgentThread> {
    const path = this.path(id);
    const info = await lstat(path);
    if (!info.isFile() || info.isSymbolicLink()) {
      throw new Error("Invalid local agent thread file.");
    }
    const parsed = JSON.parse(await readFile(path, "utf8")) as unknown;
    if (!isRecord(parsed) || !isRecord(parsed.thread) || parsed.thread.id !== id ||
      !Array.isArray(parsed.messages) || !Array.isArray(parsed.actions)) {
      throw new Error("Local agent thread file is malformed.");
    }
    return parsed as unknown as StoredAgentThread;
  }

  async list(): Promise<StoredAgentThread[]> {
    await this.ensureDirectories();
    const names = await readdir(this.threadsDir);
    const states: StoredAgentThread[] = [];
    for (const name of names) {
      const match = /^([0-9a-f-]{36})\.json$/i.exec(name);
      if (!match || !UUID.test(match[1]!)) continue;
      try {
        states.push(await this.load(match[1]!));
      } catch {
        // One corrupt entry must not make all durable conversations unavailable.
      }
    }
    return states.sort((a, b) =>
      b.thread.updated_at.localeCompare(a.thread.updated_at)
    );
  }
}
