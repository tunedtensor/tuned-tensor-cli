import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
} from "vitest";
import {
  mkdtemp,
  readFile,
  realpath,
  stat,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, dirname, basename } from "node:path";
import { PassThrough } from "node:stream";
import {
  localCommandStreamsStdout,
  runLocalCommand,
} from "../local-runner.js";

interface CapturedStream {
  stream: PassThrough;
  value(): string;
}

function capturedStream(): CapturedStream {
  const stream = new PassThrough();
  let output = "";
  stream.setEncoding("utf8");
  stream.on("data", (chunk: string) => {
    output += chunk;
  });
  return { stream, value: () => output };
}

async function fakeEntrypoint(root: string): Promise<string> {
  const path = join(root, "fake-local.mjs");
  await writeFile(path, `
import { readFileSync } from "node:fs";

const args = process.argv.slice(2);
console.error("tt-local progress");

if (args.includes("--fail")) {
  console.error("tt-local exploded");
  process.exitCode = 7;
} else if (!args.includes("--print-command") && (
  args[0] === "serve"
  || (args[0] === "models" && args[1] === "serve")
)) {
  console.log("local server stdout");
} else if (args[0] === "wait-signal") {
  const finishSignal = (signal) => {
    setTimeout(() => {
      console.log(JSON.stringify({ cancelled: signal }));
      process.exit(0);
    }, 75);
  };
  process.on("SIGINT", () => finishSignal("SIGINT"));
  process.on("SIGTERM", () => finishSignal("SIGTERM"));
  setInterval(() => {}, 1_000);
} else if (args[0] === "wait-two-signals") {
  const signals = [];
  const recordSignal = (signal) => {
    signals.push(signal);
    if (signals.length === 2) {
      console.log(JSON.stringify({ signals }));
      process.exit(0);
    }
  };
  process.on("SIGINT", () => recordSignal("SIGINT"));
  process.on("SIGTERM", () => recordSignal("SIGTERM"));
  setInterval(() => {}, 1_000);
} else {
  const specPath = args.find((arg) =>
    arg.includes(".json") && !arg.endsWith("local-runner.json")
  );
  console.log(JSON.stringify({
    args,
    specPath: specPath ?? null,
    spec: specPath ? JSON.parse(readFileSync(specPath, "utf8")) : null,
    cwd: process.cwd(),
  }));
}
`, "utf8");
  return path;
}

let roots: string[] = [];

beforeEach(() => {
  roots = [];
});

afterEach(async () => {
  await Promise.all(roots.map(async (root) => {
    const { rm } = await import("node:fs/promises");
    await rm(root, { recursive: true, force: true });
  }));
});

async function testRoot(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "tt-local-adapter-"));
  roots.push(root);
  return root;
}

describe("runLocalCommand", () => {
  it("forwards the complete argument vector and numeric exit code", async () => {
    const root = await testRoot();
    const entrypoint = await fakeEntrypoint(root);
    const errors = capturedStream();
    const result = await runLocalCommand(
      ["runs", "get", "run-123", "--unknown-local-option=value"],
      {
        entrypoint,
        cwd: root,
        stderr: errors.stream,
      },
    );

    expect(result.exitCode).toBe(0);
    expect(result.signal).toBeNull();
    expect(result.entrypoint).toBe(entrypoint);
    expect(result.json).toMatchObject({
      args: ["runs", "get", "run-123", "--unknown-local-option=value"],
      cwd: await realpath(root),
    });
    expect(errors.value()).toContain("tt progress");

    const failed = await runLocalCommand(["info", "--fail"], {
      entrypoint,
      cwd: root,
      stderr: errors.stream,
      jsonMode: true,
    });
    expect(failed.exitCode).toBe(7);
    expect(failed.stderr).toContain("tt-local exploded");
  });

  it("streams server stdout while non-server commands remain captured", async () => {
    const root = await testRoot();
    const entrypoint = await fakeEntrypoint(root);
    const output = capturedStream();
    const errors = capturedStream();

    const result = await runLocalCommand(["serve", "active"], {
      entrypoint,
      cwd: root,
      stdout: output.stream,
      stderr: errors.stream,
    });

    expect(result.streamingStdout).toBe(true);
    expect(result.stdout).toBe("");
    expect(output.value()).toContain("local server stdout");

    const plan = await runLocalCommand(
      ["models", "serve", "active", "--print-command"],
      {
        entrypoint,
        cwd: root,
        stdout: output.stream,
        stderr: errors.stream,
      },
    );
    expect(plan.streamingStdout).toBe(false);
    expect(plan.stdout).toContain('"args"');
  });

  it.each([
    (spec: string) => ["validate", spec],
    (spec: string) => ["models", "prefetch", spec],
    (spec: string) => ["serve", "base", "--spec", spec, "--print-command"],
  ])(
    "projects only the local spec fields into an adjacent temporary file",
    async (command) => {
      const root = await testRoot();
      const entrypoint = await fakeEntrypoint(root);
      const specPath = join(root, "tunedtensor.json");
      const original = {
        id: "5c6a2436-f43f-4d50-8f44-5150ee4af16e",
        name: "Unified project",
        description: "",
        base_model: "qwen/qwen3.5-2b-base",
        system_prompt: "Return one label.",
        guidelines: [],
        constraints: [],
        examples: [
          { input: "good", output: "positive" },
          { input: "bad", output: "negative" },
        ],
        eval_cases: [{ input: "ignored locally", runtime: "python", tests: [] }],
        dataset_prebuilt: {
          training: "data/train.jsonl",
          validation: "data/validation.jsonl",
          format: "chat_jsonl",
        },
      };
      await writeFile(specPath, `${JSON.stringify(original, null, 2)}\n`);

      const result = await runLocalCommand(command(specPath), {
        entrypoint,
        cwd: root,
        stderr: capturedStream().stream,
      });
      const child = result.json as {
        specPath: string;
        spec: Record<string, unknown>;
      };
      const projectedPath = result.projectedArgs.find((argument) =>
        argument.includes(".tt-local-")
      );

      expect(projectedPath).toBeDefined();
      expect(dirname(projectedPath!)).toBe(root);
      expect(basename(projectedPath!)).toMatch(
        /^\.tunedtensor\.json\.tt-local-/,
      );
      expect(child.specPath).toBe(specPath);
      expect(child.spec).not.toHaveProperty("eval_cases");
      expect(child.spec).toMatchObject({
        base_model: "Qwen/Qwen3.5-2B",
        dataset_prebuilt: {
          training: "data/train.jsonl",
          validation: "data/validation.jsonl",
        },
      });
      expect(result.droppedSpecKeys).toEqual(["eval_cases"]);
      await expect(stat(projectedPath!)).rejects.toMatchObject({
        code: "ENOENT",
      });
      expect(JSON.parse(await readFile(specPath, "utf8"))).toEqual(original);
    },
  );

  it("keeps an already local-compatible spec at its original path", async () => {
    const root = await testRoot();
    const entrypoint = await fakeEntrypoint(root);
    const specPath = join(root, "tunedtensor.json");
    await writeFile(specPath, `${JSON.stringify({
      name: "Local project",
      description: "",
      base_model: "Qwen/Qwen3.5-2B",
      system_prompt: "Be concise.",
      guidelines: [],
      constraints: [],
      examples: [
        { input: "one", output: "1" },
        { input: "two", output: "2" },
      ],
    })}\n`);

    const result = await runLocalCommand(["validate", specPath], {
      entrypoint,
      cwd: root,
      stderr: capturedStream().stream,
    });

    expect((result.json as { specPath: string }).specPath).toBe(specPath);
    expect(result.projectedArgs).toEqual(["validate", specPath]);
    expect(result.droppedSpecKeys).toEqual([]);
  });

  it("preserves native implicit serve-base semantics without injecting a spec", async () => {
    const root = await testRoot();
    const entrypoint = await fakeEntrypoint(root);
    await writeFile(join(root, "tunedtensor.json"), `${JSON.stringify({
      name: "Cloud-shaped adjacent project",
      base_model: "Qwen/Qwen3.5-2B",
      eval_cases: [{ input: "do not project implicitly" }],
    })}\n`);

    const result = await runLocalCommand(
      ["serve", "base", "--print-command"],
      {
        entrypoint,
        cwd: root,
        stderr: capturedStream().stream,
      },
    );

    expect(result.projectedArgs).toEqual([
      "serve",
      "base",
      "--print-command",
    ]);
    expect(result.droppedSpecKeys).toEqual([]);
    expect((result.json as { specPath: string | null }).specPath).toBeNull();
  });

  it("lets the local CLI handle help before reading an adjacent spec", async () => {
    const root = await testRoot();
    const entrypoint = await fakeEntrypoint(root);
    await writeFile(join(root, "tunedtensor.json"), "{not valid json");

    const result = await runLocalCommand(["validate", "--help"], {
      entrypoint,
      cwd: root,
      stderr: capturedStream().stream,
    });

    expect(result.exitCode).toBe(0);
    expect(result.projectedArgs).toEqual(["validate", "--help"]);
    expect((result.json as { args: string[] }).args).toEqual([
      "validate",
      "--help",
    ]);
  });

  it("leaves full run-request payloads for the local CLI to reject", async () => {
    const root = await testRoot();
    const entrypoint = await fakeEntrypoint(root);
    const requestPath = join(root, "request.json");
    await writeFile(requestPath, `${JSON.stringify({
      run_id: "5c6a2436-f43f-4d50-8f44-5150ee4af16e",
      behavior_spec_id: "cd438417-36b6-4ac6-9201-f3bed0bdfc2c",
      run_number: 1,
      spec_snapshot: {
        name: "Not a public spec file",
        base_model: "Qwen/Qwen3.5-2B",
      },
    })}\n`);

    const result = await runLocalCommand(["validate", requestPath], {
      entrypoint,
      cwd: root,
      stderr: capturedStream().stream,
    });

    expect((result.json as { specPath: string }).specPath).toBe(requestPath);
    expect(result.projectedArgs).toEqual(["validate", requestPath]);
  });

  it("returns a structured setup failure instead of throwing", async () => {
    const root = await testRoot();
    const result = await runLocalCommand(["info"], {
      entrypoint: await fakeEntrypoint(root),
      nodeExecutable: join(root, "missing-node"),
      cwd: root,
      stderr: capturedStream().stream,
    });

    expect(result.exitCode).not.toBe(0);
    expect(result.errorMessage).toBeTruthy();
    expect(result.hasJson).toBe(false);
  });

  it("forwards SIGINT once and waits for the local child to clean up", async () => {
    const root = await testRoot();
    const entrypoint = await fakeEntrypoint(root);
    const listenersBefore = process.listenerCount("SIGINT");
    const running = runLocalCommand(["wait-signal"], {
      entrypoint,
      cwd: root,
      stderr: capturedStream().stream,
    });
    await new Promise((resolveWait) => setTimeout(resolveWait, 75));
    process.emit("SIGINT");

    const result = await running;
    expect(result.exitCode).toBe(0);
    expect(result.json).toEqual({ cancelled: "SIGINT" });
    expect(process.listenerCount("SIGINT")).toBe(listenersBefore);
  });

  it.runIf(process.platform !== "win32")(
    "translates SIGHUP into a cleanup-capable SIGTERM",
    async () => {
      const root = await testRoot();
      const entrypoint = await fakeEntrypoint(root);
      const listenersBefore = process.listenerCount("SIGHUP");
      const running = runLocalCommand(["wait-signal"], {
        entrypoint,
        cwd: root,
        stderr: capturedStream().stream,
      });
      await new Promise((resolveWait) => setTimeout(resolveWait, 75));
      process.emit("SIGHUP");

      const result = await running;
      expect(result.exitCode).toBe(0);
      expect(result.json).toEqual({ cancelled: "SIGTERM" });
      expect(process.listenerCount("SIGHUP")).toBe(listenersBefore);
    },
  );

  it("forwards a later termination signal while cleanup is pending", async () => {
    const root = await testRoot();
    const entrypoint = await fakeEntrypoint(root);
    const sigintListeners = process.listenerCount("SIGINT");
    const sigtermListeners = process.listenerCount("SIGTERM");
    const running = runLocalCommand(["wait-two-signals"], {
      entrypoint,
      cwd: root,
      stderr: capturedStream().stream,
    });
    await new Promise((resolveWait) => setTimeout(resolveWait, 75));
    process.emit("SIGINT");
    process.emit("SIGTERM");

    const result = await running;
    expect(result.exitCode).toBe(0);
    expect(result.json).toEqual({ signals: ["SIGINT", "SIGTERM"] });
    expect(process.listenerCount("SIGINT")).toBe(sigintListeners);
    expect(process.listenerCount("SIGTERM")).toBe(sigtermListeners);
  });
});

describe("localCommandStreamsStdout", () => {
  it("distinguishes a serving process from a printed launch plan", () => {
    expect(localCommandStreamsStdout(["serve", "active"])).toBe(true);
    expect(localCommandStreamsStdout(["models", "serve", "base"])).toBe(true);
    expect(
      localCommandStreamsStdout([
        "models",
        "serve",
        "active",
        "--print-command",
      ]),
    ).toBe(false);
    expect(localCommandStreamsStdout(["serve", "base", "--help"])).toBe(false);
    expect(localCommandStreamsStdout(["models", "serve", "-h"])).toBe(false);
    expect(localCommandStreamsStdout(["runs", "report", "abc"])).toBe(false);
  });
});
