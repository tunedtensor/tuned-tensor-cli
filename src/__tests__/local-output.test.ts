import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import { PassThrough } from "node:stream";
import {
  adaptLocalCliText,
  localCliErrorEnvelope,
  localCliTextEnvelope,
  renderLocalOutput,
  type LocalOutputPayload,
} from "../local-output.js";

function payload(
  overrides: Partial<LocalOutputPayload> = {},
): LocalOutputPayload {
  return {
    args: ["info"],
    exitCode: 0,
    signal: null,
    stdout: "",
    stderr: "",
    hasJson: false,
    json: undefined,
    streamingStdout: false,
    droppedSpecKeys: [],
    ...overrides,
  };
}

function capture(): {
  stream: PassThrough;
  value(): string;
} {
  const stream = new PassThrough();
  let value = "";
  stream.setEncoding("utf8");
  stream.on("data", (chunk: string) => {
    value += chunk;
  });
  return { stream, value: () => value };
}

beforeEach(() => {
  vi.spyOn(console, "log").mockImplementation(() => {});
  vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("local output primitives", () => {
  it("adapts the standalone binary name to the unified command", () => {
    expect(
      adaptLocalCliText(
        "Usage: tt-local run\nInstall tuned-tensor-local first\n"
          + "State: ~/.tuned-tensor-local and .tt-local/artifacts\n",
      ),
    ).toBe(
      "Usage: tt run\nInstall tt first\n"
        + "State: ~/.tuned-tensor-local and .tt-local/artifacts\n",
    );
  });

  it("preserves successful raw JSON in JSON mode", () => {
    const output = capture();
    const raw = '{"ok":true,"run_id":"abc"}\n';
    renderLocalOutput(payload({
      args: ["run"],
      stdout: raw,
      hasJson: true,
      json: { ok: true, run_id: "abc" },
    }), {
      jsonMode: true,
      stdout: output.stream,
    });

    expect(output.value()).toBe(raw);
  });

  it("normalizes a non-JSON local failure", () => {
    const output = capture();
    renderLocalOutput(payload({
      args: ["validate"],
      exitCode: 2,
      stderr: "tt-local: invalid spec\n",
    }), {
      jsonMode: true,
      stdout: output.stream,
    });

    expect(JSON.parse(output.value())).toEqual({
      error: {
        status: null,
        code: "LOCAL_CLI_ERROR",
        message: "tt: invalid spec",
        exit_code: 2,
        signal: null,
      },
    });
    expect(localCliErrorEnvelope(payload({
      exitCode: 9,
      errorMessage: "spawn failed",
    }))).toMatchObject({
      error: { code: "LOCAL_CLI_ERROR", exit_code: 9 },
    });
    expect(localCliErrorEnvelope(payload({
      exitCode: 1,
      stderr: '[\n  {"path":["name"],"message":"Name is required"}\n]\n',
    }))).toMatchObject({
      error: {
        message: '[\n  {"path":["name"],"message":"Name is required"}\n]',
      },
    });
  });

  it("normalizes a failed serve preflight even when stdout was streaming", () => {
    const output = capture();
    renderLocalOutput(payload({
      args: ["serve", "missing-model"],
      exitCode: 1,
      stderr: "Model not found: missing-model\n",
      streamingStdout: true,
    }), {
      jsonMode: true,
      stdout: output.stream,
    });

    expect(JSON.parse(output.value())).toMatchObject({
      error: {
        code: "LOCAL_CLI_ERROR",
        message: "Model not found: missing-model",
        exit_code: 1,
      },
    });
  });

  it("normalizes successful text commands into one JSON document", () => {
    const output = capture();
    const info = [
      "tuned-tensor-local: Local CUDA fine-tuning.",
      "Version: 0.4.0",
      "Status: local",
      "",
    ].join("\n");
    renderLocalOutput(payload({
      args: ["info"],
      stdout: info,
    }), {
      jsonMode: true,
      stdout: output.stream,
    });

    expect(JSON.parse(output.value())).toEqual({
      data: {
        name: "tt",
        description: "Local CUDA fine-tuning.",
        version: "0.4.0",
        status: "local",
      },
    });
    expect(localCliTextEnvelope(payload({
      args: ["--help"],
      stdout: "Usage: tt-local <command>\n",
    }))).toEqual({
      data: { output: "Usage: tt <command>" },
    });
  });
});

describe("human local rendering", () => {
  it("renders doctor checks as a table and a distinct failure summary", () => {
    renderLocalOutput(payload({
      args: ["doctor"],
      exitCode: 1,
      hasJson: true,
      json: {
        ok: false,
        checks: [
          { name: "node", ok: true, message: "Node 22.18.0" },
          { name: "python-runtime", ok: false, detail: "CUDA unavailable" },
        ],
      },
    }));

    const logs = vi.mocked(console.log).mock.calls.flat().join("\n");
    const errors = vi.mocked(console.error).mock.calls.flat().join("\n");
    expect(logs).toContain("python-runtime");
    expect(logs).toContain("CUDA unavailable");
    expect(errors).toContain("needs attention");
  });

  it("renders hardware capability as a host summary and workload table", () => {
    renderLocalOutput(payload({
      args: ["hardware"],
      hasJson: true,
      json: {
        summary: "GPU NVIDIA GB10, 128 GiB unified, CUDA yes",
        quick: true,
        capabilities: {
          cuda_available: true,
          gpu: { name: "NVIDIA GB10", memory_total_bytes: 137438953472 },
          adapters: [{
            id: "Qwen/Qwen3.5-2B",
            train: { status: "ready" },
            finetune: { status: "ready" },
            inference: { status: "ready" },
          }],
          foundation: {
            default_depth: 2,
            suggested_max_depth: 48,
            train: { status: "ready" },
            finetune: { status: "ready" },
            inference: { status: "ready" },
            serve: { reason: "tt serve cannot host foundation checkpoints yet" },
          },
          notes: [],
        },
      },
    }));

    const logs = vi.mocked(console.log).mock.calls.flat().join("\n");
    expect(logs).toContain("Qwen/Qwen3.5-2B");
    expect(logs).toContain("ready");
    expect(logs).toContain("NVIDIA GB10");
  });

  it("keeps process completion distinct from a failed regression gate", () => {
    renderLocalOutput(payload({
      args: ["run"],
      hasJson: true,
      json: {
        status: "completed",
        run_id: "5c6a2436-f43f-4d50-8f44-5150ee4af16e",
        model_id: "local-5c6a2436-f43f-4d50-8f44-5150ee4af16e",
        report_path: "/tmp/report.json",
        artifact_dir: "/tmp/artifacts",
        comparison: {
          avg_score_delta: 0.2,
          pass_rate_delta: 0.1,
        },
        general_regression: {
          passed: false,
          failures: ["pass-rate drop 0.08 exceeds 0.05"],
        },
      },
    }));

    const logs = vi.mocked(console.log).mock.calls.flat().join("\n");
    expect(logs).toContain("Local run completed");
    expect(logs).toContain("general regression gate failed");
    expect(logs).toContain("pass-rate drop");
    expect(console.error).not.toHaveBeenCalled();
  });

  it("renders run/model collections and activation with current tables", () => {
    renderLocalOutput(payload({
      args: ["runs", "list"],
      hasJson: true,
      json: [{
        id: "5c6a2436-f43f-4d50-8f44-5150ee4af16e",
        spec_name: "Support adapter",
        status: "completed",
        current_stage: "completed",
        updated_at: "2026-07-26T12:00:00.000Z",
      }],
    }));
    renderLocalOutput(payload({
      args: ["models", "list"],
      hasJson: true,
      json: [{
        id: "local-5c6a2436-f43f-4d50-8f44-5150ee4af16e",
        base_model: "Qwen/Qwen3.5-2B",
        run_id: "5c6a2436-f43f-4d50-8f44-5150ee4af16e",
        provider: "local-uv",
        created_at: "2026-07-26T12:00:00.000Z",
      }],
    }));
    renderLocalOutput(payload({
      args: ["models", "activate"],
      hasJson: true,
      json: {
        active: "local-5c6a2436-f43f-4d50-8f44-5150ee4af16e",
        pointer: {
          action: "activate",
          run_id: "5c6a2436-f43f-4d50-8f44-5150ee4af16e",
          previous_model_id: null,
          activated_at: "2026-07-26T12:00:00.000Z",
        },
      },
    }));

    const logs = vi.mocked(console.log).mock.calls.flat().join("\n");
    expect(logs).toContain("Support adapter");
    expect(logs).toContain("Qwen/Qwen3.5-2B");
    expect(logs).toContain("local-5c6a2436");
    expect(logs).toContain("Activated local model");
  });

  it("warns when cloud-only fields were projected away", () => {
    renderLocalOutput(payload({
      args: ["validate"],
      hasJson: true,
      json: {
        ok: true,
        input_path: "/tmp/projected.json",
        behavior_spec_id: "abc",
        base_model: "Qwen/Qwen3.5-2B",
      },
      droppedSpecKeys: ["eval_cases"],
    }));

    const logs = vi.mocked(console.log).mock.calls.flat().join("\n");
    expect(logs).toContain("eval_cases");
    expect(logs).toContain("valid");
  });
});
