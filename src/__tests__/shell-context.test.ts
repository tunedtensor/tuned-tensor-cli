import { afterEach, describe, expect, it } from "vitest";
import { existsSync } from "node:fs";
import {
  mkdir,
  mkdtemp,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  discoverShellContext,
  formatShellContext,
  formatShellStatus,
  redactApiKey,
  targetFromEnvironment,
} from "../shell-context.js";

const roots: string[] = [];

async function temporaryRoot(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "tt-shell-context-"));
  roots.push(root);
  return root;
}

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) =>
    rm(root, { recursive: true, force: true })
  ));
});

describe("discoverShellContext", () => {
  it("discovers an adjacent local project and never exposes the API key", async () => {
    const root = await temporaryRoot();
    const project = join(root, "support adapter");
    const home = join(root, "home");
    const cloudConfigDirectory = join(home, ".config", "tuned-tensor");
    const storeRoot = join(project, "state");
    const runId = "11111111-1111-4111-8111-111111111111";
    const fullKey = `tt_${"s".repeat(48)}`;
    await mkdir(cloudConfigDirectory, { recursive: true });
    await mkdir(join(storeRoot, "runs", runId), { recursive: true });
    await mkdir(join(storeRoot, "models", "local-model-one"), { recursive: true });
    await writeFile(join(project, "tunedtensor.json"), JSON.stringify({
      id: "22222222-2222-4222-8222-222222222222",
      name: "Support Adapter",
      base_model: "Qwen/Qwen3.5-2B",
      examples: [{ input: "one", output: "1" }, { input: "two", output: "2" }],
    }));
    await writeFile(join(project, "local-runner.json"), JSON.stringify({
      artifactRoot: "artifacts",
      storeRoot: "state",
    }));
    await writeFile(join(cloudConfigDirectory, "config.json"), JSON.stringify({
      api_key: fullKey,
      base_url: "https://api.example.test",
    }));
    await writeFile(join(storeRoot, "active-model.json"), JSON.stringify({
      model_id: "local-model-one",
    }));
    await writeFile(join(storeRoot, "models", "local-model-one", "model.json"), JSON.stringify({
      id: "local-model-one",
      name: "Qwen/Qwen3.5-2B (11111111)",
      base_model: "Qwen/Qwen3.5-2B",
      created_at: "2026-07-25T10:00:00.000Z",
    }));
    await writeFile(join(storeRoot, "runs", runId, "state.json"), JSON.stringify({
      id: runId,
      status: "completed",
      spec_name: "Support Adapter",
      updated_at: "2026-07-26T10:00:00.000Z",
    }));

    const context = await discoverShellContext({
      cwd: project,
      env: { HOME: home },
    });

    expect(context.inferredTarget).toBe("local");
    expect(context.targetSource).toBe("default-local");
    expect(context.spec).toMatchObject({
      name: "Support Adapter",
      baseModel: "Qwen/Qwen3.5-2B",
      exampleCount: 2,
      parseError: false,
    });
    expect(context.local).toMatchObject({
      configPath: join(project, "local-runner.json"),
      artifactRoot: join(project, "artifacts"),
      storeRoot,
      activeModelId: "local-model-one",
      latestRun: {
        id: runId,
        status: "completed",
      },
    });
    expect(context.local.models).toEqual([{
      id: "local-model-one",
      name: "Qwen/Qwen3.5-2B (11111111)",
      baseModel: "Qwen/Qwen3.5-2B",
      createdAt: "2026-07-25T10:00:00.000Z",
    }]);
    expect(context.cloud).toMatchObject({
      authenticated: true,
      baseUrl: "https://api.example.test",
      keyPrefix: `${fullKey.slice(0, 8)}…`,
    });
    expect(JSON.stringify(context)).not.toContain(fullKey);
  });

  it("surfaces the configured agent provider and model", async () => {
    const root = await temporaryRoot();
    const home = join(root, "home");
    const cloudConfigDirectory = join(home, ".config", "tuned-tensor");
    await mkdir(cloudConfigDirectory, { recursive: true });
    await writeFile(join(cloudConfigDirectory, "config.json"), JSON.stringify({
      agent: { provider: "anthropic", model: "claude-sonnet-4-5", thinking: "high" },
    }));

    const context = await discoverShellContext({
      cwd: root,
      env: { HOME: home },
    });

    expect(context.agent).toEqual({
      provider: "anthropic",
      model: "claude-sonnet-4-5",
      thinking: "high",
    });
    expect(formatShellContext(context, "cloud", context.targetSource).join("\n"))
      .toContain("Agent model    anthropic/claude-sonnet-4-5 (thinking high)");
  });

  it("prefers agent environment overrides over stored config", async () => {
    const root = await temporaryRoot();
    const home = join(root, "home");
    const cloudConfigDirectory = join(home, ".config", "tuned-tensor");
    await mkdir(cloudConfigDirectory, { recursive: true });
    await writeFile(join(cloudConfigDirectory, "config.json"), JSON.stringify({
      agent: { provider: "anthropic", model: "claude-sonnet-4-5" },
    }));

    const context = await discoverShellContext({
      cwd: root,
      env: {
        HOME: home,
        TUNED_TENSOR_AGENT_PROVIDER: "openrouter",
        TUNED_TENSOR_AGENT_MODEL: "meta-llama/llama-3.3-70b",
      },
    });

    expect(context.agent).toEqual({
      provider: "openrouter",
      model: "meta-llama/llama-3.3-70b",
      thinking: undefined,
    });
  });

  it("uses TT_TARGET when valid and reports an invalid override safely", async () => {
    const root = await temporaryRoot();
    await writeFile(join(root, "local-runner.json"), "{}");

    const cloud = await discoverShellContext({
      cwd: root,
      env: { HOME: join(root, "home"), TT_TARGET: "cloud" },
    });
    expect(cloud.inferredTarget).toBe("cloud");
    expect(cloud.targetSource).toBe("environment");

    const invalid = await discoverShellContext({
      cwd: root,
      env: { HOME: join(root, "home"), TT_TARGET: "remote" },
    });
    expect(invalid.inferredTarget).toBe("local");
    expect(invalid.warnings).toContain(
      'Ignoring invalid TT_TARGET="remote"; use cloud or local.',
    );
  });

  it("defaults to local even without an adjacent local config", async () => {
    const root = await temporaryRoot();
    const child = join(root, "child");
    await mkdir(child);
    await writeFile(join(root, "local-runner.json"), "{}");

    const context = await discoverShellContext({
      cwd: child,
      env: { HOME: join(root, "home") },
    });

    expect(context.inferredTarget).toBe("local");
    expect(context.targetSource).toBe("default-local");
    expect(context.local.configPath).toBeUndefined();
  });

  it("is read-only when project, config, and local store do not exist", async () => {
    const root = await temporaryRoot();
    const project = join(root, "empty-project");
    const home = join(root, "missing-home");
    await mkdir(project);

    const context = await discoverShellContext({
      cwd: project,
      env: { HOME: home },
    });

    expect(context.inferredTarget).toBe("local");
    expect(context.spec).toBeUndefined();
    expect(existsSync(join(project, ".tt-local"))).toBe(false);
    expect(existsSync(home)).toBe(false);
  });

  it("reports malformed project files without leaking their contents", async () => {
    const root = await temporaryRoot();
    await writeFile(join(root, "tunedtensor.json"), '{"api_key":"secret",');
    await writeFile(join(root, "local-runner.json"), '{"token":"secret",');

    const context = await discoverShellContext({
      cwd: root,
      env: { HOME: join(root, "home") },
    });

    expect(context.spec?.parseError).toBe(true);
    expect(context.warnings).toEqual([
      "tunedtensor.json could not be parsed.",
      "local-runner.json could not be parsed.",
    ]);
    expect(JSON.stringify(context)).not.toContain("secret");
  });
});

describe("context helpers", () => {
  it("redacts keys and parses only supported targets", () => {
    expect(redactApiKey("tt_abcdefghijklmnopqrstuvwxyz")).toBe("tt_abcde…");
    expect(redactApiKey("key")).toBe("…");
    expect(redactApiKey(undefined)).toBeUndefined();
    expect(targetFromEnvironment({ TT_TARGET: " LOCAL " })).toBe("local");
    expect(targetFromEnvironment({ TT_TARGET: "remote" })).toBeUndefined();
  });

  it("formats useful context and status without performing probes", async () => {
    const root = await temporaryRoot();
    const context = await discoverShellContext({
      cwd: root,
      env: {
        HOME: join(root, "home"),
        TUNED_TENSOR_API_KEY: `tt_${"a".repeat(48)}`,
      },
    });

    const contextText = formatShellContext(
      context,
      "cloud",
      context.targetSource,
    ).join("\n");
    const statusText = formatShellStatus(context, "local").join("\n");
    expect(contextText).toContain("Cloud auth     yes (tt_aaaaa…)");
    expect(statusText).toContain("Host checks    not run (use doctor)");
    expect(statusText).not.toContain("nvidia-smi");
  });
});
