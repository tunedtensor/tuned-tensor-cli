import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  existsSync,
  lstatSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { rename, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Value } from "typebox/value";
import {
  createLocalSpecProject,
  LocalProjectSpecSchema,
  prepareLocalSpecProject,
} from "../local-spec-workspace.js";

let workspace: string;

const spec = {
  name: "Sentiment classifier",
  base_model: "Qwen/Qwen3.5-2B" as const,
  system_prompt: "Classify sentiment and return only the label.",
  guidelines: ["Return positive, neutral, or negative."],
  constraints: ["Return one lowercase label."],
  examples: [
    { input: "Excellent work.", output: "positive" },
    { input: "This is disappointing.", output: "negative" },
  ],
};

beforeEach(() => {
  workspace = mkdtempSync(join(tmpdir(), "tt-local-spec-workspace-"));
});

afterEach(() => {
  rmSync(workspace, { recursive: true, force: true });
});

describe("local spec workspace", () => {
  const specPath = () => join(workspace, "tunedtensor.json");

  it("validates foundation execution settings before proposing a new spec", async () => {
    const { base_model: _model, ...behavior } = spec;
    await expect(prepareLocalSpecProject(workspace, {
      ...behavior, engine: "foundation", foundation: { nproc_per_node: 2 },
    })).rejects.toThrow(/nproc_per_node must be 1/);
    expect(existsSync(specPath())).toBe(false);
  });

  it("accepts the certified Nemotron Lightning model in local specs", async () => {
    const spec = {
      name: "Nemotron Worker",
      base_model: "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
      system_prompt: "Return one label.",
      guidelines: ["Return the requested label."],
      examples: [
        { input: "good", output: "positive" },
        { input: "bad", output: "negative" },
      ],
    };

    expect(Value.Check(LocalProjectSpecSchema, spec)).toBe(true);
  });
  it("accepts the certified Muse Glimmer model in local specs", async () => {
    const spec = {
      name: "Muse Glimmer Worker",
      base_model: "meta-models/Muse-Glimmer-30B",
      system_prompt: "Return one label.",
      guidelines: ["Return the requested label."],
      examples: [
        { input: "good", output: "positive" },
        { input: "bad", output: "negative" },
      ],
    };

    expect(Value.Check(LocalProjectSpecSchema, spec)).toBe(true);
  });
  it("applies the shared behavior-spec rules to new specs", async () => {
    await expect(prepareLocalSpecProject(workspace, {
      ...spec,
      examples: [{ input: "Classify this.", output: "positive" }],
    })).rejects.toThrow(/canonical tunedtensor\.json schema/i);
    await expect(prepareLocalSpecProject(workspace, {
      ...spec,
      examples: [
        { input: "Classify this.", output: "positive" },
        { input: "Classify this.", output: "negative" },
      ],
    })).rejects.toThrow(/repeats an input/i);
    expect(existsSync(specPath())).toBe(false);
  });

  it("targets tunedtensor.json in the workspace itself", async () => {
    await expect(prepareLocalSpecProject(workspace, spec)).resolves.toMatchObject({
      specPath: "./tunedtensor.json",
    });
  });

  it("refuses an existing spec and symlinked workspace roots", async () => {
    writeFileSync(specPath(), "{}");
    await expect(prepareLocalSpecProject(workspace, spec)).rejects.toThrow(
      /refusing to overwrite existing \.\/tunedtensor\.json/i,
    );

    const parent = mkdtempSync(join(tmpdir(), "tt-local-spec-link-"));
    const link = join(parent, "workspace-link");
    symlinkSync(workspace, link, "dir");
    try {
      await expect(prepareLocalSpecProject(link, spec)).rejects.toThrow(
        /real directory, not a symlink/i,
      );
    } finally {
      rmSync(parent, { recursive: true, force: true });
    }
  });

  it("revalidates tamperable action arguments and workspace identity", async () => {
    const prepared = await prepareLocalSpecProject(workspace, spec);
    await expect(createLocalSpecProject(
      workspace,
      { ...spec, base_model: "not a repo id" },
      prepared.workspaceFingerprint,
    )).rejects.toThrow(/canonical tunedtensor\.json schema/i);
    await expect(createLocalSpecProject(workspace, spec, "0".repeat(64))).rejects.toThrow(/workspace changed/i);
    expect(existsSync(specPath())).toBe(false);
  });

  it("creates private canonical JSON and never overwrites it", async () => {
    const prepared = await prepareLocalSpecProject(workspace, spec);
    await expect(createLocalSpecProject(workspace, spec, prepared.workspaceFingerprint)).resolves.toEqual({
      created: true,
      path: "./tunedtensor.json",
    });

    expect(JSON.parse(readFileSync(specPath(), "utf8"))).toEqual({ id: expect.stringMatching(/^[0-9a-f-]{36}$/), ...spec });
    expect(lstatSync(specPath()).mode & 0o777).toBe(0o600);
    await expect(createLocalSpecProject(workspace, spec, prepared.workspaceFingerprint)).rejects.toThrow(
      /refusing to overwrite/i,
    );
  });

  it("preserves a colliding spec created after approval", async () => {
    const prepared = await prepareLocalSpecProject(workspace, spec);

    await expect(createLocalSpecProject(workspace, spec, prepared.workspaceFingerprint, {
      writeFile: async (path, data, options) => {
        await writeFile(path, "external content", options);
        await writeFile(path, data, options);
      },
    })).rejects.toMatchObject({ outcome: "not_applied" });

    expect(readFileSync(specPath(), "utf8")).toBe("external content");
  });

  it("does not follow a workspace root replaced with a symlink before writing", async () => {
    const prepared = await prepareLocalSpecProject(workspace, spec);
    const movedWorkspace = `${workspace}-moved`;
    const outside = mkdtempSync(join(tmpdir(), "tt-local-spec-root-outside-"));

    try {
      await createLocalSpecProject(workspace, spec, prepared.workspaceFingerprint, {
        writeFile: async (path, data, options) => {
          await rename(workspace, movedWorkspace);
          symlinkSync(outside, workspace, "dir");
          await writeFile(path, data, options);
        },
      });

      expect(existsSync(join(outside, "tunedtensor.json"))).toBe(false);
      expect(existsSync(join(movedWorkspace, "tunedtensor.json"))).toBe(true);
    } finally {
      rmSync(workspace, { recursive: true, force: true });
      rmSync(movedWorkspace, { recursive: true, force: true });
      rmSync(outside, { recursive: true, force: true });
    }
  });

  it("keeps an ambiguous written spec and reports an unknown outcome", async () => {
    const prepared = await prepareLocalSpecProject(workspace, spec);

    await expect(createLocalSpecProject(workspace, spec, prepared.workspaceFingerprint, {
      writeFile: async (path, data, options) => {
        await writeFile(path, data, options);
        throw new Error("injected post-write failure");
      },
    })).rejects.toMatchObject({ outcome: "unknown" });

    expect(JSON.parse(readFileSync(specPath(), "utf8"))).toEqual({ id: expect.any(String), ...spec });
  });
});
