import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  existsSync,
  lstatSync,
  mkdtempSync,
  mkdirSync,
  readFileSync,
  rmSync,
  symlinkSync,
} from "node:fs";
import { mkdir as mkdirAsync, rename, writeFile } from "node:fs/promises";
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
  it.each(["../escape", "/tmp/escape", "nested/spec", "two words", ".", ".."])(
    "rejects unsafe or ambiguous folder name %s",
    async (directory) => {
      await expect(prepareLocalSpecProject(workspace, directory, spec)).rejects.toThrow(
        /one portable folder name/i,
      );
    },
  );

  it("rejects specs whose examples violate their constraints", async () => {
    await expect(prepareLocalSpecProject(workspace, "too-small-spec", {
      ...spec,
      examples: [{ input: "Classify this.", output: "positive" }],
    })).rejects.toThrow(/canonical tunedtensor\.json schema/i);
    await expect(prepareLocalSpecProject(workspace, "invalid-spec", {
      ...spec,
      constraints: ["Never mention secret"],
      examples: [
        { input: "Classify this.", output: "secret" },
        { input: "Classify that.", output: "another secret" },
      ],
    })).rejects.toThrow(/violates/i);
    expect(existsSync(join(workspace, "invalid-spec"))).toBe(false);
  });

  it("refuses existing targets and symlinked workspace roots", async () => {
    mkdirSync(join(workspace, "existing"));
    await expect(prepareLocalSpecProject(workspace, "existing", spec)).rejects.toThrow(
      /refusing to overwrite/i,
    );

    const parent = mkdtempSync(join(tmpdir(), "tt-local-spec-link-"));
    const link = join(parent, "workspace-link");
    symlinkSync(workspace, link, "dir");
    try {
      await expect(prepareLocalSpecProject(link, "new-spec", spec)).rejects.toThrow(
        /real directory, not a symlink/i,
      );
    } finally {
      rmSync(parent, { recursive: true, force: true });
    }
  });

  it("revalidates tamperable action arguments and workspace identity", async () => {
    const prepared = await prepareLocalSpecProject(workspace, "safe-spec", spec);
    await expect(createLocalSpecProject(
      workspace,
      "safe-spec",
      { ...spec, base_model: "unsupported/model" },
      prepared.workspaceFingerprint,
    )).rejects.toThrow(/canonical tunedtensor\.json schema/i);
    expect(existsSync(join(workspace, "safe-spec"))).toBe(false);

    await expect(createLocalSpecProject(
      workspace,
      "safe-spec",
      spec,
      "0".repeat(64),
    )).rejects.toThrow(/workspace changed/i);
    expect(existsSync(join(workspace, "safe-spec"))).toBe(false);
  });

  it("creates private canonical JSON and never overwrites it", async () => {
    const prepared = await prepareLocalSpecProject(workspace, "safe-spec", spec);
    await createLocalSpecProject(
      workspace,
      "safe-spec",
      spec,
      prepared.workspaceFingerprint,
    );

    const target = join(workspace, "safe-spec");
    const specPath = join(target, "tunedtensor.json");
    expect(JSON.parse(readFileSync(specPath, "utf8"))).toEqual(spec);
    expect(lstatSync(target).mode & 0o777).toBe(0o700);
    expect(lstatSync(specPath).mode & 0o777).toBe(0o600);
    await expect(createLocalSpecProject(
      workspace,
      "safe-spec",
      spec,
      prepared.workspaceFingerprint,
    )).rejects.toThrow(/refusing to overwrite/i);
  });

  it("preserves a colliding spec created after its directory", async () => {
    const prepared = await prepareLocalSpecProject(workspace, "collision-spec", spec);
    const target = join(workspace, "collision-spec");
    const specPath = join(target, "tunedtensor.json");

    await expect(createLocalSpecProject(
      workspace,
      "collision-spec",
      spec,
      prepared.workspaceFingerprint,
      {
        writeFile: async (path, data, options) => {
          await writeFile(path, "external content", options);
          await writeFile(path, data, options);
        },
      },
    )).rejects.toMatchObject({ outcome: "unknown" });

    expect(readFileSync(specPath, "utf8")).toBe("external content");
  });

  it("does not follow a destination replaced with a symlink after mkdir", async () => {
    const prepared = await prepareLocalSpecProject(workspace, "swapped-spec", spec);
    const target = join(workspace, "swapped-spec");
    const movedTarget = join(workspace, "moved-by-racer");
    const outside = mkdtempSync(join(tmpdir(), "tt-local-spec-outside-"));
    const outsideSpec = join(outside, "tunedtensor.json");

    try {
      await expect(createLocalSpecProject(
        workspace,
        "swapped-spec",
        spec,
        prepared.workspaceFingerprint,
        {
          writeFile: async (path, data, options) => {
            await rename(target, movedTarget);
            symlinkSync(outside, target, "dir");
            await writeFile(path, data, options);
          },
        },
      )).rejects.toMatchObject({ outcome: "unknown" });

      expect(existsSync(outsideSpec)).toBe(false);
    } finally {
      rmSync(outside, { recursive: true, force: true });
    }
  });

  it("does not follow a workspace root replaced with a symlink before mkdir", async () => {
    const prepared = await prepareLocalSpecProject(workspace, "root-swapped-spec", spec);
    const movedWorkspace = `${workspace}-moved`;
    const outside = mkdtempSync(join(tmpdir(), "tt-local-spec-root-outside-"));

    try {
      await expect(createLocalSpecProject(
        workspace,
        "root-swapped-spec",
        spec,
        prepared.workspaceFingerprint,
        {
          mkdir: async (path, options) => {
            await rename(workspace, movedWorkspace);
            symlinkSync(outside, workspace, "dir");
            await mkdirAsync(path, options);
          },
        },
      )).rejects.toMatchObject({ outcome: "unknown" });

      expect(existsSync(join(outside, "root-swapped-spec"))).toBe(false);
      expect(existsSync(join(movedWorkspace, "root-swapped-spec", "tunedtensor.json"))).toBe(true);
    } finally {
      rmSync(workspace, { recursive: true, force: true });
      rmSync(movedWorkspace, { recursive: true, force: true });
      rmSync(outside, { recursive: true, force: true });
    }
  });

  it("keeps an ambiguous written spec and reports an unknown outcome", async () => {
    const prepared = await prepareLocalSpecProject(workspace, "rollback-spec", spec);
    const target = join(workspace, "rollback-spec");

    await expect(createLocalSpecProject(
      workspace,
      "rollback-spec",
      spec,
      prepared.workspaceFingerprint,
      {
        writeFile: async (
          path: string,
          data: string,
          options: { encoding: "utf8"; flag: "wx"; mode: number },
        ) => {
          await writeFile(path, data, options);
          throw new Error("injected post-write failure");
        },
      },
    )).rejects.toMatchObject({ outcome: "unknown" });

    expect(existsSync(target)).toBe(true);
    expect(JSON.parse(readFileSync(join(target, "tunedtensor.json"), "utf8"))).toEqual(spec);
  });
});
