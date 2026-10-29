import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { fileUri } from "../local-runtime/artifacts.js";
import { localModelArtifactPath } from "../local-runtime/model-registry.js";

describe("local model paths and file URIs", () => {
  it("round-trips native paths containing spaces, Unicode and URI punctuation", () => {
    const path = resolve(tmpdir(), "model café #1 100%", "adapter.safetensors");
    const uri = fileUri(path);
    expect(new URL(uri).protocol).toBe("file:");
    expect(localModelArtifactPath(uri)).toBe(path);
    expect(localModelArtifactPath(path)).toBe(path);
    expect(() => localModelArtifactPath("https://example.com/model")).toThrow(/local path/);
  });
  it.runIf(process.platform === "win32")("accepts native drive paths and legacy Windows file URIs", () => {
    const path = join(tmpdir(), "model with spaces", "weights.bin");
    expect(localModelArtifactPath(path)).toBe(path);
    expect(localModelArtifactPath(`file://${path}`)).toBe(path);
    expect(localModelArtifactPath("\\\\server\\share\\weights.bin")).toBe("\\\\server\\share\\weights.bin");
  });
});
