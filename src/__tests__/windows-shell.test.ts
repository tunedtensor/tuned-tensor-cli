import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { parseSlashCommand, routeShellCommand, tokenizeShellInput } from "../shell.js";
import { inspectLocalSpec } from "../spec-workspace.js";

describe("Windows conversational paths", () => {
  it("preserves drive paths, relative paths, UNC paths, and trailing separators", () => {
    expect(tokenizeShellInput('validate C:\\work\\tunedtensor.json', "win32"))
      .toEqual(["validate", "C:\\work\\tunedtensor.json"]);
    expect(tokenizeShellInput('spec view .\\tunedtensor.json', "win32"))
      .toEqual(["spec", "view", ".\\tunedtensor.json"]);
    expect(tokenizeShellInput('/cd "C:\\work files\\"', "win32"))
      .toEqual(["/cd", "C:\\work files\\"]);
    expect(tokenizeShellInput('/cd \\\\server\\share\\project', "win32"))
      .toEqual(["/cd", "\\\\server\\share\\project"]);
    expect(tokenizeShellInput("/cd 'C:\\work files'", "win32"))
      .toEqual(["/cd", "C:\\work files"]);
  });

  it("keeps POSIX escaping and rejects shell execution on both platforms", () => {
    expect(tokenizeShellInput('validate my\\ spec.json', "linux")).toEqual(["validate", "my spec.json"]);
    for (const platform of ["linux", "win32"] as const) {
      for (const input of ['runs list | whoami', 'runs list & whoami', 'runs get $(whoami)', 'runs get `whoami`']) {
        expect(() => tokenizeShellInput(input, platform)).toThrow(/operator/);
      }
    }
  });

  it.skipIf(process.platform !== "win32")("routes native paths through command and session parsers", () => {
    expect(routeShellCommand('validate "C:\\work files\\tunedtensor.json"')?.args)
      .toEqual(["validate", "C:\\work files\\tunedtensor.json"]);
    expect(parseSlashCommand('/cd "C:\\work files\\"'))
      .toEqual({ name: "cd", args: ["C:\\work files\\"] });
  });

  it("inspects a PowerShell-style relative spec while still refusing traversal", async () => {
    const root = await mkdtemp(join(tmpdir(), "tt-windows-spec-"));
    try {
      const source = await readFile("examples/local-runtime/smoke-spec.json", "utf8");
      await writeFile(join(root, "tunedtensor.json"), source);
      const result = await inspectLocalSpec(root, ".\\tunedtensor.json");
      expect(result.document).toEqual(JSON.parse(source));
      for (const path of ["..\\tunedtensor.json", ".\\..\\tunedtensor.json", ".\\.\\tunedtensor.json"]) {
        await expect(inspectLocalSpec(root, path)).rejects.toThrow(/traversal/);
      }
    } finally { await rm(root, { recursive: true, force: true }); }
  });
});
