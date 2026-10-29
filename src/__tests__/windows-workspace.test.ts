import { mkdtemp, readFile, rename, rm, symlink } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { WindowsWorkspace } from "../windows-workspace.js";

describe.skipIf(process.platform !== "win32")("native Windows workspace handles", () => {
  it("pins directories until close and writes UTF-8 without a BOM", async () => {
    const root = await mkdtemp(join(tmpdir(), "tt Windows café "));
    try {
      const windows = await WindowsWorkspace.open(root);
      try {
        await expect(rename(root, root + "-moved")).rejects.toThrow();
        await windows.create(root, "tunedtensor.json", '{"name":"café"}');
        await expect(windows.create(root, "tunedtensor.json", "overwrite")).rejects.toMatchObject({ code: "EEXIST" });
        expect(await windows.read(root, "tunedtensor.json", 200_000)).toBe('{"name":"café"}');
        const aclScript = String.raw`$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Console]::InputEncoding = New-Object Text.UTF8Encoding($false)
$path = [Console]::ReadLine()
$rules = @([IO.File]::GetAccessControl($path).GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier]) | ForEach-Object { @{ sid = $_.IdentityReference.Value; inherited = $_.IsInherited } })
@{ user = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value; rules = $rules } | ConvertTo-Json -Compress -Depth 4`;
        const acl = JSON.parse(execFileSync(join(process.env.SystemRoot ?? "C:\\Windows", "System32", "WindowsPowerShell", "v1.0", "powershell.exe"),
          ["-NoProfile", "-OutputFormat", "Text", "-EncodedCommand", Buffer.from(aclScript, "utf16le").toString("base64")],
          { input: join(root, "tunedtensor.json") + "\n", encoding: "utf8", windowsHide: true })) as { user: string; rules: unknown[] };
        expect(acl.rules).toEqual([{ sid: acl.user, inherited: false }]);
        await expect(windows.read(root, "tunedtensor.json", 2)).rejects.toThrow(/limit/);
        await expect(windows.read(root, "missing.json", 100)).rejects.toMatchObject({ code: "ENOENT" });
        await expect(windows.create(root, "../outside.json", "bad")).rejects.toThrow(/entry name/);
        const state = await windows.childDirectory(root, ".tuned-tensor");
        await windows.atomic(state, "revision.json", '{"status":"prepared"}');
        await windows.atomic(state, "revision.json", '{"status":"applied"}');
        expect(await readFile(join(state, "revision.json"), "utf8")).toBe('{"status":"applied"}');
      } finally { await windows.close(); }
      await rename(root, root + "-moved");
      await rename(root + "-moved", root);
    } finally { await rm(root, { recursive: true, force: true }); }
  });

  it("rejects unprivileged directory junctions and never writes through them", async () => {
    const root = await mkdtemp(join(tmpdir(), "tt Windows junction "));
    const outside = await mkdtemp(join(tmpdir(), "tt Windows outside "));
    try {
      await symlink(outside, join(root, "redirect"), "junction");
      await expect(WindowsWorkspace.open(join(root, "redirect"))).rejects.toThrow(/reparse/);
      const windows = await WindowsWorkspace.open(root);
      try { await expect(windows.childDirectory(root, "redirect")).rejects.toThrow(/reparse/); }
      finally { await windows.close(); }
      await expect(readFile(join(outside, "tunedtensor.json"))).rejects.toMatchObject({ code: "ENOENT" });
    } finally { await rm(root, { recursive: true, force: true }); await rm(outside, { recursive: true, force: true }); }
  });

  it("serializes writers and releases an abandoned transaction's lock on close", async () => {
    const root = await mkdtemp(join(tmpdir(), "tt Windows lock "));
    try {
      const first = await WindowsWorkspace.open(root);
      const second = await WindowsWorkspace.open(root);
      try {
        await first.lock(root, ".tunedtensor-spec.lock");
        await expect(second.lock(root, ".tunedtensor-spec.lock")).rejects.toThrow();
        await first.close();
        await second.lock(root, ".tunedtensor-spec.lock");
      } finally { await first.close(); await second.close(); }
    } finally { await rm(root, { recursive: true, force: true }); }
  });
});
