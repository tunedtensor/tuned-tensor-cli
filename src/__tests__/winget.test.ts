import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";

describe("release WinGet manifests", () => {
  it("binds the installer URL and checksum to the built package version", () => {
    const root = mkdtempSync(join(tmpdir(), "tt-winget-"));
    try {
      const { version } = JSON.parse(readFileSync("package.json", "utf8"));
      if (version.includes("-")) return; // Prereleases deliberately have no stable manifest.
      const name = `tuned-tensor-cli-${version}-windows-x64.exe`;
      const installer = join(root, name);
      writeFileSync(installer, "fixture installer bytes");
      const hash = createHash("sha256").update(readFileSync(installer)).digest("hex").toUpperCase();
      execFileSync(process.execPath, [resolve("scripts/generate-winget.mjs"), installer, root]);
      const manifest = readFileSync(join(root, "manifests", "t", "TunedTensor", "CLI", version, "TunedTensor.CLI.installer.yaml"), "utf8");
      expect(manifest).toContain(`InstallerSha256: ${hash}`);
      expect(manifest).toContain(`/releases/download/v${version}/${name}`);
      expect(manifest).toContain("InstallerType: inno\nScope: user");
      expect(manifest).toContain("ProductCode: TunedTensor.CLI_is1");
      expect(() => execFileSync(process.execPath, [resolve("scripts/generate-winget.mjs"), join(root, "wrong.exe"), root], { stdio: "pipe" })).toThrow();
    } finally { rmSync(root, { recursive: true, force: true }); }
  });
});
