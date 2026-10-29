import assert from "node:assert/strict";
import { execFile, execFileSync } from "node:child_process";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { packageFixture } from "../../scripts/npm-package-fixture.mjs";

const runner = fileURLToPath(new URL("../../scripts/run-npm-windows.mjs", import.meta.url));
const run = (args: string[]) => new Promise<{ code: number; output: string }>((resolve, reject) => {
  execFile(process.execPath, [runner, ...args], { timeout: 10_000 }, (error, stdout, stderr) => {
    if (error && typeof error.code !== "number") return reject(error);
    resolve({ code: typeof error?.code === "number" ? error.code : 0, output: stdout + stderr });
  });
});
test("npm supervision preserves literal arguments, diagnostics and the child exit code", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-npm-supervisor-"));
  try {
    const cli = join(root, "fake npm.mjs");
    await writeFile(cli, 'console.log(JSON.stringify(process.argv.slice(2))); console.error("diagnostic"); process.exitCode = 7;');
    const result = await run(["--npm-cli", cli, "--timeout-ms", "5000", "--", "prefix with spaces & punctuation", "--ignore-scripts"]);
    assert.equal(result.code, 7);
    assert.match(result.output, /\["prefix with spaces & punctuation","--ignore-scripts"\]/);
    assert.match(result.output, /diagnostic/);
  } finally { await rm(root, { recursive: true, force: true }); }
});
test("npm supervision emits a heartbeat and bounds an unresponsive install", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-npm-timeout-"));
  try {
    const cli = join(root, "fake npm.mjs");
    await writeFile(cli, "setInterval(() => {}, 1000);");
    const result = await run(["--npm-cli", cli, "--timeout-ms", "400", "--heartbeat-ms", "50", "--", "install"]);
    assert.equal(result.code, 124);
    assert.match(result.output, /still running/);
    assert.match(result.output, /timed out/);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test("registry-shaped installs serve the exact packed artifact and redirect public dependencies", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-npm-registry-"));
  let registry: Awaited<ReturnType<typeof packageFixture>> | undefined;
  try {
    await mkdir(join(root, "package"));
    await writeFile(join(root, "package", "package.json"), JSON.stringify({ name: "@tuned-tensor/fixture", version: "1.2.3", dependencies: { chalk: "6.0.0" } }));
    const archive = join(root, "packed fixture.tgz");
    const tar = process.platform === "win32" ? join(process.env.SystemRoot ?? "C:\\Windows", "System32", "tar.exe") : "tar";
    execFileSync(tar, ["-czf", archive, "-C", root, "package"], { windowsHide: true });
    registry = await packageFixture(archive);
    const manifest = await (await fetch(registry.base + "/@tuned-tensor%2ffixture")).json() as any;
    assert.equal(manifest.versions["1.2.3"].dependencies.chalk, "6.0.0");
    const bytes = await (await fetch(manifest.versions["1.2.3"].dist.tarball)).arrayBuffer();
    assert.equal("sha512-" + createHash("sha512").update(Buffer.from(bytes)).digest("base64"), registry.integrity);
    const dependency = await fetch(registry.base + "/chalk?write=true", { redirect: "manual" });
    assert.equal(dependency.status, 302);
    assert.equal(dependency.headers.get("location"), "https://registry.npmjs.org/chalk?write=true");
    assert.equal((await fetch(registry.base, { method: "POST" })).status, 405);
  } finally { registry?.close(); await rm(root, { recursive: true, force: true }); }
});
