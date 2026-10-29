import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile, stat } from "node:fs/promises";
import { createServer } from "node:http";
import { join } from "node:path";

/** Serve the tested tarball using the metadata path of a real npm publication.
 * Local-file installs otherwise unpack a large bundled dependency tree merely
 * to inspect its manifest. Dependencies still come from the public npm registry.
 */
export async function packageFixture(archive) {
  const tar = process.platform === "win32" ? join(process.env.SystemRoot ?? "C:\\Windows", "System32", "tar.exe") : "tar";
  const source = await new Promise((resolve, reject) => {
    execFile(tar, ["-xOf", archive, "package/package.json"], { encoding: "utf8", timeout: 30_000, maxBuffer: 2_000_000, windowsHide: true },
      (error, stdout) => error ? reject(error) : resolve(stdout));
  });
  const manifest = JSON.parse(source);
  if (typeof manifest.name !== "string" || typeof manifest.version !== "string") throw new Error("Packed npm manifest must contain a name and version.");
  const integrity = "sha512-" + createHash("sha512").update(await readFile(archive)).digest("base64");
  const size = (await stat(archive)).size;
  let base;
  const server = createServer((request, response) => {
    if (!["GET", "HEAD"].includes(request.method)) { response.writeHead(405).end(); return; }
    let url;
    let path;
    try { url = new URL(request.url, base); path = decodeURIComponent(url.pathname); }
    catch { response.writeHead(400).end(); return; }
    if (path === "/_tt/package.tgz") {
      response.writeHead(200, { "content-type": "application/octet-stream", "content-length": size });
      if (request.method === "HEAD") { response.end(); return; }
      const input = createReadStream(archive);
      input.on("error", error => response.destroy(error));
      response.on("close", () => input.destroy());
      input.pipe(response);
      return;
    }
    if (path === "/" + manifest.name || path === "/" + manifest.name + "/" + manifest.version) {
      const version = { ...manifest, dist: { tarball: base + "/_tt/package.tgz", integrity } };
      const document = path.endsWith("/" + manifest.version) ? version
        : { name: manifest.name, "dist-tags": { latest: manifest.version }, versions: { [manifest.version]: version } };
      response.writeHead(200, { "content-type": "application/json" });
      response.end(request.method === "HEAD" ? undefined : JSON.stringify(document));
      return;
    }
    // This is a metadata redirect, not a proxy: no caller headers are forwarded.
    const official = new URL("https://registry.npmjs.org");
    official.pathname = url.pathname;
    official.search = url.search;
    response.writeHead(302, { location: official.href }).end();
  });
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  base = `http://127.0.0.1:${server.address().port}`;
  return { base, spec: `${manifest.name}@${manifest.version}`, integrity,
    close() { server.closeAllConnections(); server.close(); } };
}
