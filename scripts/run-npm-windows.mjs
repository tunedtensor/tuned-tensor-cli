import { spawn } from "node:child_process";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { packageFixture } from "./npm-package-fixture.mjs";

const { values, positionals } = parseArgs({ allowPositionals: true, options: {
  "npm-cli": { type: "string" },
  "timeout-ms": { type: "string", default: "1500000" },
  "heartbeat-ms": { type: "string", default: "30000" },
  label: { type: "string", default: "npm" },
  package: { type: "string" },
} });
const timeout = Number(values["timeout-ms"]);
const heartbeat = Number(values["heartbeat-ms"]);
if (!values["npm-cli"] || !Number.isSafeInteger(timeout) || timeout < 1
  || !Number.isSafeInteger(heartbeat) || heartbeat < 1) throw new Error("Expected npm CLI path and positive timing limits.");

const started = Date.now();
const fixture = values.package ? await packageFixture(values.package) : undefined;
const npmArgs = fixture ? [...positionals, fixture.spec, "--registry", fixture.base, "--no-update-notifier"] : positionals;
let forcedExit;
let forceTimer;
const elapsed = () => `${Math.round((Date.now() - started) / 1000)}s`;
const child = spawn(process.execPath, [values["npm-cli"], ...npmArgs], {
  stdio: ["ignore", "pipe", "pipe"], windowsHide: true,
});
// Keep npm's diagnostic stderr visible without PowerShell 5.1 interpreting it
// as a NativeCommandError. Exit status, rather than the output stream, decides success.
child.stdout.pipe(process.stdout);
child.stderr.pipe(process.stdout);
const stop = (code) => {
  if (forcedExit !== undefined) return;
  forcedExit = code;
  if (child.pid && process.platform === "win32") {
    const kill = spawn(join(process.env.SystemRoot ?? "C:\\Windows", "System32", "taskkill.exe"),
      ["/PID", String(child.pid), "/T", "/F"], { windowsHide: true, stdio: "ignore" });
    kill.on("error", () => child.kill("SIGKILL"));
    kill.on("close", result => { if (result !== 0) child.kill("SIGKILL"); });
  } else child.kill("SIGKILL");
  forceTimer = setTimeout(() => { child.kill("SIGKILL"); process.exit(code); }, 5000);
  forceTimer.unref();
};
const progress = setInterval(() => console.log(`[${values.label}] still running; elapsed ${elapsed()}. npm timing logs are retained.`), heartbeat);
const deadline = setTimeout(() => {
  console.log(`[${values.label}] timed out after ${timeout}ms; stopping the owned npm process tree.`);
  stop(124);
}, timeout);
process.once("SIGTERM", () => stop(143));
process.once("SIGINT", () => stop(130));
child.once("error", error => { console.log(`[${values.label}] could not start: ${error.message}`); forcedExit = 1; });
child.once("close", code => {
  fixture?.close();
  clearInterval(progress);
  clearTimeout(deadline);
  if (forceTimer) clearTimeout(forceTimer);
  process.exitCode = forcedExit ?? code ?? 1;
  console.log(`[${values.label}] finished in ${elapsed()}; exit ${process.exitCode}.`);
});
