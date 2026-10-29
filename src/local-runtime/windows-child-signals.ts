import { spawn, type ChildProcess } from "node:child_process";
import { join } from "node:path";

/** Give a Windows CLI time to clean up before stopping its owned process tree. */
export function windowsChildSignals(child: ChildProcess) {
  let timer: NodeJS.Timeout | undefined;
  const dispose = () => { if (timer) clearTimeout(timer); };
  child.once("close", dispose);
  return {
    dispose,
    forward(signal: NodeJS.Signals): boolean {
      if (process.platform !== "win32" || !child.connected) return false;
      child.send({ ttSignal: signal }, () => {});
      timer ??= setTimeout(() => {
        if (child.pid) {
          spawn(join(process.env.SystemRoot ?? "C:\\Windows", "System32", "taskkill.exe"),
            ["/PID", String(child.pid), "/T", "/F"], { windowsHide: true, stdio: "ignore" }).on("error", () => child.kill());
        }
      }, 5_000);
      return true;
    },
  };
}
