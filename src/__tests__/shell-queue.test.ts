import { PassThrough } from "node:stream";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";
import { startInteractiveShell } from "../shell.js";

describe("interactive follow-up queue", () => {
  it("acknowledges follow-ups immediately and processes them in order after the action settles", async () => {
    const root = await mkdtemp(join(tmpdir(), "tt-shell-queue-"));
    const input = new PassThrough();
    const output = new PassThrough();
    let text = "";
    output.on("data", (chunk) => { text += String(chunk); });
    let finish!: () => void;
    const action = new Promise<void>((resolve) => { finish = resolve; });
    const handleLine = vi.fn(async (line: string) => {
      if (line === "download Muse") await action;
      return "continue" as const;
    });
    const running = startInteractiveShell({
      cwd: root, env: { HOME: root }, input, output, error: output, requireTTY: false,
      runner: async () => ({ exitCode: 0 }),
      agent: { busy: false, handleLine, interrupt: () => false },
      systemProbe: async () => ({ sampled_at: "now", cpu: { model: "test", cores: 1, load_average: [0, 0, 0] },
        memory: { used_bytes: 0, total_bytes: 1 }, gpus: [] }),
    });
    try {
      await vi.waitFor(() => expect(text).toContain("/help"));
      input.write("download Muse\n");
      await vi.waitFor(() => expect(handleLine).toHaveBeenCalledTimes(1));
      input.write("so we have muse already?\nhello\n");
      await vi.waitFor(() => expect(text.match(/Input queued/g)).toHaveLength(2));
      expect(handleLine).toHaveBeenCalledTimes(1);
      finish();
      await vi.waitFor(() => expect(handleLine).toHaveBeenCalledTimes(3));
      expect(handleLine.mock.calls.map(([line]) => line)).toEqual([
        "download Muse", "so we have muse already?", "hello",
      ]);
    } finally {
      finish();
      input.write("/exit\n");
      await running;
      input.destroy();
      output.destroy();
      await rm(root, { recursive: true, force: true });
    }
  });
});
