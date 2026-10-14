import { afterEach, describe, expect, it, vi } from "vitest";
import { withProgressHeartbeat } from "../local-runtime/run-reporter.js";

afterEach(() => vi.useRealTimers());

describe("long model verification feedback", () => {
  it.each([false, true])("reports while pending and stops after settlement (failure=%s)", async (fail) => {
    vi.useFakeTimers();
    const onEvent = vi.fn();
    let finish!: () => void;
    const work = new Promise<void>((resolve, reject) => {
      finish = () => fail ? reject(new Error("checksum mismatch")) : resolve();
    });
    const result = withProgressHeartbeat({ onEvent }, {
      stage: "model_prefetch", status: "running", message: "Verifying cached model files.",
    }, () => work);
    const settled = result.catch((error: Error) => error.message);
    expect(onEvent).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(30_000);
    expect(onEvent).toHaveBeenCalledTimes(3);
    expect(onEvent.mock.lastCall?.[0].message).toContain("Still working: Verifying");
    finish();
    expect(await settled).toBe(fail ? "checksum mismatch" : undefined);
    await vi.advanceTimersByTimeAsync(30_000);
    expect(onEvent).toHaveBeenCalledTimes(3);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("does not let a failing reporter interrupt verification", async () => {
    await expect(withProgressHeartbeat({ onEvent: async () => { throw new Error("closed output"); } }, {
      stage: "model_prefetch", status: "running", message: "Verifying.",
    }, async () => "verified")).resolves.toBe("verified");
  });
});
