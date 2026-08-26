import { afterEach, describe, expect, it } from "vitest";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  DEFAULT_ARTIFACT_ROOT,
  defaultCacheRoot,
  defaultStoreRoot,
  getAgentConfigDir,
  getConfigDir,
  getTunedTensorHome,
  pythonEnvironmentPath,
} from "../paths.js";

const originalHome = process.env.TUNED_TENSOR_HOME;
const originalLocalHome = process.env.TT_LOCAL_HOME;
const originalUserHome = process.env.HOME;

afterEach(() => {
  if (originalHome === undefined) delete process.env.TUNED_TENSOR_HOME;
  else process.env.TUNED_TENSOR_HOME = originalHome;
  if (originalLocalHome === undefined) delete process.env.TT_LOCAL_HOME;
  else process.env.TT_LOCAL_HOME = originalLocalHome;
  if (originalUserHome === undefined) delete process.env.HOME;
  else process.env.HOME = originalUserHome;
});

describe("tuned tensor paths", () => {
  it("keeps laptop state under TUNED_TENSOR_HOME", () => {
    const home = join(tmpdir(), `tt-paths-home-${process.pid}`);
    process.env.TUNED_TENSOR_HOME = home;
    delete process.env.TT_LOCAL_HOME;

    expect(getTunedTensorHome()).toBe(home);
    expect(getConfigDir()).toBe(home);
    expect(getAgentConfigDir()).toBe(join(home, "agent"));
    expect(defaultStoreRoot()).toBe(join(home, "store"));
    expect(defaultCacheRoot()).toBe(join(home, "cache"));
    expect(pythonEnvironmentPath("uv", "abc")).toBe(join(home, "cache", "uv", "abc"));
    expect(DEFAULT_ARTIFACT_ROOT).toBe(".tuned-tensor/artifacts");
  });

  it("defaults the laptop parent to ~/.tuned-tensor", () => {
    const userHome = join(tmpdir(), `tt-paths-user-${process.pid}`);
    delete process.env.TUNED_TENSOR_HOME;
    delete process.env.TT_LOCAL_HOME;
    process.env.HOME = userHome;

    expect(getTunedTensorHome()).toBe(join(userHome, ".tuned-tensor"));
    expect(defaultStoreRoot()).toBe(join(userHome, ".tuned-tensor", "store"));
  });

  it("lets TT_LOCAL_HOME override only the store", () => {
    const home = join(tmpdir(), `tt-paths-store-${process.pid}`);
    process.env.TUNED_TENSOR_HOME = home;
    process.env.TT_LOCAL_HOME = join(home, "custom-store");

    expect(getAgentConfigDir()).toBe(join(home, "agent"));
    expect(defaultStoreRoot()).toBe(join(home, "custom-store"));
  });

  it("reuses a legacy store when the new one does not exist", () => {
    const userHome = join(tmpdir(), `tt-paths-legacy-${process.pid}`);
    rmSync(userHome, { recursive: true, force: true });
    const legacy = join(userHome, ".tuned-tensor-local");
    mkdirSync(legacy, { recursive: true });
    writeFileSync(join(legacy, "marker"), "old");
    delete process.env.TUNED_TENSOR_HOME;
    delete process.env.TT_LOCAL_HOME;
    process.env.HOME = userHome;

    expect(defaultStoreRoot()).toBe(legacy);
    rmSync(userHome, { recursive: true, force: true });
  });
});
