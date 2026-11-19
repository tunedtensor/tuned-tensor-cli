import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    globals: true,
    restoreMocks: true,
    // Native Windows workspace transactions start the system PowerShell host,
    // which can take tens of seconds per test on loaded ARM64 runners.
    testTimeout: process.platform === "win32" ? 120_000 : 5_000,
    hookTimeout: process.platform === "win32" ? 60_000 : 10_000,
    maxWorkers: process.platform === "win32" ? 4 : undefined,
    exclude: ["test/local-runtime/**", "node_modules/**", "dist/**"],
  },
});
