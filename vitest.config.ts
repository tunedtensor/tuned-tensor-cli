import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    globals: true,
    restoreMocks: true,
    // Native Windows workspace transactions start the system PowerShell host.
    testTimeout: process.platform === "win32" ? 30_000 : 5_000,
    hookTimeout: process.platform === "win32" ? 30_000 : 10_000,
    maxWorkers: process.platform === "win32" ? 4 : undefined,
    exclude: ["test/local-runtime/**", "node_modules/**", "dist/**"],
  },
});
