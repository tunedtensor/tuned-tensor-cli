import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    globals: true,
    restoreMocks: true,
    exclude: ["test/local-runtime/**", "node_modules/**", "dist/**"],
  },
});
