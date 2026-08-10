import { defineConfig } from "tsup";
import packageJson from "./package.json";

export default defineConfig({
  entry: {
    index: "src/index.ts",
    "local-runtime": "src/local-runtime/index.ts",
  },
  format: ["esm"],
  target: "node22",
  clean: true,
  sourcemap: true,
  define: {
    __TT_VERSION__: JSON.stringify(packageJson.version),
  },
  banner: {
    js: "#!/usr/bin/env node",
  },
});
