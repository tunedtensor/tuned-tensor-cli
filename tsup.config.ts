import { defineConfig } from "tsup";
import packageJson from "./package.json";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm"],
  target: "node20",
  clean: true,
  sourcemap: true,
  define: {
    __TT_VERSION__: JSON.stringify(packageJson.version),
  },
  banner: {
    js: "#!/usr/bin/env node",
  },
});
