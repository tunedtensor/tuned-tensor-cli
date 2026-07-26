import { runCli } from "./cli.js";
import { reportError } from "./output.js";

declare const __TT_VERSION__: string;

runCli(__TT_VERSION__).catch((err) => {
  reportError(err);
  process.exit(1);
});
