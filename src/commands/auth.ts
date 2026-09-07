import { Command } from "commander";
import { createInterface } from "node:readline/promises";
import { stdin, stdout } from "node:process";
import { Writable } from "node:stream";
import {
  readConfig,
  updateConfig,
  writeConfig,
  getApiKey,
  getBaseUrl,
  validateAccessToken,
} from "../config.js";
import {
  printSuccess,
  printDetail,
  printWarning,
  isJsonMode,
  printJson,
} from "../output.js";
import chalk from "chalk";

export async function promptForApiKey(
  input: NodeJS.ReadableStream = stdin,
  output: NodeJS.WritableStream = stdout,
): Promise<string> {
  if (
    (input as { isTTY?: boolean }).isTTY !== true
    || (output as { isTTY?: boolean }).isTTY !== true
  ) {
    throw new Error(
      "A TT access token is required in non-interactive mode. Pass it to `tt auth login <key>` or set TUNED_TENSOR_API_KEY.",
    );
  }

  let muted = false;
  const maskedOutput = new Writable({
    write(chunk, _encoding, callback) {
      if (!muted) output.write(chunk);
      callback();
    },
  });
  const rl = createInterface({
    input,
    output: maskedOutput,
    terminal: true,
  });

  try {
    const pending = rl.question("Enter your TT access token (tt_...): ");
    muted = true;
    return await pending;
  } finally {
    muted = false;
    output.write("\n");
    rl.close();
  }
}

export function registerAuthCommands(parent: Command) {
  const auth = parent.command("auth").description("Manage the TT access token for managed inference and cloud operations");

  auth
    .command("login")
    .description("Save a TT access token for managed inference and cloud operations")
    .argument("[key]", "TT access token (tt_...). If omitted, you will be prompted.")
    .action(async (key?: string) => {
      let apiKey = key;

      if (!apiKey) {
        if (isJsonMode()) {
          throw new Error(
            "A TT access token argument is required in JSON mode. Use `tt --json auth login <key>`.",
          );
        }

        const settingsUrl = `${getBaseUrl(parent.optsWithGlobals()).replace(/\/$/, "")}/dashboard/settings`;
        console.log();
        console.log(
          `  To get your TT access token, go to ${chalk.bold("Settings → API Keys")} in the Tuned Tensor dashboard:`,
        );
        console.log(`  ${chalk.hex("#A78BFA").underline(settingsUrl)}`);
        console.log();

        apiKey = await promptForApiKey();
      }

      apiKey = validateAccessToken(apiKey);

      updateConfig({ api_key: apiKey, base_url: getBaseUrl(parent.optsWithGlobals()) });
      if (isJsonMode()) {
        printJson({
          authenticated: true,
          key_prefix: `${apiKey.slice(0, 8)}...`,
          base_url: getBaseUrl(parent.optsWithGlobals()),
        });
        return;
      }
      printSuccess(
        `TT access token stored (${apiKey.slice(0, 8)}...).`,
      );
    });

  auth
    .command("logout")
    .description("Remove the stored TT access token")
    .action(() => {
      const current = readConfig();
      const { api_key: _removed, ...rest } = current;
      writeConfig(rest);
      if (isJsonMode()) {
        printJson({ authenticated: false });
        return;
      }
      printSuccess("Stored TT access token removed.");
    });

  auth
    .command("status")
    .description("Show current authentication status")
    .action(() => {
      const opts = parent.optsWithGlobals();
      const apiKey = getApiKey(opts);
      const baseUrl = getBaseUrl(opts);

      if (isJsonMode()) {
        printJson({
          authenticated: !!apiKey,
          key_prefix: apiKey ? apiKey.slice(0, 8) + "..." : null,
          base_url: baseUrl,
        });
        return;
      }

      if (apiKey) {
        printDetail([
          ["Authenticated", "Yes"],
          ["Access token", apiKey.slice(0, 8) + "..."],
          ["Base URL", baseUrl],
        ]);
      } else {
        printWarning(
          "No TT access token configured. Run `tt auth login` for managed inference or cloud access. Local commands need no token.",
        );
        printDetail([["Base URL", baseUrl]]);
      }
    });
}
