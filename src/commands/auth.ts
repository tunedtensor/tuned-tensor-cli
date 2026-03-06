import { Command } from "commander";
import { createInterface } from "node:readline/promises";
import { stdin, stdout } from "node:process";
import {
  readConfig,
  updateConfig,
  clearConfig,
  getApiKey,
  getBaseUrl,
} from "../config.js";
import {
  printSuccess,
  printDetail,
  printError,
  printWarning,
  isJsonMode,
  printJson,
} from "../output.js";

export function registerAuthCommands(parent: Command) {
  const auth = parent.command("auth").description("Manage authentication");

  auth
    .command("login")
    .description("Store an API key for authentication")
    .argument("[key]", "API key (tt_...). If omitted, you will be prompted.")
    .action(async (key?: string) => {
      let apiKey = key;

      if (!apiKey) {
        const rl = createInterface({ input: stdin, output: stdout });
        apiKey = await rl.question("Enter your API key (tt_...): ");
        rl.close();
      }

      apiKey = apiKey.trim();

      if (!apiKey.startsWith("tt_") || apiKey.length !== 51) {
        printError(
          "Invalid API key format. Keys start with tt_ and are 51 characters long.",
        );
        process.exit(1);
      }

      updateConfig({ api_key: apiKey });
      printSuccess(
        `API key stored (${apiKey.slice(0, 8)}...). You're ready to go.`,
      );
    });

  auth
    .command("logout")
    .description("Remove stored credentials")
    .action(() => {
      clearConfig();
      printSuccess("Credentials removed.");
    });

  auth
    .command("status")
    .description("Show current authentication status")
    .action(() => {
      const opts = parent.opts();
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
          ["API Key", apiKey.slice(0, 8) + "..."],
          ["Base URL", baseUrl],
        ]);
      } else {
        printWarning(
          "Not authenticated. Run `tt auth login` to store an API key.",
        );
        printDetail([["Base URL", baseUrl]]);
      }
    });
}
