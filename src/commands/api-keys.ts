import { Command } from "commander";
import chalk from "chalk";
import { get, post, del, type ClientOpts } from "../client.js";
import {
  printTable,
  printSuccess,
  printJson,
  printWarning,
  isJsonMode,
  formatDate,
  shortId,
} from "../output.js";

interface ApiKey {
  id: string;
  name: string;
  key_prefix: string;
  last_used_at: string | null;
  created_at: string;
  revoked_at: string | null;
  key?: string; // only present on creation
}

export function registerApiKeysCommands(parent: Command) {
  const apiKeys = parent
    .command("api-keys")
    .description("Manage API keys");

  apiKeys
    .command("list")
    .description("List API keys")
    .action(async () => {
      const opts = parent.opts() as ClientOpts;
      const { data } = await get<ApiKey[]>("/api-keys", undefined, opts);

      if (isJsonMode()) return printJson(data);

      printTable(
        ["ID", "Name", "Prefix", "Last Used", "Created", "Revoked"],
        data.map((k) => [
          shortId(k.id),
          k.name,
          k.key_prefix + "...",
          formatDate(k.last_used_at),
          formatDate(k.created_at),
          k.revoked_at ? formatDate(k.revoked_at) : "—",
        ]),
      );
    });

  apiKeys
    .command("create")
    .description("Create a new API key")
    .requiredOption("-n, --name <name>", "Name for the key")
    .action(async (cmdOpts) => {
      const opts = parent.opts() as ClientOpts;
      const { data } = await post<ApiKey>(
        "/api-keys",
        { name: cmdOpts.name },
        opts,
      );

      if (isJsonMode()) return printJson(data);

      printSuccess(`API key created: ${data.name}`);
      console.log();
      console.log(chalk.bold("Your API key (shown once only):"));
      console.log();
      console.log(`  ${chalk.green(data.key)}`);
      console.log();
      printWarning("Store this key securely — it cannot be retrieved later.");
    });

  apiKeys
    .command("revoke")
    .description("Revoke an API key")
    .argument("<id>", "API key ID")
    .action(async (id: string) => {
      const opts = parent.opts() as ClientOpts;
      await del(`/api-keys/${id}`, opts);

      if (isJsonMode()) return printJson({ id, revoked: true });
      printSuccess(`API key revoked: ${shortId(id)}`);
    });
}
