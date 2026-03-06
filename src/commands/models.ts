import { Command } from "commander";
import { get, del, type ClientOpts } from "../client.js";
import {
  printTable,
  printDetail,
  printSuccess,
  printJson,
  isJsonMode,
  formatDate,
  shortId,
  truncate,
} from "../output.js";

interface Model {
  id: string;
  name: string;
  provider: string;
  provider_model_id: string;
  base_model: string;
  description: string | null;
  created_at: string;
}

export function registerModelsCommands(parent: Command) {
  const models = parent.command("models").description("Manage fine-tuned models");

  models
    .command("list")
    .description("List fine-tuned models")
    .option("-p, --page <n>", "Page number", "1")
    .option("--per-page <n>", "Results per page", "20")
    .action(async (cmdOpts) => {
      const opts = parent.opts() as ClientOpts;
      const { data, meta } = await get<Model[]>(
        "/models",
        { page: cmdOpts.page, per_page: cmdOpts.perPage },
        opts,
      );

      if (isJsonMode()) return printJson({ data, meta });

      printTable(
        ["ID", "Name", "Base Model", "Provider", "Created"],
        data.map((m) => [
          shortId(m.id),
          truncate(m.name, 30),
          m.base_model.split("/").pop() || m.base_model,
          m.provider,
          formatDate(m.created_at),
        ]),
        meta,
      );
    });

  models
    .command("get")
    .description("Show model details")
    .argument("<id>", "Model ID")
    .action(async (id: string) => {
      const opts = parent.opts() as ClientOpts;
      const { data } = await get<Model>(`/models/${id}`, undefined, opts);

      if (isJsonMode()) return printJson(data);

      printDetail([
        ["ID", data.id],
        ["Name", data.name],
        ["Base Model", data.base_model],
        ["Provider", data.provider],
        ["Provider Model", data.provider_model_id],
        ["Description", data.description ?? undefined],
        ["Created", formatDate(data.created_at)],
      ]);
    });

  models
    .command("delete")
    .description("Delete a model")
    .argument("<id>", "Model ID")
    .action(async (id: string) => {
      const opts = parent.opts() as ClientOpts;
      await del(`/models/${id}`, opts);

      if (isJsonMode()) return printJson({ id, deleted: true });
      printSuccess(`Model deleted: ${shortId(id)}`);
    });
}
