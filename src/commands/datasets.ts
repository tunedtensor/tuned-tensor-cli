import { Command } from "commander";
import { existsSync } from "node:fs";
import { get, del, upload, type ClientOpts } from "../client.js";
import {
  printTable,
  printDetail,
  printSuccess,
  printJson,
  printError,
  isJsonMode,
  formatDate,
  formatStatus,
  shortId,
  truncate,
} from "../output.js";

interface Dataset {
  id: string;
  name: string;
  description: string | null;
  storage_path: string;
  file_size_bytes: number;
  row_count: number;
  format: string;
  status: string;
  validation_errors: string[] | null;
  created_at: string;
  updated_at: string;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function registerDatasetsCommands(parent: Command) {
  const datasets = parent
    .command("datasets")
    .description("Manage datasets");

  datasets
    .command("list")
    .description("List datasets")
    .option("-p, --page <n>", "Page number", "1")
    .option("--per-page <n>", "Results per page", "20")
    .action(async (cmdOpts) => {
      const opts = parent.opts() as ClientOpts;
      const { data, meta } = await get<Dataset[]>(
        "/datasets",
        { page: cmdOpts.page, per_page: cmdOpts.perPage },
        opts,
      );

      if (isJsonMode()) return printJson({ data, meta });

      printTable(
        ["ID", "Name", "Rows", "Size", "Status", "Created"],
        data.map((d) => [
          shortId(d.id),
          truncate(d.name, 30),
          String(d.row_count),
          formatBytes(d.file_size_bytes),
          formatStatus(d.status),
          formatDate(d.created_at),
        ]),
        meta,
      );
    });

  datasets
    .command("get")
    .description("Show dataset details")
    .argument("<id>", "Dataset ID")
    .action(async (id: string) => {
      const opts = parent.opts() as ClientOpts;
      const { data } = await get<Dataset>(`/datasets/${id}`, undefined, opts);

      if (isJsonMode()) return printJson(data);

      printDetail([
        ["ID", data.id],
        ["Name", data.name],
        ["Description", data.description ?? undefined],
        ["Format", data.format],
        ["Rows", String(data.row_count)],
        ["Size", formatBytes(data.file_size_bytes)],
        ["Status", formatStatus(data.status)],
        ["Created", formatDate(data.created_at)],
        ["Updated", formatDate(data.updated_at)],
      ]);

      if (data.validation_errors?.length) {
        console.log("\nValidation Errors:");
        for (const err of data.validation_errors) {
          console.log(`  - ${err}`);
        }
      }
    });

  datasets
    .command("upload")
    .description("Upload a JSONL dataset file")
    .argument("<file>", "Path to JSONL file")
    .option("-n, --name <name>", "Dataset name (defaults to filename)")
    .option("-d, --description <desc>", "Dataset description")
    .action(async (file: string, cmdOpts) => {
      const opts = parent.opts() as ClientOpts;

      if (!existsSync(file)) {
        printError(`File not found: ${file}`);
        process.exit(1);
      }

      const fields: Record<string, string> = {
        name: cmdOpts.name || file.split("/").pop()!.replace(/\.jsonl?$/, ""),
      };
      if (cmdOpts.description) fields.description = cmdOpts.description;

      const { data } = await upload<Dataset>(
        "/datasets",
        file,
        fields,
        opts,
      );

      if (isJsonMode()) return printJson(data);
      printSuccess(
        `Dataset uploaded: ${data.name} (${shortId(data.id)}) — ${data.row_count} rows`,
      );
    });

  datasets
    .command("delete")
    .description("Delete a dataset")
    .argument("<id>", "Dataset ID")
    .action(async (id: string) => {
      const opts = parent.opts() as ClientOpts;
      await del(`/datasets/${id}`, opts);

      if (isJsonMode()) return printJson({ id, deleted: true });
      printSuccess(`Dataset deleted: ${shortId(id)}`);
    });
}
