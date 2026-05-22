import { Command } from "commander";
import { existsSync, mkdirSync, statSync, createWriteStream, unlinkSync } from "node:fs";
import { dirname, join } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { get, del, type ClientOpts } from "../client.js";
import { resolveModelId } from "../resolve.js";
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

interface ModelDownload {
  url: string;
  filename: string;
  expires_at: string;
}

function resolveOutputPath(output: string | undefined, filename: string): string {
  if (!output) return filename;
  if (existsSync(output) && statSync(output).isDirectory()) {
    return join(output, filename);
  }
  return output;
}

async function downloadUrlToFile(url: string, outputPath: string): Promise<number | null> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Download failed: ${res.status} ${res.statusText}`);
  }
  if (!res.body) {
    throw new Error("Download failed: response body was empty");
  }

  mkdirSync(dirname(outputPath), { recursive: true });
  const file = createWriteStream(outputPath);

  try {
    await pipeline(Readable.fromWeb(res.body), file);
  } catch (err) {
    try {
      unlinkSync(outputPath);
    } catch {
      // Best effort cleanup for partially written downloads.
    }
    throw err;
  }

  const length = res.headers.get("content-length");
  return length ? Number(length) : null;
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
    .argument("<id>", "Model ID (full UUID or 4+ char prefix)")
    .action(async (id: string) => {
      const opts = parent.opts() as ClientOpts;
      const fullId = await resolveModelId(id, opts);
      const { data } = await get<Model>(`/models/${fullId}`, undefined, opts);

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
    .command("download")
    .description("Download a fine-tuned model artifact")
    .argument("<id>", "Model ID (full UUID or 4+ char prefix)")
    .option("-o, --output <path>", "Output file or directory")
    .option("-f, --force", "Overwrite the output file if it already exists")
    .action(async (id: string, cmdOpts) => {
      const opts = parent.opts() as ClientOpts;
      const fullId = await resolveModelId(id, opts);
      const { data } = await get<ModelDownload>(
        `/models/${fullId}/download`,
        undefined,
        opts,
      );

      const outputPath = resolveOutputPath(cmdOpts.output, data.filename);
      if (existsSync(outputPath) && !cmdOpts.force) {
        throw new Error(`Output file already exists: ${outputPath}. Use --force to overwrite.`);
      }

      const bytes = await downloadUrlToFile(data.url, outputPath);

      if (isJsonMode()) {
        return printJson({
          id: fullId,
          output_path: outputPath,
          filename: data.filename,
          bytes,
          expires_at: data.expires_at,
        });
      }

      const size = bytes == null ? "" : ` (${bytes} bytes)`;
      printSuccess(`Model downloaded to ${outputPath}${size}`);
    });

  models
    .command("delete")
    .description("Delete a model")
    .argument("<id>", "Model ID (full UUID or 4+ char prefix)")
    .action(async (id: string) => {
      const opts = parent.opts() as ClientOpts;
      const fullId = await resolveModelId(id, opts);
      await del(`/models/${fullId}`, opts);

      if (isJsonMode()) return printJson({ id: fullId, deleted: true });
      printSuccess(`Model deleted: ${shortId(fullId)}`);
    });
}
