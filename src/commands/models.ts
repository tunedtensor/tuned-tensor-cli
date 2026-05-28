import { Command } from "commander";
import { existsSync, mkdirSync, statSync, createWriteStream, unlinkSync } from "node:fs";
import { dirname, join } from "node:path";
import { Readable, Transform } from "node:stream";
import { pipeline } from "node:stream/promises";
import { get, del, type ClientOpts } from "../client.js";
import { resolveModelId } from "../resolve.js";
import { SUPPORTED_BASE_MODELS } from "../base-models.js";
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

function parseContentLength(value: string | null): number | null {
  if (!value) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : null;
}

function formatBytes(bytes: number): string {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  const precision = unit === 0 || value >= 10 ? 0 : 1;
  return `${value.toFixed(precision)} ${units[unit]}`;
}

function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "--";
  const rounded = Math.max(1, Math.round(seconds));
  const hours = Math.floor(rounded / 3600);
  const minutes = Math.floor((rounded % 3600) / 60);
  const secs = rounded % 60;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m ${secs}s`;
  return `${secs}s`;
}

function chunkLength(chunk: unknown): number {
  if (typeof chunk === "string") return Buffer.byteLength(chunk);
  if (chunk instanceof Uint8Array) return chunk.byteLength;
  return Buffer.byteLength(String(chunk));
}

function createDownloadProgress(totalBytes: number | null, enabled: boolean) {
  const stream = process.stderr;
  const start = Date.now();
  let lastRender = 0;
  let rendered = false;

  function render(downloadedBytes: number, force = false) {
    if (!enabled) return;
    const now = Date.now();
    if (!force && now - lastRender < 100 && downloadedBytes !== totalBytes) return;
    lastRender = now;
    rendered = true;

    const elapsedSeconds = Math.max((now - start) / 1000, 0.001);
    const rate = downloadedBytes / elapsedSeconds;
    const rateText = `${formatBytes(rate)}/s`;

    let line: string;
    if (totalBytes != null && totalBytes > 0) {
      const ratio = Math.min(downloadedBytes / totalBytes, 1);
      const barWidth = 24;
      const filled = Math.round(ratio * barWidth);
      const bar = "#".repeat(filled) + "-".repeat(barWidth - filled);
      const etaSeconds = rate > 0 ? (totalBytes - downloadedBytes) / rate : Number.NaN;
      line = `Downloading [${bar}] ${(ratio * 100).toFixed(1).padStart(5)}% ${formatBytes(downloadedBytes)}/${formatBytes(totalBytes)} ${rateText} ETA ${formatDuration(etaSeconds)}`;
    } else {
      line = `Downloading ${formatBytes(downloadedBytes)} ${rateText} elapsed ${formatDuration(elapsedSeconds)}`;
    }

    const columns = stream.columns || 120;
    stream.write(`\r\x1b[K${line.slice(0, Math.max(columns - 1, 0))}`);
  }

  function stop() {
    if (enabled && rendered) stream.write("\n");
  }

  return { render, stop };
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
  const contentLength = parseContentLength(res.headers.get("content-length"));
  const progress = createDownloadProgress(
    contentLength,
    !isJsonMode() && process.stderr.isTTY === true,
  );
  let downloadedBytes = 0;
  const progressStream = new Transform({
    transform(chunk, _encoding, callback) {
      downloadedBytes += chunkLength(chunk);
      progress.render(downloadedBytes);
      callback(null, chunk);
    },
  });

  try {
    progress.render(0, true);
    await pipeline(Readable.fromWeb(res.body), progressStream, file);
    progress.render(downloadedBytes, true);
    progress.stop();
  } catch (err) {
    progress.stop();
    try {
      unlinkSync(outputPath);
    } catch {
      // Best effort cleanup for partially written downloads.
    }
    throw err;
  }

  return contentLength;
}

export function registerModelsCommands(parent: Command) {
  const models = parent.command("models").description("Manage fine-tuned models");

  models
    .command("base")
    .description("List supported base models")
    .action(async () => {
      const data = SUPPORTED_BASE_MODELS.map((model) => ({
        id: model,
        name: model.split("/").pop() || model,
        type: "base" as const,
      }));

      if (isJsonMode()) return printJson({ data });

      printTable(
        ["ID", "Name", "Type"],
        data.map((m) => [m.id, m.name, m.type]),
      );
    });

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
        ["Hosted Model", data.provider_model_id],
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
