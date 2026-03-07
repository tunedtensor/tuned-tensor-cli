import { Command } from "commander";
import { readFileSync } from "node:fs";
import { get, post, put, del, type ClientOpts } from "../client.js";
import {
  printTable,
  printDetail,
  printSuccess,
  printJson,
  isJsonMode,
  formatDate,
  formatStatus,
  truncate,
  shortId,
} from "../output.js";

interface BehaviorSpec {
  id: string;
  name: string;
  description: string;
  system_prompt: string;
  guidelines: string[];
  examples: { input: string; output: string }[];
  constraints: string[];
  base_model: string;
  created_at: string;
  updated_at: string;
  _run_count?: number;
  _latest_run_status?: string;
  _latest_run_score?: number;
  _runs?: unknown[];
}

export function registerSpecsCommands(parent: Command) {
  const specs = parent
    .command("specs")
    .description("Manage behaviour specs");

  specs
    .command("list")
    .description("List behaviour specs")
    .option("-p, --page <n>", "Page number", "1")
    .option("--per-page <n>", "Results per page", "20")
    .action(async (cmdOpts) => {
      const opts = parent.opts() as ClientOpts;
      const { data, meta } = await get<BehaviorSpec[]>(
        "/behavior-specs",
        { page: cmdOpts.page, per_page: cmdOpts.perPage },
        opts,
      );

      if (isJsonMode()) return printJson({ data, meta });

      printTable(
        ["ID", "Name", "Model", "Examples", "Runs", "Status", "Updated"],
        data.map((s) => [
          shortId(s.id),
          truncate(s.name, 30),
          s.base_model.split("/").pop() || s.base_model,
          String(s.examples?.length ?? 0),
          String(s._run_count ?? 0),
          s._latest_run_status ? formatStatus(s._latest_run_status) : "—",
          formatDate(s.updated_at),
        ]),
        meta,
      );
    });

  specs
    .command("get")
    .description("Show spec details")
    .argument("<id>", "Spec ID (full or prefix)")
    .action(async (id: string) => {
      const opts = parent.opts() as ClientOpts;
      const { data } = await get<BehaviorSpec>(`/behavior-specs/${id}`, undefined, opts);

      if (isJsonMode()) return printJson(data);

      printDetail([
        ["ID", data.id],
        ["Name", data.name],
        ["Description", data.description || undefined],
        ["Base Model", data.base_model],
        ["System Prompt", data.system_prompt ? truncate(data.system_prompt, 80) : undefined],
        ["Guidelines", data.guidelines?.length ? data.guidelines.join(", ") : undefined],
        ["Constraints", data.constraints?.length ? data.constraints.join(", ") : undefined],
        ["Examples", String(data.examples?.length ?? 0)],
        ["Runs", String(data._run_count ?? data._runs?.length ?? 0)],
        ["Created", formatDate(data.created_at)],
        ["Updated", formatDate(data.updated_at)],
      ]);

      if (data.examples?.length) {
        console.log("\nExamples:");
        for (const [i, ex] of data.examples.slice(0, 5).entries()) {
          console.log(`  ${i + 1}. Input:  ${truncate(ex.input, 60)}`);
          console.log(`     Output: ${truncate(ex.output, 60)}`);
        }
        if (data.examples.length > 5) {
          console.log(`  ... and ${data.examples.length - 5} more`);
        }
      }
    });

  specs
    .command("create")
    .description("Create a behaviour spec")
    .option("-f, --file <path>", "JSON file with spec definition")
    .option("-n, --name <name>", "Spec name")
    .option("--model <model>", "Base model ID")
    .action(async (cmdOpts) => {
      const opts = parent.opts() as ClientOpts;
      let body: Record<string, unknown>;

      if (cmdOpts.file) {
        body = JSON.parse(readFileSync(cmdOpts.file, "utf-8"));
      } else if (cmdOpts.name) {
        body = { name: cmdOpts.name };
        if (cmdOpts.model) body.base_model = cmdOpts.model;
      } else {
        console.error("Provide --file or --name");
        process.exit(1);
      }

      const { data } = await post<BehaviorSpec>("/behavior-specs", body, opts);

      if (isJsonMode()) return printJson(data);
      printSuccess(`Spec created: ${data.name} (${shortId(data.id)})`);
    });

  specs
    .command("update")
    .description("Update a behaviour spec")
    .argument("<id>", "Spec ID")
    .option("-f, --file <path>", "JSON file with fields to update")
    .option("-n, --name <name>", "New name")
    .option("--model <model>", "New base model ID")
    .action(async (id: string, cmdOpts) => {
      const opts = parent.opts() as ClientOpts;
      let body: Record<string, unknown>;

      if (cmdOpts.file) {
        body = JSON.parse(readFileSync(cmdOpts.file, "utf-8"));
      } else {
        body = {};
        if (cmdOpts.name) body.name = cmdOpts.name;
        if (cmdOpts.model) body.base_model = cmdOpts.model;
      }

      const { data } = await put<BehaviorSpec>(
        `/behavior-specs/${id}`,
        body,
        opts,
      );

      if (isJsonMode()) return printJson(data);
      printSuccess(`Spec updated: ${data.name} (${shortId(data.id)})`);
    });

  specs
    .command("delete")
    .description("Delete a behaviour spec")
    .argument("<id>", "Spec ID")
    .action(async (id: string) => {
      const opts = parent.opts() as ClientOpts;
      await del(`/behavior-specs/${id}`, opts);

      if (isJsonMode()) return printJson({ id, deleted: true });
      printSuccess(`Spec deleted: ${shortId(id)}`);
    });
}
