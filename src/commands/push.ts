import { Command } from "commander";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { post, put, type ClientOpts } from "../client.js";
import { printSuccess, printJson, isJsonMode, shortId } from "../output.js";
import { loadSpec, DEFAULT_SPEC_FILE } from "./init.js";

interface RemoteSpec {
  id: string;
  name: string;
  [key: string]: unknown;
}

export function registerPushCommand(parent: Command) {
  parent
    .command("push")
    .description("Push local spec to the Tuned Tensor API")
    .option("-f, --file <path>", "Spec file path", DEFAULT_SPEC_FILE)
    .action(async (cmdOpts) => {
      const opts = parent.opts() as ClientOpts;
      const filePath = resolve(cmdOpts.file);
      const spec = loadSpec(cmdOpts.file);

      const { id, eval_cases, ...body } = spec as unknown as Record<string, unknown>;

      let data: RemoteSpec;

      if (id) {
        const res = await put<RemoteSpec>(`/behavior-specs/${id}`, body, opts);
        data = res.data;
      } else {
        const res = await post<RemoteSpec>("/behavior-specs", body, opts);
        data = res.data;

        const raw = JSON.parse(readFileSync(filePath, "utf-8"));
        raw.id = data.id;
        writeFileSync(filePath, JSON.stringify(raw, null, 2) + "\n");
      }

      if (isJsonMode()) return printJson(data);
      printSuccess(
        `Spec ${id ? "updated" : "created"}: ${data.name} (${shortId(data.id)})`,
      );
    });
}
