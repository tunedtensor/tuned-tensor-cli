import { Command } from "commander";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { post, put, type ClientOpts } from "../client.js";
import { printSuccess, printJson, isJsonMode, shortId } from "../output.js";
import { loadSpec, DEFAULT_SPEC_FILE } from "./init.js";
import { canonicalizeSpecBaseModel } from "../base-models.js";
import { validateEvalCases } from "../eval/rules.js";

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
      const evalCaseErrors = validateEvalCases(spec);
      if (evalCaseErrors.length > 0) {
        throw new Error(`Invalid eval_cases: ${evalCaseErrors.join("; ")}`);
      }

      const { id, ...rawBody } = spec as unknown as Record<string, unknown>;
      const body = canonicalizeSpecBaseModel(rawBody);

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
