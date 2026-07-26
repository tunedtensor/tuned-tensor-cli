import { Command } from "commander";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { ApiError, post, put, type ClientOpts } from "../client.js";
import { printSuccess, printJson, isJsonMode, shortId } from "../output.js";
import { loadSpec, DEFAULT_SPEC_FILE } from "./init.js";
import { validateEvalCases } from "../eval/rules.js";
import {
  hasLocalOnlySpecFields,
  projectCloudSpec,
} from "../project-spec.js";

interface RemoteSpec {
  id: string;
  name: string;
  [key: string]: unknown;
}

function persistRemoteId(filePath: string, id: string): void {
  const raw = JSON.parse(readFileSync(filePath, "utf-8"));
  raw.id = id;
  writeFileSync(filePath, JSON.stringify(raw, null, 2) + "\n");
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

      const rawSpec = spec as unknown as Record<string, unknown>;
      const { body } = projectCloudSpec(rawSpec);
      const id = spec.id;
      const canRecoverLocalId = Boolean(
        id && hasLocalOnlySpecFields(rawSpec),
      );

      let data: RemoteSpec;
      let created = false;

      if (id) {
        try {
          const res = await put<RemoteSpec>(
            `/behavior-specs/${id}`,
            body,
            opts,
          );
          data = res.data;
        } catch (error) {
          if (
            !canRecoverLocalId
            || !(error instanceof ApiError)
            || error.status !== 404
          ) {
            throw error;
          }

          const res = await post<RemoteSpec>("/behavior-specs", body, opts);
          data = res.data;
          created = true;
          persistRemoteId(filePath, data.id);
        }
      } else {
        const res = await post<RemoteSpec>("/behavior-specs", body, opts);
        data = res.data;
        created = true;
        persistRemoteId(filePath, data.id);
      }

      if (isJsonMode()) return printJson(data);
      printSuccess(
        `Spec ${created ? "created" : "updated"}: ${data.name} (${shortId(data.id)})`,
      );
    });
}
