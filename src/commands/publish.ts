import { Command } from "commander";
import { createInterface } from "node:readline/promises";
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { isAbsolute, join, resolve } from "node:path";
import { stdin as defaultStdin, stdout as defaultStdout } from "node:process";
import { post, type ClientOpts } from "../client.js";
import { getBaseUrl } from "../config.js";
import {
  createLocalStore,
  defaultLocalHome,
  type LocalRunState,
} from "../local-runtime/store.js";
import {
  printDetail,
  printJson,
  printSuccess,
  printWarning,
  isJsonMode,
  shortId,
} from "../output.js";

interface PublishResult {
  id: string;
  spec_id: string;
  run_number: number;
  origin: string;
  source_run_id: string;
  source_spec_id: string;
}

function expandPath(
  value: string,
  baseDirectory: string,
  homeDirectory: string,
): string {
  if (value === "~") return homeDirectory;
  if (value.startsWith("~/")) return resolve(homeDirectory, value.slice(2));
  return isAbsolute(value) ? resolve(value) : resolve(baseDirectory, value);
}

export function resolvePublishStoreRoot(options?: {
  cwd?: string;
  env?: NodeJS.ProcessEnv;
  homeDirectory?: string;
}): string {
  const cwd = options?.cwd ?? process.cwd();
  const env = options?.env ?? process.env;
  const homeDirectory = options?.homeDirectory ?? (env.HOME ? resolve(env.HOME) : homedir());
  const localConfigPath = join(cwd, "local-runner.json");

  if (existsSync(localConfigPath)) {
    try {
      const raw = JSON.parse(readFileSync(localConfigPath, "utf-8")) as {
        storeRoot?: unknown;
      };
      if (typeof raw.storeRoot === "string" && raw.storeRoot.trim()) {
        return expandPath(raw.storeRoot, cwd, homeDirectory);
      }
    } catch {
      // Fall through to defaults when config is malformed.
    }
  }

  if (env.TT_LOCAL_HOME && env.TT_LOCAL_HOME.trim()) {
    return expandPath(env.TT_LOCAL_HOME, cwd, homeDirectory);
  }

  return defaultLocalHome();
}

async function confirmPublish(
  question: string,
  input: NodeJS.ReadableStream = defaultStdin,
  output: NodeJS.WritableStream = defaultStdout,
): Promise<boolean> {
  if (
    (input as { isTTY?: boolean }).isTTY !== true
    || (output as { isTTY?: boolean }).isTTY !== true
  ) {
    throw new Error(
      "Publishing requires confirmation in interactive mode. Re-run with --yes.",
    );
  }

  const rl = createInterface({ input, output, terminal: true });
  try {
    const answer = (await rl.question(question)).trim().toLowerCase();
    return answer === "y" || answer === "yes";
  } finally {
    rl.close();
  }
}

function pickPublishableRun(
  runs: LocalRunState[],
  runId?: string,
): LocalRunState {
  if (runId) {
    const match = runs.find(
      (run) => run.id === runId || run.id.startsWith(runId),
    );
    if (!match) {
      throw new Error(`Run not found: ${runId}`);
    }
    if (match.status !== "completed" && match.status !== "failed") {
      throw new Error(
        `Run ${shortId(match.id)} is still ${match.status}; only completed or failed runs can be published.`,
      );
    }
    return match;
  }

  const candidates = runs
    .filter((run) => run.status === "completed" || run.status === "failed")
    .sort((a, b) => b.updated_at.localeCompare(a.updated_at));
  const latest = candidates[0];
  if (!latest) {
    throw new Error(
      "No completed or failed local runs found. Run `tt run` first, then publish.",
    );
  }
  return latest;
}

export function registerPublishCommand(parent: Command) {
  parent
    .command("publish")
    .description("Publish local run evidence to the Tuned Tensor dashboard")
    .argument("[runId]", "Local run id or prefix (defaults to latest completed/failed run)")
    .option("-k, --api-key <key>", "API key (tt_...)")
    .option("-u, --base-url <url>", "API base URL")
    .option("-y, --yes", "Skip confirmation prompt")
    .option("--dry-run", "Show what would be published without uploading")
    .action(async (runId: string | undefined, cmdOpts) => {
      const opts: ClientOpts = {
        apiKey: cmdOpts.apiKey,
        baseUrl: cmdOpts.baseUrl,
      };
      const storeRoot = resolvePublishStoreRoot();
      const store = createLocalStore(storeRoot);
      const runs = await store.listRuns();
      const run = pickPublishableRun(runs, runId);
      const report = await store.getRunReport(run.id);
      const specRecord = await store.getSpec(run.behavior_spec_id);
      const exampleCount = report.candidate?.results?.length
        ?? report.baseline?.results?.length
        ?? 0;

      const payload = {
        spec: {
          id: specRecord.id,
          ...specRecord.spec,
        },
        report,
      };

      if (isJsonMode() && cmdOpts.dryRun) {
        printJson({
          dry_run: true,
          store_root: storeRoot,
          run_id: run.id,
          spec_id: specRecord.id,
          spec_name: specRecord.spec.name,
          run_number: run.run_number,
          status: run.status,
          example_count: exampleCount,
          payload,
        });
        return;
      }

      if (!isJsonMode()) {
        printDetail([
          ["Spec", specRecord.spec.name],
          ["Run", `${shortId(run.id)} (#${run.run_number})`],
          ["Status", run.status],
          ["Examples", String(exampleCount)],
          ["Store", storeRoot],
        ]);
        printWarning(
          "Prompts and model outputs from the run report will be uploaded to your Tuned Tensor account.",
        );
      }

      if (cmdOpts.dryRun) {
        if (isJsonMode()) {
          printJson({
            dry_run: true,
            run_id: run.id,
            spec_id: specRecord.id,
            example_count: exampleCount,
          });
          return;
        }
        printSuccess("Dry run only — nothing was uploaded.");
        return;
      }

      if (!cmdOpts.yes && !isJsonMode()) {
        const ok = await confirmPublish("Publish this run evidence? [y/N] ");
        if (!ok) {
          printWarning("Publish cancelled.");
          return;
        }
      }

      if (!cmdOpts.yes && isJsonMode()) {
        throw new Error(
          "JSON mode requires --yes to publish without an interactive prompt.",
        );
      }

      const { data } = await post<PublishResult>("/publish/runs", payload, opts);
      const baseUrl = getBaseUrl(opts).replace(/\/$/, "");
      const dashboardUrl = `${baseUrl}/dashboard/runs/${data.id}`;

      if (isJsonMode()) {
        printJson({ ...data, dashboard_url: dashboardUrl });
        return;
      }

      printSuccess(
        `Published run #${data.run_number} (${shortId(data.id)}) for ${specRecord.spec.name}`,
      );
      printDetail([
        ["Dashboard", dashboardUrl],
        ["Spec ID", shortId(data.spec_id)],
      ]);
    });
}
