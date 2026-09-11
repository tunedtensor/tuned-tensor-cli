export type WorkflowMode = "local";

export type CommandGroup =
  | "Workflow"
  | "Inspect"
  | "Serving"
  | "Account"
  | "Cloud";

export interface CatalogCommand {
  path: string;
  description: string;
  group: CommandGroup;
  modes: readonly WorkflowMode[];
}

const LOCAL = ["local"] as const;

/**
 * The catalog is deliberately metadata-only. Commander and the local runtime
 * remain the source of truth for argument validation; the shell uses this list
 * for discovery, help, and completion without executing either backend.
 */
export const COMMAND_CATALOG: readonly CatalogCommand[] = [
  { path: "init", description: "Create a local behaviour-spec project.", group: "Workflow", modes: LOCAL },
  { path: "validate", description: "Validate a local fine-tuning project.", group: "Workflow", modes: LOCAL },
  { path: "doctor", description: "Check the local host and run prerequisites.", group: "Workflow", modes: LOCAL },
  { path: "hardware", description: "Inventory this host and report train, fine-tune, and inference limits.", group: "Inspect", modes: LOCAL },
  { path: "pipeline init", description: "Write a canonical v1 pipeline recipe.", group: "Workflow", modes: LOCAL },
  { path: "pipeline validate", description: "Validate a pipeline without executing it.", group: "Workflow", modes: LOCAL },
  { path: "pipeline plan", description: "Resolve pipeline steps and required artifacts.", group: "Workflow", modes: LOCAL },
  { path: "pipeline run", description: "Run or dry-run a local pipeline.", group: "Workflow", modes: LOCAL },
  { path: "shell", description: "Open the conversational terminal.", group: "Workflow", modes: LOCAL },

  { path: "runs list", description: "List local runs.", group: "Inspect", modes: LOCAL },
  { path: "runs get", description: "Show a local run.", group: "Inspect", modes: LOCAL },
  { path: "runs report", description: "Show a local run report.", group: "Inspect", modes: LOCAL },
  { path: "runs compare", description: "Compare two local run reports.", group: "Inspect", modes: LOCAL },
  { path: "runs events", description: "Show local run progress events.", group: "Inspect", modes: LOCAL },

  { path: "models list", description: "List local models.", group: "Inspect", modes: LOCAL },
  { path: "models get", description: "Show a local model.", group: "Inspect", modes: LOCAL },
  { path: "models serve", description: "Serve a model through an OpenAI-compatible API.", group: "Serving", modes: LOCAL },
  { path: "models verify", description: "Verify a local model artifact.", group: "Serving", modes: LOCAL },
  { path: "models prefetch", description: "Download the local base-model snapshot.", group: "Serving", modes: LOCAL },
  { path: "models verify-base", description: "Verify the local base-model snapshot.", group: "Serving", modes: LOCAL },
  { path: "models active", description: "Show the active local model.", group: "Serving", modes: LOCAL },
  { path: "models activate", description: "Activate a verified local model.", group: "Serving", modes: LOCAL },
  { path: "models rollback", description: "Roll back the active local model.", group: "Serving", modes: LOCAL },
  { path: "serve", description: "Serve a local adapter, active model, or base model.", group: "Serving", modes: LOCAL },

  { path: "info", description: "Show local runtime package information.", group: "Inspect", modes: LOCAL },
  { path: "status", description: "Show local project context.", group: "Inspect", modes: LOCAL },
  { path: "agent models", description: "List provider models for the laptop-local TT agent.", group: "Inspect", modes: LOCAL },
  { path: "agent configure", description: "Select the laptop-local TT agent model.", group: "Inspect", modes: LOCAL },
  { path: "cloud specs list", description: "List behaviour specs", group: "Cloud", modes: LOCAL },
  { path: "cloud specs get", description: "Show spec details", group: "Cloud", modes: LOCAL },
  { path: "cloud specs create", description: "Create a behaviour spec", group: "Cloud", modes: LOCAL },
  { path: "cloud specs update", description: "Update a behaviour spec", group: "Cloud", modes: LOCAL },
  { path: "cloud specs delete", description: "Delete a behaviour spec", group: "Cloud", modes: LOCAL },
  { path: "cloud datasets list", description: "List datasets", group: "Cloud", modes: LOCAL },
  { path: "cloud datasets get", description: "Show dataset details", group: "Cloud", modes: LOCAL },
  { path: "cloud datasets upload", description: "Upload a JSONL dataset file", group: "Cloud", modes: LOCAL },
  { path: "cloud datasets delete", description: "Delete a dataset", group: "Cloud", modes: LOCAL },
  { path: "cloud label upload", description: "Upload unlabeled inputs (.jsonl or .csv) and start a labeling job", group: "Cloud", modes: LOCAL },
  { path: "cloud label watch", description: "Watch a labeling job until it is ready for review", group: "Cloud", modes: LOCAL },
  { path: "cloud label list", description: "List labeling jobs", group: "Cloud", modes: LOCAL },
  { path: "cloud label status", description: "Show labeling job details and review progress", group: "Cloud", modes: LOCAL },
  { path: "cloud label rows", description: "List rows in a labeling job", group: "Cloud", modes: LOCAL },
  { path: "cloud label accept", description: "Accept teacher-labeled rows", group: "Cloud", modes: LOCAL },
  { path: "cloud label reject", description: "Reject rows so they are excluded from promotion", group: "Cloud", modes: LOCAL },
  { path: "cloud label edit", description: "Replace a row's output with your own", group: "Cloud", modes: LOCAL },
  { path: "cloud label promote", description: "Promote reviewed rows into a validated dataset", group: "Cloud", modes: LOCAL },
  { path: "cloud label cancel", description: "Cancel a labeling job and release unused credits", group: "Cloud", modes: LOCAL },
  { path: "cloud runs list", description: "List runs", group: "Cloud", modes: LOCAL },
  { path: "cloud runs get", description: "Show run details and eval results", group: "Cloud", modes: LOCAL },
  { path: "cloud runs estimate", description: "Retired: use the local pipeline with your own AWS GPU", group: "Cloud", modes: LOCAL },
  { path: "cloud runs start", description: "Retired: use the local pipeline with your own AWS GPU", group: "Cloud", modes: LOCAL },
  { path: "cloud runs cancel", description: "Cancel a running run", group: "Cloud", modes: LOCAL },
  { path: "cloud runs watch", description: "Watch a run until it completes", group: "Cloud", modes: LOCAL },
  { path: "cloud runs diagnose", description: "Show live run diagnostics", group: "Cloud", modes: LOCAL },
  { path: "cloud runs report", description: "Show run metrics and side-by-side eval output insights", group: "Cloud", modes: LOCAL },
  { path: "cloud models base", description: "List supported base models", group: "Cloud", modes: LOCAL },
  { path: "cloud models list", description: "List fine-tuned models", group: "Cloud", modes: LOCAL },
  { path: "cloud models get", description: "Show model details", group: "Cloud", modes: LOCAL },
  { path: "cloud models download", description: "Download a fine-tuned model artifact", group: "Cloud", modes: LOCAL },
  { path: "cloud models export", description: "Export a fine-tuned model to GGUF and (optionally) package it for Ollama", group: "Cloud", modes: LOCAL },
  { path: "cloud models setup-runtime", description: "Install an isolated Python runtime for local model serving", group: "Cloud", modes: LOCAL },
  { path: "cloud models serve", description: "Serve a downloaded model with an OpenAI-compatible local API", group: "Cloud", modes: LOCAL },
  { path: "cloud models delete", description: "Delete a model", group: "Cloud", modes: LOCAL },
  { path: "cloud push", description: "Push a local spec to the cloud.", group: "Cloud", modes: LOCAL },
  { path: "cloud runs archive", description: "Archive an inactive published local run report.", group: "Cloud", modes: LOCAL },
  { path: "auth login", description: "Save a TT access token for managed inference and cloud access.", group: "Account", modes: LOCAL },
  { path: "auth logout", description: "Remove the stored TT access token.", group: "Account", modes: LOCAL },
  { path: "auth status", description: "Show TT account authentication.", group: "Account", modes: LOCAL },
  { path: "balance", description: "Show cloud training credits and transactions.", group: "Account", modes: LOCAL },
  { path: "topup", description: "Add cloud training credits.", group: "Account", modes: LOCAL },
  { path: "usage", description: "Show managed agent allowance and token usage.", group: "Account", modes: LOCAL },
  { path: "publish", description: "Publish local run evidence to the dashboard.", group: "Account", modes: LOCAL },

  { path: "agent status", description: "Show the laptop-local TT agent selection.", group: "Inspect", modes: LOCAL },
] as const;

export interface SlashCommand {
  path: string;
  description: string;
}

export const SLASH_COMMANDS: readonly SlashCommand[] = [
  { path: "/help", description: "Show commands; add a word to filter." },
  { path: "/status", description: "Show lightweight workflow status." },
  { path: "/context", description: "Show the current project context." },
  { path: "/model", description: "Show or change managed inference or your own provider/model." },
  { path: "/login", description: "Save a TT access token (tunedtensor) or a provider API key." },
  { path: "/cd", description: "Change the shell's working directory." },
  { path: "/clear", description: "Clear the terminal." },
  { path: "/exit", description: "Exit the TT shell." },
] as const;

export function catalogForMode(mode: WorkflowMode = "local"): CatalogCommand[] {
  return COMMAND_CATALOG.filter((command) => command.modes.includes(mode));
}

export function commandPathsForMode(mode: WorkflowMode = "local"): string[] {
  return [...new Set(catalogForMode(mode).map((command) => command.path))]
    .sort((left, right) => left.localeCompare(right));
}

export function groupedCatalog(
  mode: WorkflowMode = "local",
  query?: string,
): Map<CommandGroup, CatalogCommand[]> {
  const normalizedQuery = query?.trim().toLowerCase();
  const matches = catalogForMode(mode).filter((command) => {
    if (!normalizedQuery) return true;
    return command.path.toLowerCase().includes(normalizedQuery)
      || command.description.toLowerCase().includes(normalizedQuery)
      || command.group.toLowerCase().includes(normalizedQuery);
  });
  const groups = new Map<CommandGroup, CatalogCommand[]>();
  for (const command of matches) {
    const group = groups.get(command.group) ?? [];
    group.push(command);
    groups.set(command.group, group);
  }
  return groups;
}

/**
 * A readline completer that replaces the whole command fragment. It performs
 * no filesystem or network work and never expands shell syntax.
 */
export function createCommandCompleter(
  getMode: () => WorkflowMode = () => "local",
): (line: string) => [string[], string] {
  return (line) => {
    const trimmed = line.trimStart();
    if (trimmed.startsWith("/")) {
      const candidates = [
        ...SLASH_COMMANDS.map((command) => command.path),
        "/new",
        "/threads",
        "/resume ",
        "/approve ",
        "/reject ",
      ];
      const matches = candidates.filter((candidate) => candidate.startsWith(trimmed));
      return [matches.length > 0 ? matches : candidates, line];
    }
    if (trimmed.startsWith("?")) {
      return [["?"], line];
    }

    const leadingWhitespace = line.match(/^\s*/)?.[0] ?? "";
    const fragment = line.slice(leadingWhitespace.length);
    const paths = commandPathsForMode(getMode());
    const targetCandidates = paths.map((path) => `${leadingWhitespace}${path}`);
    const normalizedLine = line.trimStart().toLowerCase();
    const matches = targetCandidates.filter((candidate) =>
      candidate.toLowerCase().startsWith(normalizedLine)
    );
    return [matches.length > 0 ? matches : targetCandidates, line];
  };
}
