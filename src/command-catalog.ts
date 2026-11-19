export type WorkflowMode = "local";

export type CommandGroup =
  | "Workflow"
  | "Inspect"
  | "Serving";

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
  { path: "spec", description: "Review the current behavior spec.", group: "Workflow", modes: LOCAL },
  { path: "spec view", description: "Readable, highlighted behavior spec; add a section name for detail.", group: "Workflow", modes: LOCAL },
  ...["show", "diff", "validate", "history"].map(operation => ({ path: `spec ${operation}`, description: `Review behavior spec ${operation}.`, group: "Workflow" as const, modes: LOCAL })),
  { path: "init", description: "Create a local behaviour-spec project.", group: "Workflow", modes: LOCAL },
  { path: "validate", description: "Validate a local fine-tuning project.", group: "Workflow", modes: LOCAL },
  { path: "doctor", description: "Check the local host and run prerequisites.", group: "Workflow", modes: LOCAL },
  { path: "hardware", description: "Inventory this host and report train, fine-tune, and inference limits.", group: "Inspect", modes: LOCAL },
  { path: "pipeline init", description: "Write a canonical v1 pipeline recipe.", group: "Workflow", modes: LOCAL },
  { path: "pipeline validate", description: "Validate a pipeline without executing it.", group: "Workflow", modes: LOCAL },
  { path: "pipeline plan", description: "Resolve pipeline steps and required artifacts.", group: "Workflow", modes: LOCAL },
  { path: "pipeline run", description: "Run or dry-run a local pipeline.", group: "Workflow", modes: LOCAL },
  { path: "pipeline start", description: "Start a real run in the background with an audit record.", group: "Workflow", modes: LOCAL },
  { path: "pipeline list", description: "List background runs.", group: "Inspect", modes: LOCAL },
  { path: "pipeline status", description: "Show a background run's status, audit check and recent log.", group: "Inspect", modes: LOCAL },
  { path: "pipeline logs", description: "Print a background run's log.", group: "Inspect", modes: LOCAL },
  { path: "pipeline cancel", description: "Stop an active background run.", group: "Workflow", modes: LOCAL },
  { path: "shell", description: "Open the conversational terminal.", group: "Workflow", modes: LOCAL },

  { path: "runs list", description: "List local runs.", group: "Inspect", modes: LOCAL },
  { path: "runs get", description: "Show a local run.", group: "Inspect", modes: LOCAL },
  { path: "runs report", description: "Show a local run report.", group: "Inspect", modes: LOCAL },
  { path: "runs compare", description: "Compare two local run reports.", group: "Inspect", modes: LOCAL },
  { path: "runs events", description: "Show local run progress events.", group: "Inspect", modes: LOCAL },
  { path: "runs audit", description: "Verify a run's event log and report, with timeline and provenance.", group: "Inspect", modes: LOCAL },

  { path: "models list", description: "List local models.", group: "Inspect", modes: LOCAL },
  { path: "models get", description: "Show a local model.", group: "Inspect", modes: LOCAL },
  { path: "models serve", description: "Serve a model through an OpenAI-compatible API.", group: "Serving", modes: LOCAL },
  { path: "models verify", description: "Verify a local model artifact.", group: "Serving", modes: LOCAL },
  { path: "models prefetch", description: "Download the spec's base model, or any model with --repo.", group: "Serving", modes: LOCAL },
  { path: "models verify-base", description: "Verify the local base-model snapshot.", group: "Serving", modes: LOCAL },
  { path: "datasets inspect", description: "Validate and inspect a local chat JSONL file.", group: "Workflow", modes: LOCAL },
  { path: "datasets split", description: "Split local chat JSONL with ratios, grouping, and a seed.", group: "Workflow", modes: LOCAL },
  { path: "datasets prefetch", description: "Download the spec's dataset files, or any dataset files with --repo.", group: "Workflow", modes: LOCAL },
  { path: "datasets verify", description: "Verify the cached Hugging Face dataset files.", group: "Workflow", modes: LOCAL },
  { path: "assets list", description: "List downloaded Hugging Face datasets and models.", group: "Inspect", modes: LOCAL },
  { path: "assets path", description: "Print where a downloaded dataset or model lives.", group: "Inspect", modes: LOCAL },
  { path: "assets verify", description: "Re-hash downloaded files against their manifests.", group: "Inspect", modes: LOCAL },
  { path: "models active", description: "Show the active local model.", group: "Serving", modes: LOCAL },
  { path: "models activate", description: "Activate a verified local model.", group: "Serving", modes: LOCAL },
  { path: "models rollback", description: "Roll back the active local model.", group: "Serving", modes: LOCAL },
  { path: "serve", description: "Serve a local adapter, active model, or base model.", group: "Serving", modes: LOCAL },

  { path: "info", description: "Show local runtime package information.", group: "Inspect", modes: LOCAL },
  { path: "upgrade", description: "Upgrade tt to the latest release.", group: "Inspect", modes: LOCAL },
  { path: "status", description: "Show local project context.", group: "Inspect", modes: LOCAL },
  { path: "agent models", description: "List provider models for the laptop-local TT agent.", group: "Inspect", modes: LOCAL },
  { path: "agent configure", description: "Select the laptop-local TT agent model.", group: "Inspect", modes: LOCAL },

  { path: "agent status", description: "Show the laptop-local TT agent selection.", group: "Inspect", modes: LOCAL },
] as const;

export interface SlashCommand {
  path: string;
  description: string;
}

export const SLASH_COMMANDS: readonly SlashCommand[] = [
  { path: "/spec", description: "Readable behavior spec; add a section, show, diff, validate or history." },
  { path: "/system", description: "Live GPU, CPU and memory use, and what this machine can fine-tune." },
  { path: "/help", description: "Show commands; add a word to filter." },
  { path: "/status", description: "Show lightweight workflow status." },
  { path: "/context", description: "Show the current project context." },
  { path: "/model", description: "Show or change the agent's provider and model." },
  { path: "/login", description: "Save an API key for a model provider." },
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
