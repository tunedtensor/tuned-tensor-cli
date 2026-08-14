export type WorkflowMode = "cloud" | "local";

export type CommandGroup =
  | "Workflow"
  | "Inspect"
  | "Data"
  | "Account"
  | "Serving";

export interface CatalogCommand {
  path: string;
  description: string;
  group: CommandGroup;
  modes: readonly WorkflowMode[];
}

const CLOUD = ["cloud"] as const;
const LOCAL = ["local"] as const;
const BOTH = ["cloud", "local"] as const;

/**
 * The catalog is deliberately metadata-only. Commander and TT Local remain the
 * source of truth for argument validation; the shell uses this list for
 * discovery, help, and completion without executing either backend.
 */
export const COMMAND_CATALOG: readonly CatalogCommand[] = [
  { path: "init", description: "Create a behaviour-spec project.", group: "Workflow", modes: BOTH },
  { path: "eval", description: "Validate a managed-service behaviour spec.", group: "Workflow", modes: CLOUD },
  { path: "push", description: "Create or update the cloud behaviour spec.", group: "Workflow", modes: CLOUD },
  { path: "validate", description: "Validate a local fine-tuning project.", group: "Workflow", modes: LOCAL },
  { path: "doctor", description: "Check the local host and run prerequisites.", group: "Workflow", modes: LOCAL },
  { path: "run", description: "Run the local fine-tuning and evaluation workflow.", group: "Workflow", modes: LOCAL },

  { path: "runs list", description: "List runs.", group: "Inspect", modes: BOTH },
  { path: "runs get", description: "Show a run.", group: "Inspect", modes: BOTH },
  { path: "runs report", description: "Show a run report.", group: "Inspect", modes: BOTH },
  { path: "runs compare", description: "Compare two local run reports.", group: "Inspect", modes: LOCAL },
  { path: "runs events", description: "Show local run progress events.", group: "Inspect", modes: LOCAL },
  { path: "runs estimate", description: "Estimate managed run cost and duration.", group: "Workflow", modes: CLOUD },
  { path: "runs start", description: "Start a managed fine-tuning run.", group: "Workflow", modes: CLOUD },
  { path: "runs watch", description: "Watch a managed run.", group: "Inspect", modes: CLOUD },
  { path: "runs diagnose", description: "Show managed run diagnostics.", group: "Inspect", modes: CLOUD },
  { path: "runs cancel", description: "Cancel a managed run.", group: "Workflow", modes: CLOUD },

  { path: "models list", description: "List models.", group: "Inspect", modes: BOTH },
  { path: "models get", description: "Show a model.", group: "Inspect", modes: BOTH },
  { path: "models base", description: "List managed-service base models.", group: "Inspect", modes: CLOUD },
  { path: "models download", description: "Download a managed model artifact.", group: "Serving", modes: CLOUD },
  { path: "models export", description: "Export a managed model to GGUF or Ollama.", group: "Serving", modes: CLOUD },
  { path: "models setup-runtime", description: "Install the managed model-serving runtime.", group: "Serving", modes: CLOUD },
  { path: "models serve", description: "Serve a model through an OpenAI-compatible API.", group: "Serving", modes: BOTH },
  { path: "models delete", description: "Delete a managed model.", group: "Workflow", modes: CLOUD },
  { path: "models verify", description: "Verify a local model artifact.", group: "Serving", modes: LOCAL },
  { path: "models prefetch", description: "Download the local base-model snapshot.", group: "Serving", modes: LOCAL },
  { path: "models verify-base", description: "Verify the local base-model snapshot.", group: "Serving", modes: LOCAL },
  { path: "models active", description: "Show the active local model.", group: "Serving", modes: LOCAL },
  { path: "models activate", description: "Activate a verified local model.", group: "Serving", modes: LOCAL },
  { path: "models rollback", description: "Roll back the active local model.", group: "Serving", modes: LOCAL },
  { path: "serve", description: "Serve a local adapter, active model, or base model.", group: "Serving", modes: LOCAL },

  { path: "specs list", description: "List cloud behaviour specs.", group: "Inspect", modes: CLOUD },
  { path: "specs get", description: "Show a cloud behaviour spec.", group: "Inspect", modes: CLOUD },
  { path: "specs create", description: "Create a cloud behaviour spec.", group: "Workflow", modes: CLOUD },
  { path: "specs update", description: "Update a cloud behaviour spec.", group: "Workflow", modes: CLOUD },
  { path: "specs delete", description: "Delete a cloud behaviour spec.", group: "Workflow", modes: CLOUD },

  { path: "datasets list", description: "List cloud datasets.", group: "Data", modes: CLOUD },
  { path: "datasets get", description: "Show a cloud dataset.", group: "Data", modes: CLOUD },
  { path: "datasets upload", description: "Upload a cloud dataset.", group: "Data", modes: CLOUD },
  { path: "datasets delete", description: "Delete a cloud dataset.", group: "Data", modes: CLOUD },

  { path: "label upload", description: "Start a cloud teacher-labeling job.", group: "Data", modes: CLOUD },
  { path: "label watch", description: "Watch a cloud labeling job.", group: "Data", modes: CLOUD },
  { path: "label list", description: "List cloud labeling jobs.", group: "Data", modes: CLOUD },
  { path: "label status", description: "Show cloud labeling-job status.", group: "Data", modes: CLOUD },
  { path: "label rows", description: "Review labeled rows.", group: "Data", modes: CLOUD },
  { path: "label accept", description: "Accept labeled rows.", group: "Data", modes: CLOUD },
  { path: "label reject", description: "Reject labeled rows.", group: "Data", modes: CLOUD },
  { path: "label edit", description: "Edit a labeled row.", group: "Data", modes: CLOUD },
  { path: "label promote", description: "Promote reviewed rows to a dataset.", group: "Data", modes: CLOUD },
  { path: "label cancel", description: "Cancel a labeling job.", group: "Data", modes: CLOUD },

  { path: "auth login", description: "Store a cloud API key.", group: "Account", modes: CLOUD },
  { path: "auth logout", description: "Remove stored cloud credentials.", group: "Account", modes: CLOUD },
  { path: "auth status", description: "Show cloud authentication status.", group: "Account", modes: CLOUD },
  { path: "balance", description: "Show cloud credit balance.", group: "Account", modes: CLOUD },
  { path: "topup", description: "Open a cloud credit checkout.", group: "Account", modes: CLOUD },
  { path: "info", description: "Show TT Local package information.", group: "Inspect", modes: LOCAL },
] as const;

export interface SlashCommand {
  path: string;
  description: string;
}

export const SLASH_COMMANDS: readonly SlashCommand[] = [
  { path: "/help", description: "Show commands; add a word to filter." },
  { path: "/status", description: "Show lightweight workflow status." },
  { path: "/context", description: "Show the current project and backend context." },
  { path: "/model", description: "Show or change the TT agent model." },
  { path: "/cd", description: "Change the shell's working directory." },
  { path: "/clear", description: "Clear the terminal." },
  { path: "/exit", description: "Exit the TT shell." },
] as const;

export function catalogForMode(mode: WorkflowMode): CatalogCommand[] {
  return COMMAND_CATALOG.filter((command) => command.modes.includes(mode));
}

export function commandPathsForMode(mode: WorkflowMode): string[] {
  return [...new Set(catalogForMode(mode).map((command) => command.path))]
    .sort((left, right) => left.localeCompare(right));
}

export function groupedCatalog(
  mode: WorkflowMode,
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

function completionTarget(
  line: string,
  activeMode: WorkflowMode,
): { prefix: string; mode: WorkflowMode; fragment: string } {
  const leadingWhitespace = line.match(/^\s*/)?.[0] ?? "";
  const content = line.slice(leadingWhitespace.length);
  const override = content.match(/^(cloud|local)(?:\s+|$)/);
  if (!override) {
    return { prefix: leadingWhitespace, mode: activeMode, fragment: content };
  }
  const mode = override[1] as WorkflowMode;
  const prefix = `${leadingWhitespace}${mode} `;
  return {
    prefix,
    mode,
    fragment: content.slice(override[0].length),
  };
}

/**
 * A readline completer that replaces the whole command fragment. It performs
 * no filesystem or network work and never expands shell syntax.
 */
export function createCommandCompleter(
  getMode: () => WorkflowMode,
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

    const { prefix, mode, fragment } = completionTarget(line, getMode());
    const paths = commandPathsForMode(mode);
    const targetCandidates = paths.map((path) => `${prefix}${path}`);
    if (!prefix.trim()) targetCandidates.unshift("cloud ", "local ");
    const normalizedLine = line.trimStart().toLowerCase();
    const normalizedFragment = fragment.toLowerCase();
    const matches = targetCandidates.filter((candidate) => {
      if (prefix.trim()) return candidate.toLowerCase().startsWith(`${prefix.toLowerCase()}${normalizedFragment}`);
      return candidate.toLowerCase().startsWith(normalizedLine);
    });
    return [matches.length > 0 ? matches : targetCandidates, line];
  };
}
