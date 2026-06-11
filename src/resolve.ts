import { get, type ClientOpts } from "./client.js";

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

const RESOLVE_PAGE_SIZE = 100;
const MIN_PREFIX_LEN = 4;

interface Identifiable {
  id: string;
  name?: string;
  run_number?: number;
}

interface Listed<T> {
  data: T[];
  meta?: { page: number; per_page: number; total: number };
}

export class ResolveError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ResolveError";
  }
}

export function isFullUuid(value: string): boolean {
  return UUID_RE.test(value);
}

async function resolvePrefix<T extends Identifiable>(
  prefix: string,
  kind: "spec" | "run" | "dataset" | "model" | "labeling job",
  listPath: string,
  describe: (item: T) => string,
  opts?: ClientOpts,
  listCmdOverride?: string,
): Promise<string> {
  if (isFullUuid(prefix)) return prefix;

  const noun = kind;
  const listCmd = listCmdOverride ?? `tt ${kind}s list --json`;

  if (prefix.length < MIN_PREFIX_LEN) {
    throw new ResolveError(
      `${capitalize(noun)} ID prefix "${prefix}" is too short — provide at least ${MIN_PREFIX_LEN} characters or the full UUID.`,
    );
  }

  const lower = prefix.toLowerCase();
  const { data, meta } = (await get<T[]>(
    listPath,
    { page: 1, per_page: RESOLVE_PAGE_SIZE },
    opts,
  )) as Listed<T>;

  const matches = data.filter((item) => item.id.toLowerCase().startsWith(lower));

  if (matches.length === 1) return matches[0].id;

  if (matches.length === 0) {
    if (meta && meta.total > data.length) {
      throw new ResolveError(
        `No ${noun} found with ID prefix "${prefix}" in the first ${data.length} ${noun}s (you have ${meta.total} total). Use the full UUID — see \`${listCmd}\`.`,
      );
    }
    throw new ResolveError(`No ${noun} found with ID prefix "${prefix}".`);
  }

  const list = matches.map((m) => `  ${describe(m)}`).join("\n");
  throw new ResolveError(
    `${capitalize(noun)} ID prefix "${prefix}" is ambiguous (${matches.length} matches):\n${list}\nProvide more characters or the full UUID.`,
  );
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export function resolveSpecId(prefix: string, opts?: ClientOpts): Promise<string> {
  return resolvePrefix<Identifiable>(
    prefix,
    "spec",
    "/behavior-specs",
    (s) => `${s.id}${s.name ? `  (${s.name})` : ""}`,
    opts,
  );
}

export function resolveRunId(prefix: string, opts?: ClientOpts): Promise<string> {
  return resolvePrefix<Identifiable>(
    prefix,
    "run",
    "/runs",
    (r) => `${r.id}${r.run_number != null ? `  (run #${r.run_number})` : ""}`,
    opts,
  );
}

export function resolveDatasetId(prefix: string, opts?: ClientOpts): Promise<string> {
  return resolvePrefix<Identifiable>(
    prefix,
    "dataset",
    "/datasets",
    (d) => `${d.id}${d.name ? `  (${d.name})` : ""}`,
    opts,
  );
}

export function resolveModelId(prefix: string, opts?: ClientOpts): Promise<string> {
  return resolvePrefix<Identifiable>(
    prefix,
    "model",
    "/models",
    (m) => `${m.id}${m.name ? `  (${m.name})` : ""}`,
    opts,
  );
}

export function resolveLabelingJobId(
  prefix: string,
  opts?: ClientOpts,
): Promise<string> {
  return resolvePrefix<Identifiable>(
    prefix,
    "labeling job",
    "/labeling-jobs",
    (j) => `${j.id}${j.name ? `  (${j.name})` : ""}`,
    opts,
    "tt label list --json",
  );
}
