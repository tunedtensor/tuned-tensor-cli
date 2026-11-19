import type { AgentAction } from "./agent-client.js";

function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

/** The bounded part of an approved action's output that the model may read on its next turn. */
export function approvedResultForModel(operation: string | undefined, output: unknown): Record<string, unknown> | undefined {
  const value = record(output);
  if (!value) return undefined;
  if (operation === "split_local_dataset") {
    return { operation, source: value.source, source_sha256: value.source_sha256, counts: value.counts, dataset_prebuilt: value.dataset_prebuilt, note: value.note };
  }
  if (operation === "prefetch_local") {
    return {
      operation, kind: value.kind, repo: value.repo, revision: value.revision,
      snapshot_path: value.snapshot_path, manifest_path: value.manifest_path,
      lock_path: value.lock_path, files: value.files,
      file_count: value.file_count, size_bytes: value.size_bytes, checksums_verified: value.checksums_verified,
      status: value.status, local_base_model_path: value.local_base_model_path,
      reason: value.reason,
    };
  }
  return undefined;
}

export function approvedResultsNote(actions: AgentAction[]): string | undefined {
  const settled = new Set(["completed", "failed", "rejected", "outcome_unknown"]);
  const pending = actions.filter((action) => action.result_reported !== true
    && (action.result !== undefined || settled.has(action.status ?? "")));
  if (pending.length === 0) return undefined;
  for (const action of pending) action.result_reported = true;
  const outcomes = pending.map((action) => {
    const args = record(action.arguments);
    return JSON.stringify({
      action_id: action.id, operation: action.operation, status: action.status,
      // The proposal is already in the transcript. Carry its durable outcome,
      // not another copy of the spec or preview, into the next model turn.
      spec_path: args?.spec_path,
      result: action.result,
    });
  });
  return `[Action outcomes, recorded by tt]\n${outcomes.join("\n")}\nThese are historical action outcomes. Read the current spec before describing its current contents or proposing another edit.`;
}
