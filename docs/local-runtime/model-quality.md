# Review and use a customized model through the agent

Adapter projects can now inspect data, review completed runs, activate a reviewed
model, serve it on the current CUDA machine, stop that service, and roll back
through conversation. Each change uses the existing reviewed-action approval
flow; inspection needs no approval.

## Agree on success before training

Keep acceptance criteria in `tunedtensor.json`, alongside the existing scoring
and general-regression settings. These values are illustrative, not universal
quality standards:

```json
{
  "evaluation": {
    "scoring": { "mode": "json_fields" },
    "taskQuality": {
      "minScore": 0.85,
      "minPassRate": 0.85,
      "minExamples": 40,
      "minScoreDelta": 0
    }
  }
}
```

Choose the scorer and thresholds for the actual task, using representative,
disjoint held-out examples. Keep a separate test set when selecting recipes
on validation results. `minScore` and `minPassRate` range from 0 to 1;
`minExamples` is the required evaluated count; `minScoreDelta` is the required
improvement over the baseline and defaults to zero. A pass-rate regression is
always rejected. Baseline and candidate must use the same prompts/references.
Training examples and dry-run output cannot establish acceptance.

The report records the policy used by that run. Agent activation and serving
require that policy, passing task and general-regression checks, unchanged
current criteria, and verified artifacts. Editing the spec cannot attach new
criteria to an old report. CLI activation of older reports without a policy
remains supported, but now also rejects task-score/pass-rate regressions.
Explicit CLI serving by model ID remains a diagnostic path without this quality
gate; integrity verification alone is not a quality assessment.

These policy and agent lifecycle features currently apply to adapter projects.
Foundation and decision engines retain their existing engine-specific workflows.

## Conversation workflow

1. “Inspect training.jsonl and validation.jsonl for duplicate prompts, conflicting
   answers and overlap.” `assess_local_dataset` scans supported single-turn chat
   JSONL, including pinned cached split sources. It reports exact normalized
   duplicates, length distributions and bounded answer counts. This does not
   establish semantic accuracy, license suitability, PII safety or business coverage.
2. Agree on the task criteria, update the spec through its reviewed edit flow,
   and preview/run training using the existing pipeline tools.
3. “Review my latest run. Activate it if it meets our agreed criteria.”
   `list_local_runs` discovers actual run IDs and `inspect_local_run` returns
   verified scores, the recorded evaluation split, gates and a bounded set of failures. The activation proposal
   seals the selected run, report, manifest, project configuration and current
   active selection. Approval rechecks these; changes require a new proposal.
4. “Serve this reviewed model on port 8000.” Approval starts loading; the agent
   checks `get_local_service` for health and the exact `/v1/models` identity.
   Starting a process is not reported as a ready service. The evaluated system
   instruction is restored from the saved run, not taken from a later spec edit.
5. “Stop this service” releases its process group. “Roll back the active model”
   restores the previous verified selection, or clears it if none existed.

Activation changes the configured store's active pointer. If projects explicitly
share a store, undoing this project's activation may restore the exact previous
model from another project; its identity and evidence are verified again.
Unrelated projects cannot be selected through the report or activation tools.

Activation changes the active pointer. Starting/stopping a service is a separate
operation; rollback does not replace a running server. The service binds to
`127.0.0.1` only, needs Linux/CUDA and uses the pinned serving runtime. Its preview
shows a 2,048-token context, 256-token output default, one concurrent request and
GPU memory utilization 0.3. These conservative defaults are not hardware-optimal
benchmarks; use explicit CLI serving settings for larger contexts or deployment.

The agent owns one server per CLI process. It does not adopt an existing server
or arbitrary PID. Normal process exit stops its server; this is an interactive
local session, not a persistent deployment manager. The run's `agent-serving.log`
contains startup errors. Port conflicts and identity mismatch never establish
readiness. Approval results persist so subsequent turns can use the actual
model and server IDs without asking the user to paste terminal output.

An abrupt OS kill (for example SIGKILL) cannot run application cleanup; this
session manager is not a crash-recovery supervisor.

## Repeating a training experiment

Adapter specs accept `hyperparameters.seed` (an integer from 0 through
4,294,967,295; default 42). The runner seeds before loading the model and
initializing LoRA, then uses the same seed for Trainer and data sampling.
`training-metrics.json` records the effective seed, trainable parameter count
and peak CUDA allocated/reserved bytes. These are PyTorch allocator peaks,
not total device or Spark unified-memory usage. A seed improves repeatability;
it does not guarantee identical GPU kernels across hardware or library versions.
Keep model revision, data hashes, runtime versions and evaluation policy fixed
when comparing backends. Never treat training speed alone as model quality.

For a measured Spark comparison with Unsloth, including startup cost, memory,
quality and repeatability limits, see the [backend review](https://github.com/tunedtensor/tuned-tensor-cli/blob/main/docs/reviews/2026-11-10-unsloth-comparison.md).
