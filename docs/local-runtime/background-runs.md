# Background runs

A background run is a real `tt pipeline run` that keeps going after the shell or
conversation that started it moves on. Start one from the TT agent (after an
approved dry-run) or directly:

```bash
tt pipeline run --spec tunedtensor.json --dry-run
tt pipeline start --spec tunedtensor.json --reason "Check the new negation examples"
tt pipeline list
tt pipeline status <id>
tt pipeline logs <id>
tt pipeline cancel <id> --reason "Started on the wrong dataset"
```

IDs accept a unique prefix. `--json` prints machine-readable output for every
command.

## From the agent

The agent can prepare, but never start, a run. The flow is:

1. `prepare_pipeline_run`: you approve a dry-run preview of the saved spec.
2. `prepare_background_run`: the agent proposes a real run of **the same spec
   bytes** with a short reason. The proposal shows the plan, the target (laptop
   or AWS instance), active runs, the parallel-run limit and current CPU, memory
   and GPU usage. It is refused if no matching dry-run completed in the
   conversation, if the spec changed, or if the parallel limit is reached.
3. On approval, tt revalidates the spec, writes the run record and starts a
   detached supervisor. The conversation continues immediately.

While a run is active, the agent answers questions with read-only tools:
`list_background_runs`, `get_background_run` (status, metrics and the recent
log) and `get_system_usage` (CPU, memory, per-GPU load and active runs). It can
only stop a run through `prepare_cancel_background_run`, which you approve. When
a run started from a conversation finishes, the agent is told its outcome on your
next message; it cannot poll in the background.

## Isolation

The run executes the sealed spec captured at approval, not the live
`tunedtensor.json`, so later spec edits in the conversation cannot change it.
The snapshot sits beside the spec while the run is active (as
`.tunedtensor.json.tt-agent-<id>.json`), so relative paths, output directories
and `tunedtensor.lock.json` behave exactly as for a direct run. It is removed
when the run ends. The supervisor runs in its own process group, so closing tt
or the terminal does not stop the run; use `tt pipeline cancel`.

## Parallel runs

By default one background run may be active at a time. Set
`TT_MAX_PARALLEL_RUNS` (1–8) to allow more. Runs that share a target (the
laptop GPU, or the same AWS instance) share its GPU memory: check
`get_system_usage`, `nvidia-smi` or `tt doctor` first. The proposal warns when a
new run would share a target with an active one. Laptop usage figures do not
describe a remote AWS GPU.

## Audit record

Each run keeps a private directory under `~/.tuned-tensor/background-runs/<id>/`:

| File | Contents |
| --- | --- |
| `record.json` | Reason; requester (OS user, host, `agent` or `cli`, conversation, action and dry-run action IDs); approval time; choices (engine, spec path and SHA-256, config, steps, target, base model); resources at launch; process IDs; cancellation reason and requester; outcome (exit code, local run ID, model ID, report path, evaluation metrics, error). |
| `spec.json` | The sealed spec the run consumed. |
| `events.jsonl` | Hash-chained lifecycle events: requested, started, cancel requested, finished or lost. `tt pipeline status` verifies the chain. |
| `output.log` | Progress and process logs. |
| `result.json` | The pipeline's final JSON result. |

Adapter runs also keep their usual record in the local run store; use
`tt runs audit <run-id>`, `tt runs report <run-id>` and
`tt runs compare <a> <b>` with the run ID from the outcome. The hash chain makes
casual edits evident; it is not a signature.

If the supervisor disappears (crash, reboot), the run is reported as `lost` the
next time it is read. Inspect its log and the local run store before starting
again; a foundation run can be resumed with `tt pipeline run --resume`.
