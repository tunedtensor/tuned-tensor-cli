# Work from the behavior spec

Explore the [interactive architecture diagram](spec-architecture.html) for source
excerpts, code structure and command examples. Open the HTML file in a browser.

`tunedtensor.json` is the local recipe for the model's behavior, examples and
training settings. Ask the agent to improve that recipe, review the proposed
diff, approve the edit, then preview or execute a pipeline against the saved spec.
The agent reads the actual file before proposing changes; it does not use the
conversation transcript as a replacement for the current spec.

## Review without a model call

Inside the TT shell, in your project directory:

```text
/spec
/spec validate
/spec diff
/spec history
```

`/spec` shows the complete current file, validation results and SHA-256.
`/spec diff` shows pending spec edits in the current conversation; otherwise it
shows the latest approved edit, or external changes since that revision.
`/spec history` lists up to 50 recent revision records. These commands require
no credentials and work before the agent is configured.

Outside the shell, use `tt spec show`, `tt spec validate`, `tt spec diff`, and
`tt spec history`. Add a workspace-relative path, for example
`tt spec show feedback/tunedtensor.json`. For continued work on a nested
project, `/cd feedback` makes its `tunedtensor.json` the current recipe.
`--json` returns structured review results. `tt spec validate` exits nonzero
for an invalid spec.

## Edit, review, approve

```text
› Keep my training settings, but require a single lowercase sentiment label.
  [agent reads the spec and proposes a diff]
› /spec diff
› /approve
› /spec
› /spec history
```

`get_local_spec` returns the document, validation diagnostics and content hash.
`prepare_update_local_spec` requires that hash and a patch of supported fields.
Arrays replace their whole field; `hyperparameters` and `foundation` merge by
key. Unmentioned fields remain unchanged. Identity and engine cannot change
through this edit tool. The full resulting document must pass the runtime's
strict schema, readiness checks, nonblank content checks and distinct-example
checks. Small example sets produce a warning; validation is not a claim of model
quality or GPU readiness.

Proposing an edit does not write anything. `/approve` revalidates the workspace,
parent directory, current file bytes and proposed result, acquires the spec's
writer lock and atomically replaces the file. `/reject` leaves it unchanged.
If the file changes after review, prepare a fresh edit. A saved edit invalidates
any previously prepared pipeline preview for the old spec.

Secure agent edits currently require Linux filesystem handles. Reads work on
other supported CLI platforms; edit the JSON with your editor there. Spec reads
and edits are bounded to 200 KB. Large model-facing proposals may hit the
agent's existing smaller review budget; split the request or use your editor.

Revision records live beside the spec under `.tuned-tensor/spec-history/` and
contain before/after snapshots and hashes. They can contain your training
examples. They use private permissions. History records changes made through
this edit workflow; it is not a file watcher or a complete history of external
editor changes. A `prepared` record without an `applied` record is an incomplete
journal entry: inspect the current file and its hash before attempting another
edit. A crashed writer can leave `.tunedtensor-spec.lock`; after confirming no
TT writer is still running, remove that lock manually. The lock coordinates TT
writers; avoid concurrent external editor saves during approval.

## Execute the saved recipe

```bash
tt spec validate
tt pipeline run --spec tunedtensor.json --dry-run
tt pipeline run --spec tunedtensor.json
```

The default pipeline derives from the current spec. An adjacent
`tunedtensor.pipeline.json` is no longer loaded implicitly. To select an advanced
recipe explicitly, pass `--file`; a missing explicit file fails. Foundation
training parameters in an explicit recipe must agree with the spec, including
in previews and the runtime's own execution guard. Change the spec and regenerate
the recipe when they disagree. Explicit recipes may still select supported
stages; `--only` and `--skip` remain available.

CLI planning, validation, execution and agent pipeline proposals share
`pipelineForRunInput` in `src/pipeline.ts`. It uses the same
`validateBehaviorSpec` rules as spec review and editing, defined beside the
runtime parser in `src/local-runtime/local-project.ts`. The edit module only
owns workspace access, patches and revision history.

The command reads one spec snapshot for planning, hardware warnings and execution. Plan/dry-run
JSON and final result JSON identify its path and SHA-256. The spec's compiled
instructions and examples feed training and task evaluation; the independently
configured general-regression suite keeps its own prompt. Provider choice and
GPU placement remain separate runtime choices in agent configuration and
`local-runner.json`. Evaluation policy (inference/scoring settings and the
general-regression suite) also remains in the runner config. Moving that policy
into the spec would be a separate schema change.

Agent `/approve` for a pipeline still runs a preview only. Real training uses
the direct command above. The spec is the behavior recipe; approval of a spec
edit is not approval to launch training.
