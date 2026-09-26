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
`add_examples` appends examples without resending existing ones; other arrays
replace their whole field, and the diff lists only removed and added items.
`hyperparameters` and `foundation` merge by
key. `runtime` and `evaluation` merge recursively; null removes optional settings.
`pipeline` replaces the entire recipe (null restores the derived default). Unmentioned fields remain unchanged. Identity and engine cannot change
through this edit tool. An adapter spec without an `id` gets one on its first
edit, pinned to the ID its earlier runs used, so later edits keep the same run
identity and evaluation split. Specs created by the agent start with an `id`. The full resulting document must pass the runtime's
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

The pipeline uses the optional `pipeline` section or derives from the current spec. An adjacent
`tunedtensor.pipeline.json` is no longer loaded implicitly. To select an advanced
recipe explicitly, pass `--file`; a missing explicit recipe or `--spec` file fails. Foundation
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
GPU placement is configured in `runtime.gpu` in the spec. Model-provider choice
remains in agent configuration. Evaluation policy belongs in `evaluation`;
artifact/store locations and model paths belong in `runtime`. All relative paths
resolve beside the spec. New projects default to `.tuned-tensor/artifacts` and
`.tuned-tensor/store` beside it (unless `TT_LOCAL_HOME` overrides the store).
Foundation supports `runtime.artifactRoot` and `runtime.gpu`; its evaluation
and training settings remain under `foundation`.

Only `tunedtensor.json` is required. `tt pipeline init --spec tunedtensor.json`
embeds an advanced recipe when needed; omit this step to keep automatic derivation.
An explicit `--file` cannot override a conflicting embedded recipe.

For existing projects, run `tt pipeline migrate --spec tunedtensor.json` to
consolidate adjacent pipeline and runner files. It validates the result first,
keeps exclusive `.bak` copies, and preserves resolved paths and the old store
location. Conflicting sources, enabled legacy `dryRun`, or unsupported foundation
runner settings require reconciliation first. Existing `.bak` files are never
overwritten. Avoid concurrent edits during migration. Legacy runner files remain
readable with a warning, but cannot coexist with inline runtime/evaluation settings.
A legacy adjacent pipeline is used only with explicit `--file` or after migration.

Runs write generated `resolved-workflow.json` snapshots for diagnosis and
reproducibility; these are outputs, not additional user-authored configuration.


Agent `/approve` for a pipeline still runs a preview only. Real training uses
the direct command above. The spec is the behavior recipe; approval of a spec
edit is not approval to launch training.
