# The conversational agent

The interactive agent harness and conversation state run on your laptop.
Inference uses the provider and model you select, which may be a local endpoint
or a remote service: OpenRouter, OpenAI, Anthropic or any other provider in the
catalog, including a local OpenAI-compatible endpoint such as Ollama. Your
requests go directly to that provider. Workflow commands such as
`tt pipeline run`, `tt runs list` and `tt serve` need no provider at all.
Inference choice is independent of where training uses a GPU; your AWS GPU
needs only your AWS credentials and SSH access.

Run `tt` to open the primary conversational workflow. The agent can inspect the
project, derive the canonical pipeline from `tunedtensor.json`, present the
resolved plan, and prepare a sealed dry-run for `/approve`. Workflow commands
such as `runs list` and `doctor` work immediately without any credentials.

For example, with OpenRouter:

```text
› /login openrouter
› /model openrouter/<model-id>
```

`/login` asks which provider, then prompts for that provider's API key with
hidden input. `/login <provider>` skips the provider prompt.
`/model openrouter` lists the bundled catalog, but you can select any OpenRouter model ID, including one
newer than that catalog. OpenRouter validates the selected ID. New IDs use
conservative context/output defaults; custom metadata can be supplied in
`~/.tuned-tensor/agent/models.json`. For a model without reasoning support,
configure it with
`tt agent configure --provider openrouter --model <model-id> --thinking off`.
Inference is billed by your provider, if at all.

Provider keys are stored under TT's local agent config
(`~/.tuned-tensor/agent/auth.json`, readable only by you). Custom endpoints and
models go in `~/.tuned-tensor/agent/models.json`. `tt` does not read
`~/.pi/agent/`. `/model` and `/login` feature OpenAI and OpenRouter. Other
catalog providers remain available by ID. Unauthenticated entries are marked
`auth required`. Local endpoints such as Ollama use a placeholder `apiKey` in
`models.json`.


After a model is selected, the banner shows it and ordinary sentences go to
the agent:

```text
tt v0.15.0
agent openrouter/anthropic/claude-sonnet-5 · workflow model base
tt › What would you like to work on?
/help commands · ctrl+c stop · ctrl+d exit

› What happened in my latest training run?
› runs list
```

Known commands such as `runs list` run directly; tab completes them. Ordinary
sentences go through the locally orchestrated model session. The
model has no shell or general filesystem tool. It can read and propose reviewed
edits to an existing spec, or prepare a validated `tunedtensor.json` in the shell's current working directory
when none exists, or prepare a local pipeline dry-run sealed to the reviewed
workspace and spec contents. Writes and pipeline previews wait for your approval: after the
answer, tt shows the proposal and asks inline, where `y` approves, `n` rejects and `Esc`
leaves it pending for a later `/approve` or `/reject`. Any other key is ignored. The tools refuse path
traversal, symlinked workspace roots, overwriting an existing spec, unsupported spec
fields, cloud pipeline targets, and spec changes after review. Known CLI
commands such as `runs list`, `doctor`, and `models list` still execute directly.
Prefix a command with `:` when you want to make that intent explicit.

The agent can describe the built-in adapter and foundation pipeline stages,
including optional foundation RL. When asked to train or dry-run, it derives or
validates the pipeline, shows the exact plan, and prepares a deterministic
dry-run. `/approve` starts that preview outside the model, and Ctrl-C stops the
child process group. Dry-run approval never starts real training. After a dry-run
of the same spec bytes completes in the conversation, the agent can propose a
background run; a separate approval starts real training in the background.
Review the target and reason before approving; user-owned AWS compute costs
remain your responsibility. Cancellation also requires approval. You can still
run `tt pipeline run` directly. See [background runs](local-runtime/background-runs.md).
The agent can also search public Hugging Face model or dataset metadata for foundation and
fine-tuning discovery. A Hub search sends only the search query to
`huggingface.co`; it uses no Hugging Face token, downloads nothing, and omits
model-card and dataset-card text. Search results are discovery metadata, not a
guarantee that a model or dataset is compatible with a TT workflow. The agent
can inspect a public dataset's exact JSONL/Parquet filenames, sizes, revision,
and declared columns/splits before proposing a download. Declared schema comes
from dataset-card metadata; it is not validation of the file contents.

To reuse a download, ask “Which datasets have we downloaded?” followed by
“Use CodeAlpaca in this spec for fine-tuning.” The agent can list TT-recorded
local datasets and inspect actual Parquet columns/row counts or sampled JSONL
fields, plus two bounded sample rows. It proposes a `dataset_prebuilt` pointer
with the cached revision, training file, separate validation/test file and
column mapping. Approval saves the edit; listing and inspection need no
download or approval. Existing examples and model settings stay in the spec,
but prebuilt data supplies the training rows instead of automatically mixing
in the inline examples. Samples do not establish full dataset quality, and a
download with only a training split still needs distinct held-out data.

When a conversation approaches the model's context budget, TT automatically
summarizes earlier turns using Pi and the selected model, and shows progress.
The trigger reserves the full reply allowance plus 4,096 tokens for protocol
headroom; it does not apply an additional percentage reserve. Usage measured
before a summary is ignored, while new provider usage calibrates the next
trigger, including after resume.
The summary preserves goals, decisions and action outcomes and is saved for
`/resume`; recent exchanges remain in context. Summarization uses an additional
model request (or several for unusually long histories). The transcript stays
in local history subject to the existing storage limits, and pending approvals
remain intact. If the current request is still too large, summarization fails,
or the model stops at its token limit, TT reports the problem instead of silently
returning to the prompt. A failed summary does not replace the saved one.

For educational questions about how or why training works, the agent can read
the exact Python implementation shipped with the running TT build. The source
tool is limited to named foundation components (tokenizer, pretraining, checkpointing, model,
data, SFT, RL, evaluation, and common helpers) and adapter components (training,
data, model contract, and evaluation); it cannot read arbitrary paths. Answers
separate behavior directly visible in the code from inferred rationale and
include the inspected package-relative path and source hash.

Agent conversations are durable and local under
`~/.tuned-tensor/agent/threads` (or `$TUNED_TENSOR_HOME/agent/threads`), with
user-only directory/file permissions. Existing XDG agent state remains in use
until the new agent directory exists. Persistence normally retains up to 100
recent threads; unsettled safety records are never pruned even when that exceeds
the cap. Each thread is limited to 1 MB, 200 messages, and 200 actions; one-way
action claims are capped at 1,000. Use `/new` to start one, `/threads` to
list recent conversations, and `/resume <id>` to continue one.

Local spec creation waits for `/approve`, requires at least two examples,
uses exclusive private folder/file creation, and never overwrites an existing
path. Secure filesystem mutation is currently supported on Linux, where writes
are anchored to open workspace and destination-directory handles. The proposal
is bound to the current workspace so changing directories before approval fails
safely instead of writing somewhere else. Ambiguous write failures remain
sealed as `outcome_unknown` for manual inspection. `/reject` never mutates.
Pipeline dry-runs also wait for `/approve`. Preparation fingerprints the
workspace directory identity, spec, and adjacent config, revalidates them,
copies the resolved spec, config, and pipeline into private temporary inputs,
and invokes `tt pipeline run --dry-run` deterministically. A dry-run proposal
edited to request real execution fails before the child command is dispatched.
After a dry-run of the same spec bytes completes, the agent can propose a real
background run with a reason; approval starts it in a detached process and the
conversation continues. The agent reports progress with read-only tools and can
cancel a run only through another approval. Each run keeps an audit record
(reason, requester, choices, resources, outcome and metrics), a hash-chained event
log and its full log; `TT_MAX_PARALLEL_RUNS` limits concurrent runs. See
[background runs](local-runtime/background-runs.md). The direct
`tt pipeline run` and `tt pipeline start` commands remain available.
A changed workspace or spec requires a new review.


## Shell controls and model selection

Useful shell controls include `/help`, `/status`, `/context`, `/system`,
`/model`, `/login`, `/cd`, `/clear`, and `/exit`.
When the shell opens, the TT logo mark (drawn in the terminal) greets you with the current spec, a
live machine panel (GPU utilization, VRAM and temperature from `nvidia-smi`,
CPU, RAM, and free space on the model-cache filesystem) and whether this machine can
fine-tune the spec's base model, plus a suggested next step. `/system` refreshes
that panel and lists every certified base model's verdict. Fit verdicts come
from the same certified memory profiles as `tt hardware`; a fresh
`tt hardware` snapshot takes precedence over the quick live check.
On macOS, RAM marked `in use (est.)` excludes estimated reclaimable file cache;
decision-training warnings use the same available-RAM estimate. This is not a
memory-pressure measurement or a guarantee that a workload fits. CPU load is a
one-minute average, separate from the sampled utilization percentage. Disk usage
describes the filesystem containing the cache, not the size of downloaded models.
The shell banner and `/context` show the inference selection (`agent
provider/model`) separately from the workflow model. Run `/model` with no
arguments to list suggested models, `/model <provider>` to list that provider's
models, or `/model <query>` to search. Switch with `/model <provider>/<model>`.
The equivalent non-interactive commands are `tt agent models`,
`tt agent configure --provider <provider> --model <model>`, and `tt agent status`.
`tt agent configure --thinking` accepts `off`, `minimal`, `low`, `medium`,
`high`, `xhigh`, or `max`. Selection metadata can be overridden
per process with
`TUNED_TENSOR_AGENT_PROVIDER`, `TUNED_TENSOR_AGENT_MODEL`, and
`TUNED_TENSOR_AGENT_THINKING`. These values are not credentials.
The shell keeps normal terminal scrollback and command history only for the
current process.

Explicit commands remain non-interactive, including in CI. `tt --help` shows
the command surface, `tt status` inspects local project context without a
network or GPU probe, and `tt shell` opens the conversational terminal
explicitly.
