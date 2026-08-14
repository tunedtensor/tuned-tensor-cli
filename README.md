# tt - Tuned Tensor CLI

`tt` is one terminal for the complete Tuned Tensor workflow:

- hosted specs, datasets, labeling, training, reports, and model downloads;
- local CUDA fine-tuning, held-out evaluation, artifact verification, model
  activation, and OpenAI-compatible serving.

Existing commands such as `tt runs`, `tt models`, and `tt push` remain hosted
commands. The local workflow lives under `tt local`, so scripts keep their
current meaning. In the interactive shell the local workflow is the default;
prefix a command with `cloud` (for example `cloud runs list`) to run one hosted
command without changing the default.

## Install

```bash
npm install -g @tuned-tensor/cli
tt --version
```

Node.js 22.19.0 or newer is required. Hosted commands need no local ML runtime.
Local training additionally needs
[`uv`](https://docs.astral.sh/uv/) and a supported NVIDIA CUDA host; the locked
Python runner ships with the npm dependency and is prepared on first use.

Run from source:

```bash
git clone https://github.com/tunedtensor/tuned-tensor-cli.git
cd tuned-tensor-cli
npm install
npm run build
npm link
```

## Conversational terminal

The interactive agent harness and conversation state now run on your laptop
with Pi. Inference runs through the provider/model you select, which may be a
local endpoint or a remote provider. Tuned Tensor remains an authenticated
typed tool provider: `tt auth login` is used only for Tuned Tensor REST calls
and is never sent to the selected model.
Tool results needed to answer a request are sent to the model you selected, so
choose a local or remote provider whose data policy fits your workload.

Pi reuses provider authentication and custom model definitions from
`~/.pi/agent/auth.json` and `~/.pi/agent/models.json`. Authenticate a provider
with Pi's normal login flow; `tt` deliberately has no provider-secret flags.
Then inspect and select a provider/model:

```bash
tt agent models --all
tt agent configure --provider anthropic --model claude-sonnet-4-5 --thinking high
tt agent status
```

`--thinking` accepts `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, or
`max`.
Selection metadata can be overridden per process with
`TUNED_TENSOR_AGENT_PROVIDER`, `TUNED_TENSOR_AGENT_MODEL`, and
`TUNED_TENSOR_AGENT_THINKING`. These values are not credentials.

After configuration, run `tt` to open one terminal for conversation and
commands:

Before the shell opens, `tt` performs a short, non-blocking npm version check.
If a newer stable release is available it recommends
`npm install -g @tuned-tensor/cli@latest`; offline or unavailable registry
checks are ignored.

```text
tt local support-agent › What happened in my latest training run?
TT  Your latest run completed...
tt local support-agent › runs list
tt local support-agent › Ask whether my dataset is ready to train
```

Ordinary sentences go through the locally orchestrated Pi model session. The
model can call a
strict, bounded set of authenticated Tuned Tensor read tools and prepare-only
mutation tools; it has no shell or general filesystem tool. The shell runs the
local workflow by default, so it can prepare one new folder directly beneath
the shell's current working directory with a validated `tunedtensor.json`. The
tool refuses path traversal,
symlinked workspace roots, existing targets, and unsupported spec fields. Known
CLI commands such as `runs list`, `doctor`, and `models list` still execute
directly. Prefix a command with `:` when you want to make that intent
explicit. Commands run in the local workflow by default; prefix one with
`cloud` (or `local`) to override the target for that one command.

Agent conversations are durable and local under
`~/.config/tuned-tensor/agent/threads` (or the XDG config equivalent), with
user-only directory/file permissions. Persistence normally retains up to 100
recent threads; unsettled safety records are never pruned even when that exceeds
the cap. Each thread is limited to 1 MB, 200 messages, and 200 actions; one-way
action claims are capped at 1,000. Use `/new` to start one, `/threads` to
list recent conversations, and `/resume <id>` to continue one. Older hosted
AgentCore threads are not migrated or listed.

Read operations execute immediately. Creating or updating a behaviour spec is
a prepare-only model operation: the agent shows the exact proposed action, then
waits for `/approve` or `/reject`. Starting or cancelling training remains an
explicit `tt runs ...` command outside the model tool loop.
Local spec creation also waits for `/approve`, never calls the cloud API, requires
at least two examples for local validation, uses exclusive private folder/file
creation, and never overwrites an existing path. Secure filesystem mutation is
currently supported on Linux, where writes are anchored to open workspace and
destination-directory handles. The proposal is bound to the current workspace
so changing directories before approval fails safely instead of writing
somewhere else. Ambiguous write failures remain sealed as `outcome_unknown` for
manual inspection rather than deleting a path that could belong to a racing
writer.
`/approve` executes deterministic local code with an at-most-once mutation
attempt. For hosted mutations, before dispatch the CLI requires the API to
advertise mutation-guard support; older or incompatible servers are refused.
The server conditionally
checks approved spec versions and assigns the action ID as an idempotent create
ID. If a response or final state write is lost after dispatch, the action is
retained as `outcome_unknown`, cannot be retried automatically, and directs the
user to inspect the remote resource. `/reject` never calls a mutation endpoint.

Useful shell controls include `/help`, `/status`, `/context`, `/model`, `/cd`,
`/clear`, and `/exit`. The shell runs the local workflow by default; prefix a
command with `cloud` (for example `cloud runs list`) to run one hosted command
without changing the default.
The shell banner and `/context` surface the assistant's configured provider and
model (`agent provider/model`) separately from the workflow model, so the
model answering your prompts is always visible. `/model` shows the active
local serving model and a preview of available local models; `/model <id>`
activates a verified local serving model. Configure the assistant model
explicitly with `tt agent configure`.
The shell keeps normal terminal scrollback and command history only for the
current process.

Explicit commands remain non-interactive, including in CI. `tt --help` shows
the complete command surface, `tt status` inspects both targets without a
network or GPU probe, and `tt shell` opens the conversational terminal
explicitly.

## Composable pipelines (v1)

A pipeline is an ordered JSON recipe. Version 1 supports `train`, `evaluate`,
and `compare` steps; each resolves to `local` or `cloud`. The v1 evaluator is
`evaluate.with.evaluator: "behavior"`. `evaluate.with.model` is `"base"` or
`{ "from": "step.model" }`; `compare.with.before` and
`after` are `{ "from": "step.report" }`. References must point to an earlier
step and to an output the producer actually exposes. The document contract
itself is defined by the published `@tuned-tensor/pipeline-contract` package;
the CLI adds only execution planning (step selection and cross-target
transfers) on top of it.

```bash
# Generate the canonical four-step recipe for either target.
tt pipeline init --target local --file pipeline.json
tt pipeline init --target cloud --file cloud-pipeline.json

# Validate and show resolved targets plus explicit cross-target transfers.
tt pipeline validate --file pipeline.json
tt --json pipeline plan --file pipeline.json

# Preview any plan without execution, transfer, network calls, or billing.
tt --json pipeline run --dry-run --file pipeline.json --only baseline

# Execute an ordered local plan using the unified CLI's bundled runtime.
tt --json pipeline run --file pipeline.json \
  --spec tunedtensor.json --config local-runner.json
```

`--only` and `--skip` preserve dependency safety: a selected step cannot refer
to an omitted predecessor. Local plans execute directly and may omit, reorder,
or repeat supported components. Cloud and mixed-target plans remain preview-only
until the bounded hosted dispatcher and artifact handoff are deployed; execution
fails closed rather than silently running the fixed cloud workflow. The Pi agent
may describe, validate, and prepare a plan, but has no direct pipeline-execute
tool.

## Hosted quick start

```bash
tt auth login
tt init --name "Customer Support Bot" --model Qwen/Qwen3.5-2B

# Edit tunedtensor.json, then:
tt eval
tt push
tt runs estimate <spec-id>
tt runs start <spec-id>
tt runs watch <run-id>
tt runs report <run-id>
```

The hosted service remains optional. Before a paid run, inspect the estimate:

```bash
tt balance
tt runs estimate <spec-id>
tt runs start <spec-id>
```

## Local quick start

Create a local project on an NVIDIA host:

```bash
mkdir support-adapter && cd support-adapter
tt local init \
  --name "Support Adapter" \
  --model Qwen/Qwen3.5-2B \
  --profile spark
```

Edit the generated `tunedtensor.json`, replacing both placeholder examples,
then preflight and run:

```bash
tt local doctor tunedtensor.json
tt local validate tunedtensor.json
tt local models prefetch tunedtensor.json
tt local run tunedtensor.json
```

Inspect, verify, and serve a completed adapter:

```bash
tt local runs report <run-id>
tt local models verify local-<run-id>
tt local models serve local-<run-id> --config local-runner.json
```

Local currently certifies text SFT with `Qwen/Qwen3.5-2B`,
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`, and
`meta-models/Muse-Glimmer-30B`, using LoRA/PEFT and CUDA. The 30B/3B-active
Nemotron path is intended for DGX Spark-class unified memory, uses activation
checkpointing, and adapts bounded shared attention/Mamba projections rather
than every routed expert matrix. Muse Glimmer is a vision-language checkpoint
whose text tower is fine-tuned through the image-text-to-text loader with the
vision tower frozen and unused. Evaluation and serving may use CPU. Training
artifacts, datasets, and model weights remain on the execution host.

Activation is optional and requires the run to pass a configured
`generalRegression` gate. Once activated, use
`tt local serve active --config local-runner.json`.

### One project file, target-specific payloads

Both workflows use `tunedtensor.json`, but their supported fields are not
identical. `tt` keeps the source file intact and projects only fields accepted
by the selected target:

- hosted pushes omit local-only `hyperparameters` and `dataset_prebuilt`;
- local run, validation, prefetch, and explicit `--spec` serving commands omit
  cloud-only executable `eval_cases`;
- the default hosted scaffold contains two distinct examples and is valid as
  a starting point for local held-out evaluation.

Target validation is still authoritative. In particular, choosing a hosted
base model does not make that model supported by the local CUDA runner.
`tt local serve base` does not automatically inject the adjacent project spec.
Pass `--spec tunedtensor.json` when the server should enforce its instructions;
that explicit spec is projected for the local runtime.

To continue training from a completed fine-tuned model artifact:

```bash
tt runs start <spec-id> --parent-model <model-id>
```

Useful discovery commands:

```bash
tt specs list
tt datasets list
tt runs list
tt runs list --summary --json
tt models list
tt models base
tt balance
```

Use `tt runs list --summary` for agents and scripts that only need run status,
scores, and pagination. It asks the API to omit detailed evaluation and event
payloads; combine it with `--json` for compact structured output.

Label real, unlabeled data with a teacher model (JSONL with `{"input": ...}`
rows, or CSV with `--input-column`; up to 50,000 rows / 50 MB). Labeling runs
as a managed cloud workflow — upload and disconnect; the teacher drafts
outputs under your spec's system prompt and you review before anything trains:

```bash
tt label upload tickets.csv --spec <spec-id> --input-column body --watch
tt label watch <job-id>                     # re-attach to progress any time
tt label rows <job-id> --status labeled     # review the teacher's drafts
tt label accept <job-id> --all              # or accept/reject/edit by row
tt label promote <job-id> --name tickets-v1 # becomes a validated dataset
tt runs start <spec-id> --dataset <dataset-id>
```

Export a model to GGUF and package it for Ollama (so it's pluggable like any
other local model, e.g. in OpenClaw via Ollama's native `/api/chat`):

```bash
# Convert + quantize to GGUF, write a Modelfile, and run `ollama create`
tt models export <model-id> --format gguf --quant q4_k_m --ollama

# Inspect the planned llama.cpp / ollama commands without running them
tt models export <model-id> --quant q8_0 --ollama --print-command
```

This wraps llama.cpp's `convert_hf_to_gguf.py` + `llama-quantize` and Ollama's
`ollama create`. Point `tt` at your llama.cpp checkout with `--llama-cpp <dir>`
(or `--convert-script` / `--quantize-bin`); with `--ollama` the behaviour spec's
system prompt is embedded as the Modelfile `SYSTEM` block.

For the full command reference, including dataset-backed runs, long-example
policies, eval token budgets, preflight run estimates, continued fine-tuning,
evaluation caps, local model serving, configuration, and billing, see the
[CLI docs](https://tunedtensor.com/docs/cli).

## Development

```bash
npm install
npm run build
npm run dev
npm run typecheck
npm test
```

## License

Apache-2.0
