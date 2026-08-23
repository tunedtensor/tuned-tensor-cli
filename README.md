# tt - Tuned Tensor CLI

`tt` is one terminal for local Tuned Tensor workflows:

- a laptop-local conversational agent;
- CUDA fine-tuning, held-out evaluation, artifact verification, model
  activation, and OpenAI-compatible serving on hardware you control.

## Cloud sunset

This release unregisters the hosted API commands (`tt auth`, `tt publish`,
`tt push`, `tt balance`, `tt topup`, `tt cloud`, and the rest of the managed
tree). The modules remain in the repository so they can be restored later.
`tt local …` is kept as a hidden alias; new scripts should call `tt run`,
`tt doctor`, `tt runs list`, and so on directly.

## Migrating from 0.11

`0.13` is the local-only CLI. If you are coming from `0.11.0`:

- Call `tt run`, `tt doctor`, and `tt serve` at the root. `tt local …` still
  works as a hidden alias.
- Hosted account commands (`tt auth`, `tt publish`, `tt push`, and the rest)
  are unregistered.
- `Qwen/Qwen3.5-2B` is pinned to Hugging Face snapshot
  `15852e8c16360a2fea060d615a32b45270f8a8fc`. Prefetch, train, eval, and serve
  reject any other revision.
- `tt serve active` fails unless an adapter is activated. Activation requires
  a `generalRegression` suite in `local-runner.json`.
- `tt info` reports `status: local`.

## Install

On Linux or macOS:

```bash
curl -fsSL https://tunedtensor.com/install.sh | sh
tt --version
```

The installer bootstraps Node.js 22.19+ if needed, installs
`@tuned-tensor/cli` with npm `--ignore-scripts`, and falls back to
`~/.local` when the global npm prefix is not writable. Inspect it first
with `curl -fsSL https://tunedtensor.com/install.sh | less`. Pin a
dist-tag or version on `sh`, not `curl`:

```bash
curl -fsSL https://tunedtensor.com/install.sh | TT_VERSION=beta sh
curl -fsSL https://tunedtensor.com/install.sh | TT_VERSION=0.13.0 sh
```

Or install the package directly:

```bash
npm install -g --ignore-scripts @tuned-tensor/cli
tt --version
```

Uninstall with `npm uninstall -g @tuned-tensor/cli` (add
`--prefix ~/.local` if the curl installer used that prefix).

Node.js 22.19.0 or newer is required. Local training additionally needs
[`uv`](https://docs.astral.sh/uv/) and a supported NVIDIA CUDA host; the locked
Python runner ships with the npm package and is prepared on first use.

Run from source:

```bash
git clone https://github.com/tunedtensor/tuned-tensor-cli.git
cd tuned-tensor-cli
npm install
npm run build
npm link
```

## Conversational terminal

The interactive agent harness and conversation state run on your laptop.
Inference runs through the provider/model you select, which may be a
local endpoint or a remote provider.

Run `tt` to open one terminal for conversation and commands. Workflow commands
such as `runs list` and `doctor` work immediately. Chat waits until you choose
a provider and model from inside the shell:

```text
tt v0.13.1-beta.0
agent not configured · workflow model base
ctrl+c stop/clear · ctrl+d exit · /help commands · tab complete

Use /model to choose a provider and model. Workflow commands work now.
› /model
› /model anthropic
› /model anthropic/claude-sonnet-4-5
```

`tt` stores provider authentication and custom model definitions under
`~/.config/tuned-tensor/agent/auth.json` and
`~/.config/tuned-tensor/agent/models.json` (or the XDG config equivalent).
Provider environment variables such as `ANTHROPIC_API_KEY` still apply.
`tt` deliberately has no provider-secret flags and does not read
`~/.pi/agent/`. `/model` lists providers even when auth is still missing and
marks those entries `auth required`. Selecting a model still needs provider
auth: an environment variable, credentials in `auth.json`, or a placeholder
`apiKey` in `models.json` for local endpoints such as Ollama.

Before the shell opens, `tt` performs a short, non-blocking npm version check.
If a newer stable release is available it recommends
`npm install -g @tuned-tensor/cli@latest`; offline or unavailable registry
checks are ignored.

After a model is selected, the banner shows it and ordinary sentences go to
the agent:

```text
tt v0.13.1-beta.0
agent anthropic/claude-sonnet-4-5 · workflow model base
ctrl+c stop/clear · ctrl+d exit · /help commands · tab complete

Ask TT anything. Known commands run directly.
› What happened in my latest training run?
› runs list
```

Ordinary sentences go through the locally orchestrated model session. The
model has no shell or general filesystem tool. It can prepare one new folder
directly beneath the shell's current working directory with a validated
`tunedtensor.json`. The tool refuses path traversal, symlinked workspace
roots, existing targets, and unsupported spec fields. Known CLI commands such
as `runs list`, `doctor`, and `models list` still execute directly. Prefix a
command with `:` when you want to make that intent explicit.

Agent conversations are durable and local under
`~/.config/tuned-tensor/agent/threads` (or the XDG config equivalent), with
user-only directory/file permissions. Persistence normally retains up to 100
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

Useful shell controls include `/help`, `/status`, `/context`, `/model`, `/cd`,
`/clear`, and `/exit`.
The shell banner and `/context` surface the assistant's configured provider and
model (`agent provider/model`) separately from the workflow model, so the
model answering your prompts is always visible. `/model` shows and changes the
laptop-local TT agent model: run it with no arguments to list providers and a
short model sample, `/model <provider>` to list that provider's models, search
with `/model <query>` (only the closest matches are shown), or switch with
`/model <provider>/<model>` (for example `/model anthropic/claude-sonnet-4-5`).
The equivalent non-interactive commands are `tt agent models`,
`tt agent configure`, and `tt agent status`.
`tt agent configure --thinking` accepts `off`, `minimal`, `low`, `medium`,
`high`, `xhigh`, or `max`. Selection metadata can be overridden per process with
`TUNED_TENSOR_AGENT_PROVIDER`, `TUNED_TENSOR_AGENT_MODEL`, and
`TUNED_TENSOR_AGENT_THINKING`. These values are not credentials.
The shell keeps normal terminal scrollback and command history only for the
current process.

Explicit commands remain non-interactive, including in CI. `tt --help` shows
the command surface, `tt status` inspects local project context without a
network or GPU probe, and `tt shell` opens the conversational terminal
explicitly.

## Composable pipelines (v1)

A pipeline is an ordered JSON recipe. Version 1 supports `train`, `evaluate`,
and `compare` steps. This CLI executes local plans only. The v1 evaluator is
`evaluate.with.evaluator: "behavior"`. `evaluate.with.model` is `"base"` or
`{ "from": "step.model" }`; `compare.with.before` and
`after` are `{ "from": "step.report" }`. References must point to an earlier
step and to an output the producer actually exposes. The document contract
itself is defined by the published `@tuned-tensor/pipeline-contract` package;
the CLI adds only execution planning (step selection) on top of it.

```bash
tt pipeline init --file pipeline.json
tt pipeline validate --file pipeline.json
tt --json pipeline plan --file pipeline.json
tt --json pipeline run --dry-run --file pipeline.json --only baseline
tt --json pipeline run --file pipeline.json \
  --spec tunedtensor.json --config local-runner.json
```

`--only` and `--skip` preserve dependency safety: a selected step cannot refer
to an omitted predecessor. Cloud-targeted steps fail closed unless you pass
`--dry-run`. The TT agent may describe, validate, and prepare a plan, but has
no direct pipeline-execute tool.

## Quick start

Create a local project on an NVIDIA host:

```bash
mkdir support-adapter && cd support-adapter
tt init \
  --name "Support Adapter" \
  --model Qwen/Qwen3.5-2B \
  --profile spark
```

Edit the generated `tunedtensor.json`, replacing both placeholder examples,
then preflight and run:

```bash
tt doctor tunedtensor.json
tt validate tunedtensor.json
tt models prefetch tunedtensor.json
tt run tunedtensor.json
```

Inspect, verify, and serve a completed adapter:

```bash
tt runs report <run-id>
tt models verify local-<run-id>
tt models serve local-<run-id> --config local-runner.json
```

Local currently certifies text SFT with `Qwen/Qwen3.5-2B` (pinned to snapshot
`15852e8c16360a2fea060d615a32b45270f8a8fc`),
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`, and
`meta-models/Muse-Glimmer-30B`, using LoRA/PEFT and CUDA. The 30B/3B-active
Nemotron path is intended for DGX Spark-class unified memory, uses activation
checkpointing, and adapts bounded shared attention/Mamba projections rather
than every routed expert matrix. Muse Glimmer is a vision-language checkpoint
whose text tower is fine-tuned through the image-text-to-text loader with the
vision tower frozen and unused. Evaluation and serving may use CPU. Training
artifacts, datasets, and model weights remain on the execution host.

Activation is optional and requires the run to pass a configured
`generalRegression` gate. Without that suite, `tt models activate` fails
closed. `tt serve active` also fails closed if nothing is activated — it does
not silently serve the protected base model. Once activated, use
`tt serve active --config local-runner.json`.

`tt serve base` does not automatically inject the adjacent project spec.
Pass `--spec tunedtensor.json` when the server should enforce its instructions.

Useful discovery commands:

```bash
tt runs list
tt models list
tt models active
tt status
```

For the full command reference, including long-example policies, eval token
budgets, evaluation caps, and local model serving, see the
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
