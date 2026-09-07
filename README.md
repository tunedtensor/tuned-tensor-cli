# tt - Tuned Tensor CLI

`tt` is the local agent and CLI for Tuned Tensor. Use it to converse, run
workflows on your laptop or in the cloud, inspect metrics, and check account
usage. The web app is the dashboard for reviewing cloud runs and published
local evidence.

Local training, evaluation, inspection, and serving need no TT access token.
Agent inference is a separate choice:

- **Managed:** sign in once with `tt auth login`. TT reuses your access token
  through the Tuned Tensor model proxy. The server selects the model; no
  OpenRouter key or model selection is needed.
- **Bring your own:** use your own OpenRouter or other provider credentials
  and choose the model. Your requests go directly to that provider. TT does
  not apply its managed model policy to BYO requests.

A saved BYO selection stays selected after TT login. Switch back with
`tt agent configure --provider tunedtensor` or `/model tunedtensor/managed`.
Inference choice is independent of whether a training workflow runs locally or
in the cloud. Cloud account operations always require a TT access token.

## Unified commands

Local workflows remain the default: `tt pipeline run`, `tt runs list`,
`tt models list`, and `tt serve`. Cloud workflows are explicit:
`tt cloud runs start`, `tt cloud runs list`, and `tt cloud models list`.
`tt usage` reports managed agent allowance and usage; `tt balance` reports
cloud training credits. `tt publish` uploads local run evidence to the dashboard.

TT includes the former TT Local runtime. `tt local …` and `tt run` remain
hidden compatibility aliases; new scripts should use root commands and
`tt pipeline run --spec tunedtensor.json`. Existing token-free local projects
and saved BYO provider settings continue to work. Cloud operations previously
hidden in the local-only release are available under `tt cloud`.

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
Inference uses the TT managed model or a BYO provider you select. A BYO
provider may be a local endpoint or a remote service.

Run `tt` to open the primary conversational workflow. The agent can inspect the
project, derive the canonical pipeline from `tunedtensor.json`, present the
resolved plan, and prepare a sealed dry-run for `/approve`. Workflow commands
such as `runs list` and `doctor` work immediately without any credentials.

For managed inference, save your TT access token with hidden input:

```bash
tt auth login
tt
```

Get a token from the dashboard's **Settings → API Keys** page. You can also
set `TUNED_TENSOR_API_KEY` in CI. When there is no saved BYO selection, TT
selects `tunedtensor/managed` automatically. In the shell, `/login tunedtensor`
saves the same account token. Managed inference requires no OpenRouter key
or additional login. The server controls the model and request limits.
`tt usage` shows the configured daily allowance and reported usage.

For BYO OpenRouter inference:

```text
› /login openrouter
› /model openrouter/<model-id>
```

`/login` asks which provider, then prompts for the TT access token or BYO
provider key with hidden input. `/login <provider>` skips the provider prompt.
`/model openrouter` lists the bundled catalog, but you can select any OpenRouter model ID, including one
newer than that catalog. OpenRouter validates the selected ID. New IDs use
conservative context/output defaults; custom metadata can be supplied in
`~/.tuned-tensor/agent/models.json`. For a model without reasoning support,
configure it with
`tt agent configure --provider openrouter --model <model-id> --thinking off`.
BYO inference is billed by your provider.

Provider keys are stored under TT's local agent config, separately from the
account token. The managed provider never stores your TT token in provider
files and does not accept endpoint/model overrides in `models.json`; use a
separate BYO provider for custom endpoints. `tt` does not read `~/.pi/agent/`.
`/model` and `/login` feature Tuned Tensor, OpenAI, and OpenRouter. Other
catalog providers remain available by ID. Unauthenticated entries are marked
`auth required`. Local endpoints such as Ollama use a placeholder `apiKey` in
`models.json` and need no TT account.

Before the shell opens, `tt` performs a short, non-blocking npm version check.
If a newer stable release is available it recommends
`npm install -g @tuned-tensor/cli@latest`; offline or unavailable registry
checks are ignored.

After a model is selected, the banner shows it and ordinary sentences go to
the agent:

```text
tt v0.15.0
agent tunedtensor/managed · workflow model base
ctrl+c stop/clear · ctrl+d exit · /help commands · tab complete

Ask TT anything. Known commands run directly.
› What happened in my latest training run?
› runs list
```

Ordinary sentences go through the locally orchestrated model session. The
model has no shell or general filesystem tool. It can prepare one new folder
directly beneath the shell's current working directory with a validated
`tunedtensor.json`, or prepare a local pipeline dry-run sealed to the reviewed
workspace and spec contents. Neither action starts until `/approve`. The tools refuse path
traversal, symlinked workspace roots, existing spec targets, unsupported spec
fields, cloud pipeline targets, and spec changes after review. Known CLI
commands such as `runs list`, `doctor`, and `models list` still execute directly.
Prefix a command with `:` when you want to make that intent explicit.

The agent can describe the built-in adapter and foundation pipeline stages,
including optional foundation RL. When asked to train or dry-run, it derives or
validates the pipeline, shows the exact plan, and prepares a deterministic
dry-run. `/approve` starts that preview outside the model, and Ctrl-C stops the
child process group. Real training is intentionally not model-mediated: run the
displayed `tt pipeline run` command directly when you choose to execute. It can also
search public Hugging Face model or dataset metadata for foundation and
fine-tuning discovery. A Hub search sends only the search query to
`huggingface.co`; it uses no Hugging Face token, downloads nothing, and omits
model-card and dataset-card text. Search results are discovery metadata, not a
guarantee that a model or dataset is compatible with a TT workflow.

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
and invokes `tt pipeline run --dry-run` deterministically. Persisted proposals
that request real execution fail before the child command is dispatched. Real
training remains available through the explicit direct `tt pipeline run` command.
A changed workspace or spec requires a new review.

Useful shell controls include `/help`, `/status`, `/context`, `/model`,
`/login`, `/cd`, `/clear`, and `/exit`.
The shell banner and `/context` show the inference selection (`agent
provider/model`) separately from the workflow model. The managed alias identifies
inference through the server-selected model. Run `/model` with no arguments to
list the managed option and suggested BYO models, `/model <provider>` to list
that provider's models, or `/model <query>` to search. Switch with
`/model <provider>/<model>`, including `/model tunedtensor/managed` to return
to managed inference.
For managed inference, `tt agent configure --provider tunedtensor` needs no
model argument. For BYO inference, both provider and model are required.
The equivalent non-interactive commands are `tt agent models`,
`tt agent configure`, and `tt agent status`.
For BYO models, `tt agent configure --thinking` accepts `off`, `minimal`, `low`, `medium`,
`high`, `xhigh`, or `max`; managed inference uses the server's reasoning policy
and leaves the CLI thinking option off. Selection metadata can be overridden
per process with
`TUNED_TENSOR_AGENT_PROVIDER`, `TUNED_TENSOR_AGENT_MODEL`, and
`TUNED_TENSOR_AGENT_THINKING`. These values are not credentials.
The shell keeps normal terminal scrollback and command history only for the
current process.

Explicit commands remain non-interactive, including in CI. `tt --help` shows
the command surface, `tt status` inspects local project context without a
network or GPU probe, and `tt shell` opens the conversational terminal
explicitly.

## Composable pipelines (v1)

A pipeline is an ordered JSON recipe. This CLI executes local plans only. The
portable document contract lives in `@tuned-tensor/pipeline-contract`. The CLI
adds execution planning (step selection) on top of it.

Two local engines share that document version:

- **Adapter** (default): LoRA SFT on a certified Hugging Face checkpoint.
  Runs through `tt pipeline run`. Uses `train` / `evaluate` / `compare` with
  `evaluate.with.evaluator: "behavior"`.
- **Foundation**: from-scratch tokenizer + GPT. Also runs through
  `tt pipeline run`. Uses
  `tokenize` / `pretrain` / `finetune` / optional `rl` plus `bpb`, `chat`, and
  `inference` evaluators. Requires a foundation `tunedtensor.json`
  (`engine: "foundation"`, at least two examples, no `base_model`).

Both engines are spec-driven. `system_prompt`, `guidelines`, and `constraints`
are compiled into one canonical system instruction, and `examples` supply the
demonstrated behavior. Adapter training, foundation tokenizer/pretraining,
chat SFT, optional RL, and evaluation all receive that same compiled
instruction. Engine-specific fields choose how TT learns the behavior; they do
not redefine the behavior itself.

The foundation engine is a readable, single-GPU local trainer inspired by Andrej
Karpathy's [nanochat](https://github.com/karpathy/nanochat), not a port of its
distributed training stack. With `foundation.corpus_path`, it streams `.txt`
or JSONL (`{"text":"..."}`) shards into a deterministic on-disk token cache;
without that field it retains the bounded prompt/example corpus for smoke runs.
`foundation.validation_path` supplies a distinct held-out BPB corpus. Foundation
plans reject `compare` before creating run artifacts; comparison remains an
adapter-engine capability. Optional RL samples one seeded on-policy completion
per step with a sparse last-number exact reward and therefore requires a numeric
expected output for every example. Treat those metrics as an end-to-end
execution and overfit check—not held-out capability or multi-GPU evidence when
no validation corpus is configured.

```bash
tt pipeline init --file pipeline.json
tt pipeline init --engine foundation --file foundation.pipeline.json
tt pipeline validate --file pipeline.json
tt --json pipeline plan --file pipeline.json
tt --json pipeline run --dry-run --file pipeline.json --only baseline
tt --json pipeline run --file pipeline.json \
  --spec tunedtensor.json --config local-runner.json
tt init --engine foundation --name "Tiny GPT"
tt --json pipeline run --spec tunedtensor.json --dry-run
tt pipeline run --spec tunedtensor.json --resume /absolute/path/to/foundation-run
```

When `--config` is omitted, `pipeline run` uses `local-runner.json` beside the
selected spec when that file exists.

`--only` and `--skip` preserve dependency safety: a selected step cannot refer
to an omitted predecessor. Cloud-targeted pipeline steps can be inspected with `--dry-run`; actual
cloud execution uses `tt cloud runs`, described below. Foundation documents are local-only. The TT agent may describe,
validate, and prepare a sealed pipeline action; only deterministic `/approve`
handling can execute it.

Long pretraining uses BF16, warmup plus cosine decay, gradient clipping,
periodic metrics, and atomic rolling checkpoints. `--resume <run-directory>`
starts or resumes one stable run identity; completed stages are verified and
skipped, while an interrupted pretrain step restores model, optimizer,
scheduler, RNG, counters, and token cursor. See
[Long foundation runs](docs/local-runtime/foundation-long-runs.md) before an
unattended job.

## Cloud runs and account reporting

Use the same TT token for cloud operations and managed inference. A BYO model
selection continues to work while you operate cloud runs with that token.

```bash
tt auth login
tt cloud push --file tunedtensor.json
tt cloud runs estimate <spec-id>
tt cloud runs start <spec-id>
tt cloud runs watch <run-id>
tt cloud runs report <run-id>
tt cloud runs list
tt cloud models list
tt usage
tt balance
```

Use the spec ID returned by `push`. `tt cloud specs`, `tt cloud datasets`,
`tt cloud label`, and `tt cloud models` expose the corresponding account
operations; each command's `--help` documents its arguments. Cloud foundation
training and dispatching cloud-targeted pipeline JSON are not supported;
`tt cloud runs` uses the hosted adapter workflow.

With a TT token, the local agent can inspect cloud resources, run reports,
balances, transactions, and managed usage. It can prepare cloud spec edits for
`/approve`. Those proposals show the API origin and remain bound to the token
and origin used during preparation. After changing either, return to the
original account and origin or prepare a new proposal. Starting or cancelling
cloud training remains an explicit direct
CLI command. Without a TT token these account tools are unavailable, while
local workflow commands and BYO inference continue to work.

`tt usage` reports the server-configured daily managed inference allowance.
Each provider completion, including a tool follow-up, consumes one request;
failed and cancelled requests also count. Token/cost coverage shows which
requests have provider usage reports, so incomplete reporting is not mistaken
for zero consumption. The displayed provider cost is informational, not a
charge to cloud training credits. BYO provider usage is separate. `tt balance`
shows cloud training credits and recent transactions; `tt topup` opens checkout.

Publish a local report for review in the dashboard:

```bash
tt publish <local-run-id> --dry-run
tt publish <local-run-id>
tt cloud runs archive <published-run-id>
```

Publishing shows the upload before confirmation and includes the spec, metrics,
prompts, and example model outputs. Archiving is supported for inactive local
reports published to the dashboard. Local model weights stay on your machine.
Use `--json` with reporting commands for scripts. `TUNED_TENSOR_URL` or
`--base-url` selects an alternate Tuned Tensor API deployment.

## Quick start

Create a local project on an NVIDIA host:

```bash
mkdir support-adapter && cd support-adapter
tt hardware
tt init \
  --name "Support Adapter" \
  --model Qwen/Qwen3.5-2B \
  --profile spark
```

`tt hardware` reports which certified models this GPU can train, LoRA
fine-tune, or serve. The shell agent can run the same inventory via its
`examine_hardware` tool. Edit the generated `tunedtensor.json`, replacing both
placeholder examples, then preflight and run:

```bash
tt doctor tunedtensor.json
tt validate tunedtensor.json
tt models prefetch tunedtensor.json
tt pipeline run --spec tunedtensor.json
```

Inspect, verify, and serve a completed adapter:

```bash
tt runs report <run-id>
tt models verify local-<run-id>
tt serve local-<run-id> --config local-runner.json
```

Local currently certifies text SFT with `Qwen/Qwen3.5-2B` (pinned to snapshot
`15852e8c16360a2fea060d615a32b45270f8a8fc`),
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`, and
`meta-models/Muse-Glimmer-30B`, using LoRA/PEFT and CUDA. The 30B/3B-active
Nemotron path is intended for DGX Spark-class unified memory, uses activation
checkpointing, and adapts bounded shared attention/Mamba projections rather
than every routed expert matrix. Muse Glimmer is a vision-language checkpoint
whose text tower is fine-tuned through the image-text-to-text loader with the
vision tower frozen and unused. Evaluation may use CPU; packaged local
serving requires Linux and NVIDIA CUDA. Training
artifacts, datasets, and model weights remain on the execution host.

Activation is optional and requires the run to pass a configured
`generalRegression` gate. Without that suite, `tt models activate` fails
closed. `tt serve active` also fails closed if nothing is activated — it does
not silently serve the protected base model. Once activated, use
`tt serve active --config local-runner.json`.

`tt serve base` does not automatically inject the adjacent project spec.
Pass `--spec tunedtensor.json` when the server should enforce its instructions.
Serving uses a separate locked **vLLM** runtime on Linux/CUDA. A tuned launch
exposes both the adapter and its base model, sharing weights, with upstream
streaming and native tool-call parsing. `tt serve <target> --print-client-config pi`
exports isolated Pi configuration; tools no longer need to be disabled. See
[Local serving and coding harnesses](docs/local-runtime/serving.md) for Pi/OpenCode
setup, resource budgets, hardware limits, and the verified integration scope.

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
