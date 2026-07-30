# tt - Tuned Tensor CLI

`tt` is one terminal for the complete Tuned Tensor workflow:

- hosted specs, datasets, labeling, training, reports, and model downloads;
- local CUDA fine-tuning, held-out evaluation, artifact verification, model
  activation, and OpenAI-compatible serving.

Existing commands such as `tt runs`, `tt models`, and `tt push` remain hosted
commands. The local workflow lives under `tt local`, so scripts keep their
current meaning while people can switch between both targets in one
interactive shell.

## Install

```bash
npm install -g @tuned-tensor/cli
tt --version
```

Node.js 22 or newer is required. Hosted commands need no local ML runtime.
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

After `tt auth login`, run `tt` to open one terminal for conversation and
commands:

```text
tt cloud support-agent › What happened in my latest training run?
TT  Your latest run completed...
tt cloud support-agent › runs list
tt cloud support-agent › Ask whether my dataset is ready to train
```

Ordinary sentences go to the same authenticated Tuned Tensor agent used by the
web application. Known CLI commands such as `runs list`, `doctor`, and
`models list` still execute directly. Prefix a command with `:` when you want
to make that intent explicit. Commands are routed to the mode shown in the
prompt; prefix one with `cloud` or `local` to override the mode without
switching it.

Agent conversations are durable. Use `/new` to start one, `/threads` to list
recent conversations, and `/resume <id>` to continue one. Read operations can
run during the conversation. Mutations remain approval-gated: the agent shows
the exact proposed action, then waits for `/approve` or `/reject`.

Useful shell controls include `/help`, `/status`, `/context`, `/mode`,
`/model`, `/cd`, `/clear`, and `/exit`. `/model` shows the model in play — the
active local model, or the cloud spec's base model — and `/model <id>`
activates a verified local model. The shell keeps normal terminal scrollback
and command history only for the current process.

Explicit commands remain non-interactive, including in CI. `tt --help` shows
the complete command surface, `tt status` inspects both targets without a
network or GPU probe, and `tt shell` opens the conversational terminal
explicitly.

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

Local currently certifies text SFT with `Qwen/Qwen3.5-2B`, LoRA/PEFT, and CUDA.
Evaluation and serving may use CPU. Training artifacts, datasets, and model
weights remain on the execution host.

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
