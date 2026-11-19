# tt — Tuned Tensor CLI

**Fine-tune open models on hardware you control.** Describe the behavior you
want in one file, and `tt` prepares the data, trains, compares the tuned model
with its base, and serves it — from your own terminal.

```bash
tt init --name "Support triage" --model Qwen/Qwen3.5-2B   # 1. describe the behavior
tt pipeline run --spec tunedtensor.json                   # 2. train and evaluate
tt runs report <run-id>                                   # 3. see if it got better
tt serve local-<run-id>                                   # 4. serve it
```

Free and open source (Apache-2.0). No account, sign-in, hosted service or usage fee.

## Why use tt?

- **One file is the source of truth.** `tunedtensor.json` holds the system
  prompt, rules, examples and training settings. Edit it, rerun, compare.
- **Proof, not vibes.** Adapter and decision runs evaluate the tuned model
  against its base model on held-out examples and report the deltas and
  regressions, so you know whether tuning helped.
- **Your data stays yours.** Specs, datasets, runs and weights stay on your
  machines. Train on a local NVIDIA GPU or an existing GPU instance in your own
  AWS account. Small decision models also train on CPU or Apple Silicon.
- **Reproducible and auditable.** Base models and datasets are pinned to exact
  commits in a lock file, and each run keeps a tamper-evident record of what
  produced it.
- **An optional AI assistant.** Run `tt` to chat with an agent that edits your
  spec, finds Hugging Face models and datasets, and prepares runs. Nothing is
  written or started until you approve it. Bring your own model provider
  (OpenRouter, OpenAI, Anthropic, or a local endpoint such as Ollama).

## What can you train?

| Engine | What it does | Runs on |
| --- | --- | --- |
| **Adapter** (default) | LoRA fine-tuning of an open chat model, such as `Qwen/Qwen3.5-2B` | NVIDIA CUDA, local or AWS |
| **Decision** | A small classifier-style model that picks a label, answers yes/no or gives a score, with a probability per label | CPU, Apple MPS or CUDA |
| **Foundation** | A small GPT and tokenizer trained from scratch, for learning and experiments | NVIDIA CUDA, local or AWS |

## Install

**macOS or Linux:**

```bash
curl -fsSL https://tunedtensor.com/install.sh | sh
```

**Windows (PowerShell), or anywhere with Node.js 22.19+:**

```bash
npm install -g --ignore-scripts @tuned-tensor/cli
```

On Windows, use `npm.cmd` and `tt.cmd` in PowerShell. Check the install with
`tt --version`; upgrade later with `tt upgrade`.

Training also needs [`uv`](https://docs.astral.sh/uv/). LLM training needs an
NVIDIA GPU, either on this machine or [in your AWS account](docs/local-runtime/aws-gpu.md).
See [install options](docs/install.md) and the [Windows guide](docs/windows.md).

## Quick start

**1. Create a spec** in a new folder:

```bash
mkdir support-triage && cd support-triage
tt init --name "Support triage" --model Qwen/Qwen3.5-2B
```

**2. Describe the behavior.** Open `tunedtensor.json` and replace the
placeholder prompt and examples with your own (at least two):

```json
{
  "name": "Support triage",
  "base_model": "Qwen/Qwen3.5-2B",
  "system_prompt": "Classify the customer's message. Reply with one label.",
  "guidelines": ["Reply with billing, bug or other."],
  "examples": [
    { "input": "I was charged twice this month.", "output": "billing" },
    { "input": "The app crashes when I upload a photo.", "output": "bug" }
  ]
}
```

**3. Check your setup and preview the run.** The dry run needs no GPU:

```bash
tt validate tunedtensor.json
tt doctor tunedtensor.json
tt pipeline run --spec tunedtensor.json --dry-run
```

**4. Train and see the results:**

```bash
tt pipeline run --spec tunedtensor.json
tt runs list
tt runs report <run-id>
```

## Examples

### Chat with the agent

```bash
tt                       # open the conversational terminal
› /login openrouter      # enter your OpenRouter API key
› /model openrouter/<model-id>
› Add five more billing examples to my spec
› Dry-run the pipeline
```

The agent shows each proposed change or run, and you press `y` to approve or
`n` to reject. Commands such as `runs list` work without a provider.
See [the conversational agent](docs/agent.md).

### Train on a Hugging Face dataset

Point the spec at dataset files instead of inline examples:

```json
{
  "dataset_prebuilt": {
    "huggingface": {
      "repo": "SetFit/tweet_sentiment_extraction",
      "revision": "0d5359cfbb2470332f2b82e62269cf755e0ac5c3",
      "columns": { "input": "text", "output": "label_text" }
    },
    "training": "train.jsonl",
    "test": "test.jsonl"
  }
}
```

```bash
tt datasets prefetch tunedtensor.json   # download, resumable; pins the lock file
tt pipeline run --spec tunedtensor.json
```

See [datasets and downloads](docs/datasets.md), including splitting your own
JSONL data.

### Train a decision model on your laptop

```bash
tt init --name "Ticket router" --engine decision
# Edit tunedtensor.json: the question, the labels, and examples whose outputs are labels
tt pipeline run --spec tunedtensor.json
```

This trains on CPU or Apple Silicon, no NVIDIA GPU needed. See [decision models](docs/local-runtime/decision.md) and the
[card-triage example](examples/single-spec/decision/tunedtensor.json).

### Train on your own AWS GPU

Add a `runtime` section to `tunedtensor.json`, keep running `tt` on your laptop,
and the GPU work runs on your instance over SSH:

```json
{
  "runtime": {
    "gpu": {
      "provider": "aws",
      "profile": "research",
      "region": "eu-west-1",
      "instanceId": "i-0123456789abcdef0",
      "user": "ubuntu",
      "identityFile": "~/.ssh/research-gpu.pem"
    }
  }
}
```

The instance must already be running. New to this? Follow
[your first AWS run](docs/local-runtime/aws-gpu.md).

### Serve your model

```bash
tt serve local-<run-id>     # OpenAI-compatible server (Linux + NVIDIA GPU)
```

See [runs, audit and serving](docs/runs-and-serving.md).

## Everyday commands

| Command | What it does |
| --- | --- |
| `tt` | Open the conversational terminal |
| `tt init` | Create a `tunedtensor.json` spec |
| `tt spec view` | Show the spec in a readable form |
| `tt hardware` | Show what this machine can train or serve |
| `tt doctor tunedtensor.json` | Check the machine (or AWS GPU) before a run |
| `tt pipeline run --spec tunedtensor.json` | Train and evaluate (`--dry-run` to preview) |
| `tt runs list` / `tt runs report <id>` | List runs and compare tuned vs base |
| `tt runs audit <id>` | Verify a run's record has not been edited |
| `tt models list` | List trained models |
| `tt serve <model>` | Serve a model |
| `tt upgrade` | Update tt |

Run `tt --help` or `tt <command> --help` for everything else.

## Privacy and network use

Your specs, data, runs and model weights stay on your machines. Network calls
include Hugging Face downloads, SSH to your own GPU instance, and the model
provider you choose for the optional agent; agent requests go directly to that
provider. `tt` also checks the public npm registry for updates at most every
12 hours; set `TT_NO_UPDATE_CHECK=1` to turn that off.

## Guides

- [Install, upgrade and uninstall](docs/install.md)
- [Work from the behavior spec](docs/spec-workflow.md)
- [The conversational agent](docs/agent.md)
- [Hardware and platform support](docs/hardware.md)
- [Your first AWS run](docs/local-runtime/aws-gpu.md)
- [Datasets and downloads](docs/datasets.md)
- [Composable pipelines](docs/pipelines.md)
- [Runs, audit and serving](docs/runs-and-serving.md)
- [Decision models](docs/local-runtime/decision.md)
- [Background runs](docs/local-runtime/background-runs.md)
- [Windows and PowerShell](docs/windows.md)

## Development

```bash
npm install
npm run build
npm run dev
npm run typecheck
npm test
npm run test:workflows  # Focused product workflow regressions; no model key or GPU
npm run check           # Typecheck, all tests, and build (the release gate)
```

The npm package bundles the agent, pipeline contract, and Zod dependency trees
so the security overrides also reach installed CLI users. npm does not apply a
dependency's own overrides. CI audits a fresh installation of the packed CLI;
keep that check when changing dependency pins or packaging.

See [testing and workflow contracts](docs/testing.md) for what the
suite guarantees, how to evaluate a real agent with `npm run eval:agent`, and
the remaining steps toward the full conversational product promise.

## Contributing

Tuned Tensor is community-driven. Bug reports, model and hardware compatibility
results, documentation fixes and pull requests are welcome at
[github.com/tunedtensor/tuned-tensor-cli](https://github.com/tunedtensor/tuned-tensor-cli).
Run `npm run check` before opening a pull request.

## License

Apache-2.0
