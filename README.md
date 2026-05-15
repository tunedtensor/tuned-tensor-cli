# tt - Tuned Tensor CLI

`tt` is the command-line tool for [Tuned Tensor](https://www.tunedtensor.com), used to define behavior specs, validate them, and launch fine-tuning runs.

## Install

```bash
npm install -g @tuned-tensor/cli
tt --version
```

Run from source:

```bash
git clone https://github.com/tuned-tensor/tuned-tensor-cli.git
cd tuned-tensor-cli
npm install
npm run build
npm link
```

## Quick Start

1) **Authenticate**

```bash
tt auth login
tt auth status
```

2) **Create a local spec**

```bash
tt init
# or:
tt init --name "Customer Support Bot" --model "Qwen/Qwen3.5-2B"
```

Supported spec base models are `Qwen/Qwen3.5-2B`, `google/gemma-4-E2B-it`, and `google/gemma-4-26B-A4B-it`.

3) **Validate your spec**

```bash
tt eval
```

4) **Push your spec**

```bash
tt push
```

5) **Start and watch a run**

```bash
tt runs start <spec-id>
tt runs start <spec-id> --dataset <dataset-id-or-prefix> --train-ratio 0.8 --validation-ratio 0.1 --test-ratio 0.1
tt runs start <spec-id> --no-llm-judge
tt runs watch <run-id>
```

Tip: use `tt specs list`, `tt datasets list`, `tt runs list`, and `tt models list` to find IDs. Spec, run, and dataset commands accept full UUIDs or unambiguous ID prefixes.

## Typical Workflows

```bash
# Account
tt auth status
tt balance
tt topup --amount 25

# Specs
tt specs list
tt specs get <spec-id>
tt specs create --file spec.json
tt specs update <spec-id> --file updates.json

# Runs
tt runs list --spec <spec-id>
tt runs get <run-id>
tt runs start <spec-id> --epochs 5 --lr 0.0001 --batch-size 8
tt runs start <spec-id> --dataset <dataset-id-or-prefix> --train-ratio 0.8 --validation-ratio 0.1 --test-ratio 0.1
tt runs start <spec-id> --no-llm-judge
tt runs cancel <run-id>

# Datasets
tt datasets upload data.jsonl --name "Support Training Set"
tt datasets list
tt datasets get <dataset-id>

# Models
tt models list
tt models get <model-id>
tt models download <model-id> --output model.tar.gz
```

Use `--dataset <dataset-id-or-prefix>` with `tt runs start` to train from an uploaded dataset instead of inline spec examples. Add `--train-ratio`, `--validation-ratio`, and `--test-ratio` to override the default 80/10/10 split.

Use `--max-eval-examples <n>` and `--max-test-eval-examples <n>` with `tt runs start` to cap primary and secondary test evaluation passes for larger datasets; the runs backend still clamps values to its configured ceiling.

Use `--no-llm-judge` with `tt runs start` to opt out of Bedrock LLM judging for a new run.

`tt models download` downloads models that have a Tuned Tensor-hosted artifact, such as SageMaker training outputs. Provider-hosted models can still be used for inference through their `provider_model_id`, but may not expose downloadable weights.

## Billing & Credits

Tuned Tensor uses prepaid credits. New accounts start at a zero balance, so top up before starting your first fine-tuning run; you only pay for successful runs.

```bash
tt balance                 # show available credits, holds, and recent transactions
tt topup --amount 25       # opens Stripe Checkout in your browser
tt topup --amount 25 --no-open  # print the URL instead
```

`tt balance` separates **Available** credits from **Total balance**. Starting a
run or auto-tune session places an estimate on hold, so you can have a positive
total balance while `Available` is too low to start another run. If a run is
rejected with `402 insufficient_credits`, top up or wait for active holds to
settle/release, then retry.

## Spec Validation

`tt eval` validates your local `tunedtensor.json`. It checks required fields, confirms examples are present, warns when guidelines are missing, and checks simple constraints against example outputs. It does not call a model or the Playground API.

## Global Flags

- `-k, --api-key <key>`: override stored API key
- `-u, --base-url <url>`: override API base URL
- `--json`: machine-readable output
- `--no-color`: disable ANSI colors
- `-h, --help`: command help

Examples:

```bash
tt specs list --json
tt runs get <run-id> --json
tt runs start --help
```

## Configuration

Credentials are stored in `~/.config/tuned-tensor/config.json` (respects `XDG_CONFIG_HOME`).

API key precedence:

1. `--api-key`
2. `TUNED_TENSOR_API_KEY`
3. stored config

## Development

```bash
npm install
npm run build
npm run dev
npm run typecheck
npm test
```

## Troubleshooting

If the API rejects a spec with a generic server error, check that `base_model` is one of the supported spec base models listed above.

## License

MIT
