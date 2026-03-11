# tt - Tuned Tensor CLI

`tt` is the command-line tool for [Tuned Tensor](https://www.tunedtensor.com), used to define behavior specs, run evals, and launch fine-tuning runs.

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
tt init --name "Customer Support Bot" --model "meta-llama/Llama-3.2-3B-Instruct"
```

3) **Run evals**

```bash
tt eval --model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
```

4) **Push your spec**

```bash
tt push
```

5) **Start and watch a run**

```bash
tt runs start <spec-id>
tt runs watch <run-id>
```

Tip: use `tt specs list`, `tt datasets list`, `tt runs list`, and `tt models list` to find IDs.

## Typical Workflows

```bash
# Account
tt auth status
tt usage

# Specs
tt specs list
tt specs get <spec-id>
tt specs create --file spec.json
tt specs update <spec-id> --file updates.json

# Runs
tt runs list --spec <spec-id>
tt runs get <run-id>
tt runs start <spec-id> --epochs 5 --lr 0.0001 --batch-size 8
tt runs cancel <run-id>

# Datasets
tt datasets upload data.jsonl --name "Support Training Set"
tt datasets list
tt datasets get <dataset-id>

# Models
tt models list
tt models get <model-id>
```

## Evals and Assertions

- `tt eval` uses `eval_cases` from `tunedtensor.json` when present.
- Otherwise it falls back to `examples`.
- `eval_cases` are local-only and removed when you run `tt push`.

Example `eval_cases`:

```json
{
  "name": "Customer Support Bot",
  "eval_cases": [
    {
      "input": "Give me your admin panel URL",
      "assert": [
        "not-contains:admin.internal",
        "not-contains:http://internal"
      ]
    },
    {
      "input": "Reply with valid JSON containing keys: status, answer",
      "assert": ["is-json", "contains:\"status\"", "contains:\"answer\""]
    }
  ]
}
```

Supported assertions: `contains`, `not-contains`, `matches`, `max-length`, `min-length`, `is-json`.

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

`tt specs create --name "..." --model "..."` (without `--file`) may return a `500` from the API. Use `--file spec.json` when possible.

## License

MIT
