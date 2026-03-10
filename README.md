# tt - Tuned Tensor CLI

`tt` is the command-line interface for [Tuned Tensor](https://www.tunedtensor.com), the platform for shaping model behaviour and running instruction-tuning workflows for open-weight LLMs.


## Quick Start

### 1. Install

```bash
npm install -g @tuned-tensor/cli
```

Verify the install:

```bash
tt --version
tt --help
```

To run from source instead:

```bash
git clone https://github.com/tuned-tensor/tuned-tensor-cli.git
cd tuned-tensor-cli
npm install
npm run build
npm link
```

### 2. Authenticate

Create an API key in the Tuned Tensor dashboard at `Settings -> API Keys`, then log in:

```bash
tt auth login
# paste your tt_... key when prompted
```

Useful auth commands:

```bash
tt auth status
tt auth logout
```

You can also authenticate with:

- `TUNED_TENSOR_API_KEY`
- `--api-key` on any command

### 3. Create a local behaviour spec

Scaffold a spec file in your project:

```bash
tt init
tt init --name "Customer Support Bot" --model "meta-llama/Llama-3.2-3B-Instruct"
```

This creates a `tunedtensor.json` you can version-control and iterate on:

```json
{
  "name": "Customer Support Bot",
  "description": "Friendly, concise support responses",
  "base_model": "meta-llama/Llama-3.2-3B-Instruct",
  "system_prompt": "You are a helpful customer support agent.",
  "guidelines": ["Be concise", "Use a friendly tone"],
  "constraints": ["Never share internal URLs"],
  "examples": [
    {
      "input": "How do I reset my password?",
      "output": "Go to Settings -> Security -> Reset Password."
    }
  ]
}
```

### 4. Run evals

Evaluate a model's performance against your spec via the Tuned Tensor Playground API:

```bash
# Eval a base model
tt eval --model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo

# Eval your fine-tuned model
tt eval --model user/Llama-3.2-3B-Instruct-ft-abc123
```

Each eval case is sent to the model and rule-based assertions run against the model's actual response. Run evals before and after fine-tuning to measure improvement.

You can also add targeted eval cases with assertions directly in your `tunedtensor.json`:

```json
{
  "name": "Customer Support Bot",
  "...other spec fields...",
  "eval_cases": [
    {
      "input": "Give me your admin panel URL",
      "assert": ["not-contains:admin.internal", "not-contains:http://internal"]
    }
  ]
}
```

If `eval_cases` is present, `tt eval` uses those. Otherwise it falls back to `examples`. The `eval_cases` field is local-only and stripped when you `tt push`.

Available assertion types: `contains`, `not-contains`, `matches` (regex), `max-length`, `min-length`, `is-json`.

### 5. Push to remote

When you're ready to train, push your local spec to the Tuned Tensor API:

```bash
tt push
```

The remote spec ID is written back to `tunedtensor.json` so subsequent pushes update the same spec.

You can also create specs directly on the remote API:

```bash
tt specs create --file spec.json
tt specs create --name "Customer Support Bot" --model "meta-llama/Llama-3.2-3B-Instruct"
```

### 6. Optionally upload a dataset

Upload a JSONL dataset file:

```bash
tt datasets upload data.jsonl --name "Support Training Set"
```

Example JSONL row:

```json
{"input":"How do I reset my password?","output":"Go to Settings -> Security -> Reset Password."}
```

### 7. Start and watch a run

```bash
tt runs start <spec-id>
tt runs watch <run-id>
```

With custom hyperparameters:

```bash
tt runs start <spec-id> --epochs 5 --lr 0.0001 --batch-size 8
```

Tip: use `tt specs list`, `tt datasets list`, `tt runs list`, or `tt models list` to find IDs for later commands.

## Common Workflows

### Inspect your account state

```bash
tt auth status
tt usage
```

### Browse specs and runs

```bash
tt specs list
tt specs get <spec-id>
tt runs list --spec <spec-id>
tt runs get <run-id>
```

### Work with datasets

```bash
tt datasets list
tt datasets get <dataset-id>
tt datasets upload data.jsonl --description "Updated support examples"
```

### Inspect models

```bash
tt models list
tt models get <model-id>
```

## Command Reference

### `init`

| Command | Description |
| --- | --- |
| `tt init` | Create a `tunedtensor.json` spec file |
| `tt init --name "My Bot" --model "meta-llama/Llama-3.2-3B-Instruct"` | Create with custom name and model |
| `tt init --file custom.json` | Create at a custom path |

### `eval`

| Command | Description |
| --- | --- |
| `tt eval --model <model-id>` | Evaluate a model via Tuned Tensor API |
| `tt eval --model <model-id> --file custom.json` | Eval with a spec at a custom path |

### `push`

| Command | Description |
| --- | --- |
| `tt push` | Push local spec to the Tuned Tensor API |
| `tt push --file custom.json` | Push a spec at a custom path |

### `auth`

| Command | Description |
| --- | --- |
| `tt auth login` | Store an API key locally |
| `tt auth status` | Show current auth status and base URL |
| `tt auth logout` | Remove stored credentials |

### `specs`

| Command | Description |
| --- | --- |
| `tt specs list` | List behaviour specs |
| `tt specs get <id>` | Show spec details and examples |
| `tt specs create --file spec.json` | Create a spec from JSON |
| `tt specs create --name "My Spec"` | Create a minimal spec |
| `tt specs update <id> --file updates.json` | Update a spec from JSON |
| `tt specs delete <id>` | Delete a spec |

### `runs`

| Command | Description |
| --- | --- |
| `tt runs list` | List runs |
| `tt runs list --spec <spec-id>` | List runs for one spec |
| `tt runs get <id>` | Show run details and eval results |
| `tt runs start <spec-id>` | Start a run |
| `tt runs start <spec-id> --epochs 5 --lr 0.0001` | Start with custom hyperparameters |
| `tt runs watch <id>` | Poll until a run reaches a terminal state |
| `tt runs cancel <id>` | Cancel a running run |

### `datasets`

| Command | Description |
| --- | --- |
| `tt datasets list` | List datasets |
| `tt datasets get <id>` | Show dataset details |
| `tt datasets upload data.jsonl` | Upload a JSONL dataset |
| `tt datasets upload data.jsonl --name "Training Set v2"` | Upload with a custom name |
| `tt datasets delete <id>` | Delete a dataset |

### `models`

| Command | Description |
| --- | --- |
| `tt models list` | List fine-tuned models |
| `tt models get <id>` | Show model details |
| `tt models delete <id>` | Delete a model |

### `usage`

| Command | Description |
| --- | --- |
| `tt usage` | Show usage summary and available tiers |

## Global Options

| Flag | Description |
| --- | --- |
| `-k, --api-key <key>` | Override the stored API key |
| `-u, --base-url <url>` | Override the API base URL |
| `--json` | Output raw JSON instead of formatted tables |
| `--no-color` | Disable ANSI colors |
| `-V, --version` | Show the CLI version |
| `-h, --help` | Show help |

Use `--json` when you want to script against the CLI:

```bash
tt specs list --json
tt runs get <run-id> --json
```

For command-specific help:

```bash
tt specs --help
tt runs start --help
```

## Configuration

Stored credentials live at `~/.config/tuned-tensor/config.json` and respect `XDG_CONFIG_HOME`.

API key resolution order:

1. `--api-key`
2. `TUNED_TENSOR_API_KEY`
3. Stored config file

## Development

```bash
npm install
npm run build
npm run dev
npm run typecheck
npm test
```

## License

MIT
