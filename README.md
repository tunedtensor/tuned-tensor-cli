# tt — Tuned Tensor CLI

Command-line interface for [Tuned Tensor](https://www.tunedtensor.com), the end-to-end model behaviour shaping platform for instruction-tuning open-weight LLMs.

## Install

```bash
npm install -g @tuned-tensor/cli
```

Or run from source:

```bash
git clone https://github.com/tuned-tensor/tuned-tensor-cli.git && cd tuned-tensor-cli
npm install
npm run build
npm link
```

## Authentication

Generate an API key from the Tuned Tensor dashboard (**Settings → API Keys**), then:

```bash
tt auth login
# paste your tt_... key when prompted
```

Alternatively, set the `TUNED_TENSOR_API_KEY` environment variable or pass `--api-key` on every call.

```bash
tt auth status   # check current auth
tt auth logout   # remove stored credentials
```

## Commands

### Behavior Specs

```bash
tt specs list                        # list all specs
tt specs get <id>                    # show spec details + examples
tt specs create -f spec.json         # create from JSON file
tt specs create --name "My Spec"     # create with flags
tt specs update <id> -f updates.json # update from JSON file
tt specs delete <id>                 # delete a spec
```

Spec JSON format:

```json
{
  "name": "Customer Support Bot",
  "description": "Friendly, concise support responses",
  "system_prompt": "You are a helpful customer support agent.",
  "guidelines": ["Be concise", "Use a friendly tone"],
  "examples": [
    { "input": "How do I reset my password?", "output": "Go to Settings → Security → Reset Password." }
  ],
  "constraints": ["Never share internal URLs"],
  "base_model": "meta-llama/Llama-3.2-3B-Instruct"
}
```

### Runs

```bash
tt runs list                         # list all runs
tt runs list --spec <spec-id>        # list runs for a spec
tt runs get <id>                     # show run details + eval results
tt runs start <spec-id>              # start a new run
tt runs start <spec-id> --epochs 5 --lr 0.0001
tt runs cancel <id>                  # cancel a running run
tt runs watch <id>                   # poll until run completes
```

### Datasets

```bash
tt datasets list                     # list datasets
tt datasets get <id>                 # show dataset details
tt datasets upload data.jsonl        # upload a JSONL file
tt datasets upload data.jsonl --name "Training Set v2"
tt datasets delete <id>              # delete a dataset
```

### Models

```bash
tt models list                       # list fine-tuned models
tt models get <id>                   # show model details
tt models delete <id>                # delete a model
```

### Usage

```bash
tt usage                             # show usage summary & tier info
```

## Global Options

| Flag | Description |
|------|-------------|
| `-k, --api-key <key>` | Override stored API key |
| `-u, --base-url <url>` | Override API base URL |
| `--json` | Output raw JSON instead of formatted tables |
| `--no-color` | Disable ANSI colors |
| `-V, --version` | Show version |
| `-h, --help` | Show help |

## Configuration

Credentials are stored in `~/.config/tuned-tensor/config.json` (respects `XDG_CONFIG_HOME`).

API key resolution order:
1. `--api-key` flag
2. `TUNED_TENSOR_API_KEY` environment variable
3. Stored config file

## Development

```bash
npm install         # install dependencies
npm run build       # build with tsup
npm run dev         # watch mode
npm run typecheck   # type-check without emitting
```

## License

MIT
