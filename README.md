# tt - Tuned Tensor CLI

`tt` is the command-line tool for [Tuned Tensor](https://www.tunedtensor.com).
Use it to define behaviour specs, validate them, launch fine-tuning runs,
manage datasets, and download or serve trained models.

The main CLI documentation lives at
[tunedtensor.com/docs/cli](https://tunedtensor.com/docs/cli). This README is a
short install and development reference.

## Install

```bash
npm install -g @tuned-tensor/cli
tt --version
```

Run from source:

```bash
git clone https://github.com/tunedtensor/tuned-tensor-cli.git
cd tuned-tensor-cli
npm install
npm run build
npm link
```

## Quick Start

```bash
tt auth login
tt init --name "Customer Support Bot" --model Qwen/Qwen3.5-2B

# Edit tunedtensor.json, then:
tt eval
tt push
tt runs start <spec-id>
tt runs watch <run-id>
```

To continue training from a completed fine-tuned model artifact:

```bash
tt runs start <spec-id> --parent-model <model-id>
```

Useful discovery commands:

```bash
tt specs list
tt datasets list
tt runs list
tt models list
tt models base
tt balance
```

For the full command reference, including dataset-backed runs, long-example
policies, continued fine-tuning, evaluation caps, local model serving,
configuration, and billing, see the [CLI docs](https://tunedtensor.com/docs/cli).

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
