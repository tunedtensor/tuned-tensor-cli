# DGX Spark

> [!IMPORTANT]
> The standalone `tt-local` binary is deprecated. Install `@tuned-tensor/cli`
> and use `tt`. Existing projects still work; `tt local …` remains a hidden alias.

DGX Spark is the reference host for local text SFT with LoRA adapters. TT
certifies `Qwen/Qwen3.5-2B` at snapshot
`15852e8c16360a2fea060d615a32b45270f8a8fc`, the larger
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` checkpoint, and
`meta-models/Muse-Glimmer-30B`.

## Check the host

Run on the Spark:

```bash
tt hardware
nvidia-smi
node --version
uv --version
```

`tt hardware` sizes certified adapter models and the foundation engine against
this GPU. Spark unified memory should show Nemotron and Muse Glimmer as
trainable; consumer cards typically only fit Qwen LoRA. `tt status` and the
shell agent reuse the cached report.

TT Local requires Node 22+, `uv`, working CUDA PyTorch, and enough free space
for the Hugging Face cache plus run artifacts.

## Create a project

```bash
mkdir -p ~/tuned-tensor-runs/support-adapter
cd ~/tuned-tensor-runs/support-adapter
tt init --name "Support Adapter" --model Qwen/Qwen3.5-2B --profile spark
```

For a Nemotron execution worker:

```bash
mkdir -p ~/tuned-tensor-runs/nemotron-worker
cd ~/tuned-tensor-runs/nemotron-worker
tt init \
  --name "Nemotron Worker" \
  --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
  --profile spark
```

Nemotron uses the BF16 customization checkpoint rather than the NVFP4
deployment checkpoint. Its defaults enable activation checkpointing, cap the
sequence length at 1,024 tokens, and apply LoRA only to shared attention and
Mamba projections. The base weights require about 66 GB before cache and
training overhead, so this path is not intended for ordinary consumer GPUs.

For a Muse Glimmer execution worker:

```bash
mkdir -p ~/tuned-tensor-runs/muse-glimmer-worker
cd ~/tuned-tensor-runs/muse-glimmer-worker
tt init \
  --name "Muse Glimmer Worker" \
  --model meta-models/Muse-Glimmer-30B \
  --profile spark
```

Muse Glimmer is a vision-language checkpoint. The runtime fine-tunes its text
tower only: it loads the checkpoint through `AutoModelForImageTextToText` (the
text tower is not exposed in the causal-LM auto mapping) and leaves the vision
tower frozen and unused. Defaults enable activation checkpointing and cap the
sequence length at 2,048 tokens; the base weights are comparable to Nemotron
and require Spark-class unified memory.

Edit both generated examples in `tunedtensor.json`. For a meaningful run,
replace them with a larger, representative dataset and a separate validation
split.

The generated `local-runner.json` uses CUDA and project-local artifacts. A
durable Spark configuration can set:

```json
{
  "artifactRoot": "/home/eve/tuned-tensor-runs/artifacts",
  "storeRoot": "/home/eve/tuned-tensor-runs/store",
  "paths": {
    "modelCache": "/home/eve/.cache/huggingface"
  },
  "evaluation": {
    "inference": {
      "device": "cuda"
    },
    "scoring": {
      "mode": "exact_match"
    },
    "timeoutMs": 1800000
  }
}
```

Every Python stage uses the locked runtime included in the npm package; a
source checkout and a custom runner path are neither required nor supported.

## Preflight and run

```bash
tt doctor tunedtensor.json
tt validate tunedtensor.json
tt models prefetch tunedtensor.json
tt models verify-base tunedtensor.json
tt pipeline run --spec tunedtensor.json --config local-runner.json
```

`doctor` resolves the same bundled project and paths the run will use, imports
Torch/Transformers/PEFT, requires visible CUDA, checks writable storage, and
rejects unchanged placeholders. `validate` reads and normalizes the actual
dataset before any run state or artifact directory is claimed.

The runner provides these paths to Python:

- `SM_CHANNEL_TRAINING`: prepared chat JSONL directory;
- `TT_HYPERPARAMETERS_PATH`: generated SFT/LoRA parameters;
- `SM_OUTPUT_DIR`: logs and metrics;
- `SM_MODEL_DIR`: adapter output;
- `SM_CHANNEL_BASE_MODEL`: optional verified local model snapshot;
- `HF_HOME`: configured persistent model cache.

## Verify and serve

```bash
tt runs report <run-id>
tt models verify local-<run-id>
tt serve local-<run-id> --spec tunedtensor.json --port 8000
```

`tt serve active` fails unless an adapter is activated. Activation requires a
`generalRegression` suite in `local-runner.json`. Serve a specific adapter
with `tt serve local-<run-id>` until then.

In another shell:

```bash
curl http://127.0.0.1:8000/
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Classify: I loved it."}]}'
```

If a run fails, start with `tt runs events <run-id>` and
`tt runs get <run-id>`. The run record reports its `artifact_dir`; the
main subprocess logs there are `training/training.log`,
`baseline-eval.json.inference.log`, and `candidate-eval.json.inference.log`.
The adapter is registered as soon as its manifest verifies, even if candidate
evaluation fails afterward.
