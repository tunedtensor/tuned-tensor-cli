# Runs, audit and serving

## Run records and audit

`tt pipeline run` prints each stage live with the time since the run started.
`--verbose` adds training and evaluation process output, and `--quiet` hides
both. Full logs stay in the run directory either way.

Adapter evaluation batches four prompts by default, reports
completed examples and ETA, and saves completed batches for recovery. Set
`evaluation.inference.batchSize` to tune memory use; `evaluation.maxExamples`
and `sampleSeed` select an explicit smoke sample. Matching interrupted
baselines automatically continue on retry with the default baseline cache.
See [evaluation settings and recovery](local-runtime/architecture.md#evaluation).

Every run report has a `provenance` block recording what produced it:

- TT, Node and platform versions, and the digest of the locked Python runtime;
- where training ran (local, or the AWS instance and region);
- the spec file and snapshot digests;
- the base model id, revision and content fingerprint;
- every dataset file's SHA-256, the compiled training data digest, and the
  evaluation split and seed;
- the hyperparameters and the training exit code.

The run's event log is hash-chained: each event records the previous event's
hash. When a run completes, TT records the report's SHA-256.

```bash
tt runs audit <run-id>
```

`tt runs audit` checks the event chain and re-hashes the report, then prints
the stage timeline with durations and the provenance. Its verdict is
`verified`, `incomplete` (older TT versions or unfinished runs) or `tampered`.
`tampered` exits with code 2 and names the edited event or file; removing the
completion event or emptying the log also counts as tampering.

Run the audit before you share a report with anyone else. The chain makes edits
evident; it is not a signature, so anyone with write access to the store can
rewrite the whole log.

## Serve a tuned model

Serving still runs on the machine executing `tt serve` and requires a local
Linux/CUDA GPU, even when training used AWS:

```bash
tt serve local-<run-id>
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
serving requires Linux and NVIDIA CUDA. AWS execution transfers the required
datasets and model files to your instance and returns outputs locally; normal
cleanup removes the remote staging directory.

Activation is optional and requires the run to pass a configured
`generalRegression` gate. Without that suite, `tt models activate` fails
closed. `tt serve active` also fails closed if nothing is activated — it does
not silently serve the protected base model. Once activated, use
`tt serve active`.

`tt serve base` does not automatically inject the adjacent project spec.
Pass `--spec tunedtensor.json` when the server should enforce its instructions.
Serving uses a separate locked **vLLM** runtime on Linux/CUDA. A tuned launch
exposes both the adapter and its base model, sharing weights, with upstream
streaming and native tool-call parsing. `tt serve <target> --print-client-config pi`
exports isolated Pi configuration; tools no longer need to be disabled. See
[Local serving and coding harnesses](local-runtime/serving.md) for Pi/OpenCode
setup, resource budgets, hardware limits, and the verified integration scope.

## Serve a foundation checkpoint

Serve a completed pretrain, fine-tune, or RL model directory with its training tokenizer:

```sh
tt serve foundation --checkpoint /path/to/run/pretrain/output --tokenizer /path/to/run/tokenize/output/tokenizer.json
```

Use the model artifact directory containing `config.json` and `model.safetensors`,
not a resumable optimizer checkpoint. The API model name is `foundation`.
`--spec tunedtensor.json` applies a foundation spec's system prompt. Existing
port, authentication, sampling, streaming, and client configuration options apply.
The context limit defaults to the checkpoint's `sequence_length`; a smaller
`--context-length` is allowed. Foundation serving supports text completions and
chat with system/user/assistant messages, without tool-call parsing.

TT exports the weights and tokenizer to a temporary GPT-2 snapshot and uses the
existing pinned vLLM server on Linux/CUDA. Training artifacts are unchanged; the
temporary export is removed when the server exits. Export requires temporary
disk space for a copy of the model weights.

Export parity tests run on CPU as part of `npm test`. To run them directly:

```sh
uv run --frozen --project training/foundation --group test python -m unittest discover -s training/foundation/tests -p test_foundation_export.py
```

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

## Host a decision model yourself

A tuned decision model is a directory you own. Load it with Laya in Python, or
wrap it in a small HTTP service on infrastructure you control. See the
[self-hosting guide](local-runtime/hosting.md).

## Review and use a customized model

The local agent can inspect adapter datasets for conflicts/overlap, read actual
run reports, and prepare activation, local serving, stopping and rollback.
Agree `evaluation.taskQuality` before training; task and general-regression
checks protect the reviewed selection. See [model quality and agent workflow](local-runtime/model-quality.md).
