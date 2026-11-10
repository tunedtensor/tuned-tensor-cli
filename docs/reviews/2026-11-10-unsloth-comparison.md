# Spark adapter backend comparison

This is a small functional benchmark on one NVIDIA GB10, not a general speed
ranking. The production backend remains the locked Transformers/PEFT runner.
Unsloth is isolated in a separate environment and is not a shipped dependency.

## Reproduction contract

Use `scripts/benchmark-adapter-backends.py` once per fresh Python process, with
`--backend transformers` or `--backend unsloth`, `--training-file`,
`--hyperparameters`, and a new `--output` directory. The training file must
already contain the system instruction used by the production training channel.
The model must already be cached at its recorded revision. The script currently
supports Qwen/Qwen3.5-2B without gradient checkpointing, and refuses to overwrite
an existing output. Libraries may write their normal compilation caches;
`UNSLOTH_COMPILE_LOCATION` can place Unsloth's generated Python cache outside the
checkout. Do not run GPU workloads concurrently during measurement.

Matched runs use the same isolated Python 3.12.3 environment in separate
processes; only the Unsloth branch imports its patches. TT's production Python
3.13.14 environment remains unchanged and was used for separate diagnostic
repetitions. Tested shared packages: torch 2.13.0+cu130, transformers 5.15.0, peft 0.20.0,
accelerate 1.15.0 and triton 3.7.1. Unsloth 2026.11.3 uses torchvision 0.28.0.
The complete environment freezes and original failures are retained with the
local evidence. On this ARM64 host, an unconstrained install resolved to the
obsolete unsloth 2024.8. Requiring the current release exposed a missing torchao
wheel on the CUDA index. Installing torchao 0.18.0 from PyPI before installing
Unsloth with the CUDA index resolved that dependency issue. The torchvision
version also needed to match torch. These are isolated benchmark setup steps,
not changes to TT's locked environment.

The recipe is the existing 192-row banking queue fixture, one epoch, learning
rate 0.0002, per-device batch 2, accumulation 4, rank 8, alpha 16, dropout 0.05,
maximum length 256, AdamW and seed 42. Both backends use TT's final-assistant-only
dataset, padding collator and Trainer arguments. Unsloth explicitly loads only
the text tower, disables quantization and gradient checkpointing, and uses the
exact upstream revision. Its default full multimodal load was interrupted
before training because it was not a matched comparison.

Comparison checks include the dataset hash, token/label hash, trainable parameter
count, adapter parameter names/shapes/dtypes and inference through TT's ordinary
Transformers evaluator. `TORCH_DISABLE_NATIVE_JIT=1` matches TT's default child
environment. Early exploratory repetitions without that explicit default are
retained separately and excluded from the matched timings. A production-runtime
comparison process was also stopped when its different Python interpreter was
noticed; the matched TT repetitions use the same interpreter as Unsloth.

## Repeatability

The runner now seeds before model loading and LoRA initialization, then uses the
same seed for Trainer and data sampling. Two diagnostic repeats produced the
same initial trainable-weight hash, while final training losses differed. This
fix controls initialization, not all GPU nondeterminism. Backend-native
initialization is also recorded: using the same numerical seed does not promise
identical initial tensors across implementations.

Three training repetitions per backend completed. All six have identical
serialized token/label hashes, 8,409,600 trainable parameters and 372 saved
adapter tensors with identical names, shapes and FP32 dtypes. Base weights use
BF16. Each backend repeats its own initial trainable-weight hash; those hashes
differ between backends, so quality differences cannot be attributed solely to
optimized kernels. Startup includes import, model loading, compilation and saving;
Trainer time includes first-step compilation. PyTorch allocated/reserved peaks
are not total device or Spark unified-memory usage. No backend is selected by
training loss alone, and none of these experimental adapters is activated.


## Timings and memory

| Three fresh processes | TT Trainer | Unsloth |
| --- | ---: | ---: |
| Total seconds, each repetition | 52.89 / 58.61 / 55.94 | 109.91 / 95.62 / 60.54 |
| Trainer seconds, each repetition | 28.51 / 29.50 / 30.19 | 78.54 / 63.81 / 25.03 |
| Median total seconds | 55.94 | 95.62 |
| Median peak CUDA allocated GiB | 7.21 | 4.67 |
| Median peak CUDA reserved GiB | 9.51 | 4.95 |

Unsloth's allocated peak is about 35% lower on this workload. Its last repetition
has faster Trainer time, but every measured end-to-end run is slower here. Cache
and compilation warmup vary across the repetitions; neither the first run nor
the last run alone is an adequate speed ranking. This short, single-epoch fixture
does not establish the crossover point for longer training or larger contexts.
Dropout 0.05 preserves the current TT recipe; testing different Unsloth recipes
would require a separate quality-controlled experiment.


## Task quality and decision

All six frozen adapters were evaluated through the same ordinary TT evaluator on
the same 48 validation rows. The baseline pass rate was 87.50%, field score
0.9375. TT's three repetitions each passed 48/48 (field score 1.0); Unsloth passed
47/48, 46/48 and 47/48 (field scores 0.98958, 0.97917, 0.98958). All passed the
unchanged task policy: score/pass rate >=0.85, at least 40 rows and no regression.

These few rows do not establish a statistically reliable quality ranking.
Backend initialization differs, final tensors vary across same-seed runs, and
this was task validation only: it does not certify general behavior or a new
untouched final test. The earlier frozen adapter's reserved-test failure remains
in the [workflow review](2026-11-10-agent-quality.md). No threshold was lowered
and no experimental benchmark adapter was activated.

Keep the existing production backend for this workload. Unsloth merits further
work for memory-constrained or longer runs, with measured startup amortization,
several task-specific seeds, larger independent evaluations and serving checks
before changing the default. Complete numeric results, adapter hashes and parity
checks are in [the recorded results](2026-11-10-backend-results.json).

Implementation references used during setup: [Unsloth's Qwen3.5 guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
and [Spark guide](https://unsloth.ai/docs/blog/fine-tuning-llms-with-nvidia-dgx-spark-and-unsloth).
Their general performance claims are not used as measurements here.
