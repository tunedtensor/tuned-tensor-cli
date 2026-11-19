# Composable pipelines (v1)

A pipeline is an ordered JSON recipe. The CLI orchestrates every plan locally;
optional `gpu` configuration moves its GPU processes to your AWS instance.
The portable document contract lives in `@tuned-tensor/pipeline-contract`. The
CLI adds execution planning (step selection) on top of it.

Three local engines share that document version:

- **Adapter** (default): LoRA SFT on a certified Hugging Face checkpoint.
  Runs through `tt pipeline run`. Uses `train` / `evaluate` / `compare` with
  `evaluate.with.evaluator: "behavior"`.
  Certified models (`Qwen/Qwen3.5-2B`, Nemotron 3.5 Lightning, Muse Glimmer)
  have a reviewed revision, architecture, and memory profile. Any other
  Hugging Face text model can be trained as **uncertified** by pinning its
  commit; it loads as a causal LM with safetensors weights, no remote code,
  and generic LoRA defaults (override them in `hyperparameters`):

  ```json
  {
    "base_model": "mistralai/Mistral-7B-Instruct-v0.3",
    "hyperparameters": { "base_model_revision": "<40-char commit SHA>" }
  }
  ```

  Pick an instruct/chat variant (the trainer needs a chat template) and check
  hardware fit with `tt doctor`, since TT has no memory profile for it.
- **Foundation**: from-scratch tokenizer + GPT. Also runs through
  `tt pipeline run`. Uses
  `tokenize` / `pretrain` / `finetune` / optional `rl` plus `bpb`, `chat`, and
  `inference` evaluators. Requires a foundation `tunedtensor.json`
  (`engine: "foundation"`, at least two examples, no `base_model`).
- **Decision**: fine-tunes a small typed decision model
  (`convaiinnovations/laya`, 421M) that answers one choice, yes/no or score
  question per input with a probability per label, instead of generating text.
  Uses the adapter vocabulary (`train` / `evaluate` / `compare`) and runs on
  CPU, Apple MPS or CUDA. Requires `engine: "decision"` and a `decision`
  question; example outputs are labels. See
  [Decision models](local-runtime/decision.md).

All engines are spec-driven. `system_prompt`, `guidelines`, and `constraints`
are compiled into one canonical system instruction, and `examples` supply the
demonstrated behavior. Adapter training, foundation tokenizer/pretraining,
chat SFT, optional RL, and evaluation all receive that same compiled
instruction. Engine-specific fields choose how TT learns the behavior; they do
not redefine the behavior itself.

The foundation engine is a readable, single-GPU local trainer inspired by Andrej
Karpathy's [nanochat](https://github.com/karpathy/nanochat), not a port of its
distributed training stack. With `foundation.corpus_path`, it streams `.txt`
or JSONL (`{"text":"..."}`) shards into a deterministic on-disk token cache;
without that field it retains the bounded prompt/example corpus for smoke runs.
`foundation.validation_path` supplies a distinct held-out BPB corpus. Foundation
plans reject `compare` before creating run artifacts; comparison remains an
adapter-engine capability. Optional RL samples one seeded on-policy completion
per step with a sparse last-number exact reward and therefore requires a numeric
expected output for every example. Treat those metrics as an end-to-end
execution and overfit check—not held-out capability or multi-GPU evidence when
no validation corpus is configured.

`tunedtensor.json` contains behavior, training settings, optional `evaluation`,
`runtime`, and advanced `pipeline` sections. Pipelines use the embedded recipe
or derive it from the saved spec. Only explicit `--file` loads a separate recipe;
it cannot override a conflicting embedded recipe. Foundation recipes must match
the spec's training settings.

```bash
tt pipeline validate --spec tunedtensor.json
tt --json pipeline plan --spec tunedtensor.json
tt pipeline run --spec tunedtensor.json --dry-run
tt pipeline run --spec tunedtensor.json
# Optional: embed the current derived recipe for customization
tt pipeline init --spec tunedtensor.json
# Existing projects: consolidate sidecars, retaining .bak copies
tt pipeline migrate --spec tunedtensor.json
```

Relative runtime paths resolve beside the spec. New project state uses
`.tuned-tensor/store` and `.tuned-tensor/artifacts`. Legacy `local-runner.json`
remains readable with a warning; migrate it before adding inline runtime or
evaluation settings. See [single-file examples](../examples/single-spec) and the
[spec workflow](spec-workflow.md).

`--only` and `--skip` preserve dependency safety: a selected step cannot refer
to an omitted predecessor. Keep step targets local and configure `runtime.gpu` in
`tunedtensor.json` to execute GPU work on your AWS instance. Adapter and
foundation workflows use the same local pipeline. The TT agent may describe,
validate, and prepare a sealed pipeline action; only deterministic `/approve`
handling can execute it.

Long pretraining uses BF16, warmup plus cosine decay, gradient clipping,
periodic metrics, and atomic rolling checkpoints. `--resume <run-directory>`
starts or resumes one stable run identity; completed stages are verified and
skipped, while an interrupted pretrain step restores model, optimizer,
scheduler, RNG, counters, and token cursor. See
[Long foundation runs](local-runtime/foundation-long-runs.md) before an
unattended job.
