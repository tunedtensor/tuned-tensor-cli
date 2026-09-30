# Decision models

A decision model answers one typed question about an input — which queue,
yes or no, how urgent — in a single encoder forward pass. It returns a
probability for every allowed answer instead of generating text, so the output
always parses and its confidence can be thresholded. Use the decision engine
for classification, routing, triage, guardrails and score gates where a
generative LLM is more than the job needs.

TT fine-tunes [Laya](https://huggingface.co/convaiinnovations/laya)
(`convaiinnovations/laya`, Apache-2.0): a 421M-parameter ModernBERT-large
encoder with a typed decision head. Every stage uses Laya revision
`55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`, the reviewed commit that Laya 0.3.22
pins. Training and evaluation run on CPU, Apple MPS or CUDA; no GPU is required.

## Spec

```json
{
  "engine": "decision",
  "name": "Card support triage",
  "system_prompt": "Which card-support queue should handle this customer message?",
  "guidelines": ["Choose the queue for the customer's main problem."],
  "constraints": [],
  "base_model": "convaiinnovations/laya",
  "decision": {
    "type": "choice",
    "criteria": {
      "lost_or_stolen": "customer lost their card or it was stolen",
      "payment_declined": "a card payment was declined or rejected"
    }
  },
  "examples": [
    { "input": "I think I left my card at the bar last night.", "output": "lost_or_stolen" },
    { "input": "Why was my payment at the grocery store refused?", "output": "payment_declined" }
  ],
  "hyperparameters": { "n_epochs": 3 }
}
```

`tt init --engine decision` writes a starter spec. A complete example lives in
[`examples/single-spec/decision`](../../examples/single-spec/decision/tunedtensor.json).

`system_prompt`, `guidelines` and `constraints` compile into the question's
instructions, as they compile into the system message for the other engines.
The model reads roughly 190 tokens of instructions and option descriptions, so
keep them short; validation warns above 600 characters.

| `decision.type` | `criteria` | Example `output` |
| --- | --- | --- |
| `choice` | Object of label → description (2–32 labels, in option order) | A label, e.g. `payment_declined` |
| `noul` | Optional `{ "true": "...", "false": "..." }` | `true` or `false` |
| `score` | Array of level descriptions, level 0 first (2–10) | A level index, e.g. `2` |

Outputs match labels ignoring case and surrounding spaces. Validation rejects
an output that is not a label and warns when a label has no examples.

Optional `hyperparameters`:

| Field | Default | Meaning |
| --- | --- | --- |
| `n_epochs` | 3 | Passes over the training split |
| `learning_rate` | 2e-5 | AdamW learning rate, linear warmup and decay |
| `batch_size` | 8 | Examples per optimizer step |
| `freeze_encoder` | `false` | Train only the decision head (faster, usually less accurate) |
| `shuffle_options` | `true` | Shuffle option order per row so the head learns meaning, not position |
| `seed` | 0 | Training seed |
| `device` | `auto` | `auto`, `cpu`, `mps` or `cuda` |

`runtime.artifactRoot` is supported. `runtime.gpu`, `evaluation` and adapter
runtime paths are rejected: the model is small enough to run where the CLI runs.

## Run

```bash
tt validate
tt pipeline run --spec tunedtensor.json --dry-run
tt pipeline run --spec tunedtensor.json
```

The pipeline is the adapter recipe: `baseline` evaluates the pinned base model,
`train` fine-tunes it, `candidate` evaluates the tuned model and `compare`
pairs the two. TT splits the spec examples once with a seed derived from the
spec `id` (20% held out, at least one). Training sees only the training split.
The evaluator receives held-out IDs and inputs only; Node joins its predictions
to the expected labels and scores them.

Each evaluation reports accuracy, log-loss and Brier score of the expected
label, mean confidence and per-label recall. The comparison adds improved and
regressed examples and the log-loss and Brier deltas. Because Laya answers
zero-shot from the label descriptions, the baseline is a real model, not an
empty one: the report shows what fine-tuning added on top of the descriptions.

Runs are written to `.tuned-tensor/decision-runs/<run-id>` beside the spec (or
`--output <dir>`), with owner-only permissions:

```text
report.json            summary: model, revision, split, every step's metrics
resolved-workflow.json spec, plan, question and split used
data/                  train.jsonl and unlabeled eval-inputs.jsonl
baseline/  candidate/  config, log, predictions.jsonl, report.json
train/model/           the tuned checkpoint
compare/comparison.json
```

On a 4-core CPU a step of 8 examples takes about 15 seconds, so 120 examples
for 3 epochs trains in about 12 minutes. The tuned checkpoint is about 1.7 GB
because weights are saved in float32; small fine-tuning updates would be lost
in bfloat16.

## Use the tuned model

`train/model` is a complete Laya checkpoint. The fitted temperatures of the
base checkpoint are reset to 1.0, because they described the base model's
logits; cross-entropy training is itself the calibration objective.

```python
import laya

agent = laya.load("path/to/decision-runs/<run-id>/train/model")
question = {
    "type": "choice",
    "instructions": "Which card-support queue should handle this customer message?",
    "criteria": {"lost_or_stolen": "...", "payment_declined": "..."},
}
answer = agent.predict("My card was refused at the shop.", {"queue": question})["answers"]["queue"]
print(answer["choice"], answer["probabilities"])
```

Pass the same instructions and criteria the model was trained with: they are
part of its input. `resolved-workflow.json` records them under `question`.
The tuned model is not registered with `tt models` and is not served through
vLLM.

## Limits

- One question per spec. Laya can answer several questions per input; TT
  trains and evaluates one.
- `convaiinnovations/laya` is the English checkpoint. Its multilingual and
  typed-decisions variants are not certified yet.
- The held-out split comes from the spec examples. With few examples the
  comparison is noisy; add examples before trusting a small delta.
