# Spark acceptance: agent model quality workflow

This change addresses the first P0 tranche of the product review: actionable
data diagnostics, explicit task acceptance, and an agent workflow from verified
results to local serving. It does not introduce a new training backend or claim
that fine-tuning always improves a model.

## Changes and evidence

| Area | Result | Verification |
| --- | --- | --- |
| Data | Full structural scan reports duplicates, conflicting labels, train/evaluation overlap, lengths and answer frequencies | Real public dataset files; malformed/path/overlap regression cases |
| Acceptance | Criteria recorded before training; task non-regression plus general regression required; sample minimum enforced | Actual held-out inference and controlled activation refusal cases |
| Agent | Lists runs, reads verified reports, prepares activation and rollback | Real Chinese conversation against a real Spark run |
| Serving | Starts only the reviewed artifact; separately checks health and exact model identity; stops its own process group | Actual vLLM HTTP completion, stop, port release, rollback |
| Review integrity | Spec/config/report/manifest/active-selection changes invalidate proposals | Regression cases using actual generated artifact manifests |
| UX failures | Empty-project inspection creates no store; text before/after a tool is separated; empty lists do not imply hidden running jobs | An initial live failure reproduced and fixed; three follow-up empty-project cases |

The live provider was the existing OpenRouter `~anthropic/claude-sonnet-latest`
selection, thinking off. The comparison used release v0.28.1 and the working
branch with the same eight synthetic scenarios. Baseline iterations 1–3 passed
automated checks. Candidate iteration 1 found the empty-store side effect;
iterations 2–5 passed all eight automated checks after correction. Iteration 6
had one provider request timeout after successful read-only spec inspection;
its failure was retained, with no mutation or invented completion. A targeted
follow-up of that scenario passed. Manual review also found speculation about
running/failed records being absent; the tool now states its actual scope and
three final empty-project runs passed without that claim. These observations
are not a claim of perfect provider reliability.

The real Chinese workflow used five user turns and twelve model requests:
review/activate, prepare serving, check readiness, stop, rollback. The agent
needed no pasted terminal output or user-run shell commands. The four mutations
used the normal approval path. Earlier release responses to the serving case
required a command handoff; this is a capability comparison, not a latency A/B
benchmark. Serving startup includes first-use vLLM compilation/warmup.

## Actual model results

Hardware: NVIDIA GB10 / Spark, driver 580.173.02. The adapter used the certified
Qwen/Qwen3.5-2B revision `15852e8c16360a2fea060d615a32b45270f8a8fc`.
The fixture was a four-intent subset derived from Banking77, with public-source
attribution retained in row metadata. Training used 192 examples; validation
used 48 disjoint original-training examples. Another 48 original-test examples
were reserved until the recipe was frozen. The task routes requests to four
internal queue codes and never authorizes refunds. This is functional acceptance,
not a Banking77 benchmark or a representative customer dataset.

The fixed recipe was one epoch, learning rate 0.0002, batch size 2, accumulation
4, LoRA rank 8/alpha 16, maximum training length 256. Criteria were fixed before
running: score >= 0.85, pass rate >= 0.85, at least 40 evaluated examples, no
score/pass-rate regression. Twelve independent simple general-behavior examples
used a maximum score/pass-rate drop of 0.05. That small general suite establishes
only a functional regression check.

| Measurement | Base | First adapter | Outcome |
| --- | ---: | ---: | --- |
| Validation exact task pass rate, 48 rows | 87.50% | 97.92% | Recorded validation gate passed |
| Validation field score | 0.9375 | 0.9896 | Improved on this validation set |
| Reserved final-test task pass rate, 48 rows | 93.75% | 91.67% | Failed the unchanged non-regression criterion |
| Reserved final-test field score | 0.96875 | 0.95833 | Failed the unchanged non-regression criterion |

The original frozen model's final-test failure is retained. No criteria were
lowered. The service was stopped and the active pointer rolled back to empty.
An actual HTTP request before that final test returned the reviewed model ID
and `{"queue":"Q28","refund_approved":false}` for a refund request.
Repeating the original completed run reused training/evaluation artifacts;
model-process log timestamps did not change.

A separate full-pipeline repeat using the identical training recipe and the
final-test split scored 97.92% with passing gates. It trained a **different**
adapter; it does not erase the first model's failure or establish an unbiased
new test result. The recipe was not tuned to that test. The variation is a reason
to avoid selecting a lucky run: the current runner creates LoRA weights before
Trainer initializes its default RNG seed. Neither demonstration adapter is left
active or serving.

The separate final-test evaluation of the frozen model is supplemental evidence;
it does not replace the original run's validation report automatically. The
activation gate evaluates the evidence recorded for the selected run. A first-class
final evaluation of frozen weights, with a distinct promotion decision, remains
an extension needed for repeated model selection.

## Remaining work, in priority order

| Priority | Work | Why |
| --- | --- | --- |
| P1 | Seed adapter initialization before model/LoRA creation, record seeds, compare repeated runs | Stable comparisons require controlling initialization and reporting remaining GPU nondeterminism |
| P1 | Add a final evaluation/promotion stage for frozen artifacts | Avoid retraining merely to assess another split; keep final-test evidence separate from tuning |
| P1 | Benchmark the current trainer against Unsloth on matched model/data/precision/context and quality | No training speedup claim is made by this UX change |
| P1 | Hardware-calibrated training and serving resource policies | Current serving defaults are bounded, not measured optimal settings |
| P1 | Task-specific SFT, preference and verifier/reward data contracts | Structural JSONL checks cannot establish label truth, licensing or RL reward validity |

Local regression tests cover success, insufficient samples, legacy task
regression, changed acceptance policy, changed reports/specs, duplicate approval,
wrong served identity, occupied ports, log symlinks, unexpected server exit,
owned-process stopping and Ctrl-C handling that must not kill a live service
when the UI only cancels a turn. Real CUDA/provider evidence is kept separately
from these deterministic contracts.

Evidence on the review host is under
`/home/eve/projects/tunedtensor/review-evidence/agent-quality-20261110/`, including
specs, dataset hashes, run reports/manifests, original live failures, follow-up
results, Chinese conversation events, serving logs, holdout results and scripts.
No release was published and no customer model was replaced.

Final `npm run check` passed: TypeScript checks, 1,023 CLI tests, 209 runtime
tests, 62 adapter Python tests, 40 foundation Python tests, four decision Python
tests, and the build; 15 platform/integration cases were explicitly skipped.
An earlier full gate overlapping the adapter experiment failed the foundation
resume test. It passed in isolation and in the final sequential gate. The test
now reports the first subprocess's exit code/stdout/stderr and requires the
expected checkpointed SIGTERM exit, so unrelated process failure cannot be
misdiagnosed merely as missing resume output. The earlier transient's underlying
cause was not established; it is retained in `check-initial-failure.log`.
