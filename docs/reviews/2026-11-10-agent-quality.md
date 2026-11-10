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
to avoid selecting a lucky run: the original runner created LoRA weights before
Trainer initialized its default RNG seed. The follow-up fixes initialization
and records repeatability limits in the backend comparison. Neither demonstration adapter is left
active or serving.

The separate final-test evaluation of the frozen model is supplemental evidence;
it does not replace the original run's validation report automatically. The
activation gate evaluates the evidence recorded for the selected run. A first-class
final evaluation of frozen weights, with a distinct promotion decision, remains
an extension needed for repeated model selection.

## Remaining work, in priority order

| Priority | Work | Why |
| --- | --- | --- |
| Completed | Seed model/LoRA initialization and record seed/memory/parameter metrics | Repeated initial weights match; GPU training remains numerically nondeterministic |
| P1 | Add a final evaluation/promotion stage for frozen artifacts | Avoid retraining merely to assess another split; keep final-test evidence separate from tuning |
| Completed | Compare the trainer against Unsloth on matched model/data/precision/context and quality | See [measured comparison](2026-11-10-unsloth-comparison.md); production backend stays unchanged |
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

The initial tranche's `npm run check` passed: TypeScript checks, 1,023 CLI tests, 209 runtime
tests, 62 adapter Python tests, 40 foundation Python tests, four decision Python
tests, and the build; 15 platform/integration cases were explicitly skipped.
An earlier full gate overlapping the adapter experiment failed the foundation
resume test. It passed in isolation and in the final sequential gate. The test
now reports the first subprocess's exit code/stdout/stderr and requires the
expected checkpointed SIGTERM exit, so unrelated process failure cannot be
misdiagnosed merely as missing resume output. The earlier transient's underlying
cause was not established; it is retained in `check-initial-failure.log`.

## Follow-up security review

The diff was checked again around proposal validation, artifact verification,
workspace scoping, process ownership, loopback binding and filesystem writes.
Health probes now reject redirects and cannot update a replacement child's
readiness from an earlier child's probe. Tests exercise redirect refusal,
occupied ports, wrong identity, log symlinks and owned-process cleanup.

The production npm dependency audit initially reported
[GHSA-g3hv-75wg-m249](https://github.com/advisories/GHSA-g3hv-75wg-m249)
through pi-coding-agent's minimatch. The bundled dependency now resolves to
minimatch 11.0.3. Both the working installation and a separate installation of
the actual npm tarball reported zero known production npm vulnerabilities;
the installed CLI also passed version and status checks.

The adapter Python inventory reported no known vulnerabilities for audited
packages. Its CUDA-specific torch version was skipped by PyPI lookup, so
public torch 2.13.0 was queried separately and also had no reported advisories.
This checks published advisories, not the CUDA binary's implementation.

The serving inventory reported three entries for setuptools 80.10.2, representing
two unique advisories: a duplicated
[source-distribution exclusion issue](https://github.com/advisories/GHSA-h35f-9h28-mq5c)
and a [Windows legacy-wheel path issue](https://github.com/advisories/GHSA-xmj3-9gf7-h52w).
TT's serving environment is restricted to Linux, installs with uv, and does not
build/publish source distributions or use setuptools' legacy Windows wheel
installer. These vulnerable paths are not exercised by the reviewed serving
workflow. The findings are retained, not suppressed. vLLM 0.32.0 requires
setuptools <81; moving to a fixed setuptools >=84.1 needs a compatible upstream
runtime and separate serving certification. This review does not claim the
entire dependency inventory is advisory-free or prove absence of all flaws.

The follow-up real-provider evaluation completed three eight-scenario suites
with all automated assertions passing. Manual review still caught an invented
list of required fields during malformed-JSON advice and unsolicited stale
hardware summaries. The prompt now requires the actual parse error/location
without guessing missing fields, and excludes unsolicited cached hardware advice.
Three focused repeats of each affected case passed and were manually checked.
The earlier responses remain in the evidence; an automated pass is not treated
as sufficient evidence of correct conversation content. The agent also now
qualifies its task-policy requirements as requirements of the agent tools;
manual `tt serve <model-id>` is not described as universally blocked.

Report inspection also exposes the recorded split and inference provider. The
agent distinguishes validation from a separate final test and does not present
unrecorded external experiments as evidence covered by the selected report.

The final Chinese Spark acceptance repeated the five-turn workflow against the
separately recorded, passing run `d1e248c6-8657-4c55-8709-b2b1021c1993` using
11 model requests. It identified the `prebuilt_test` split and bounded score
improvement, approved activation and serving, confirmed exact model identity and
health, received the expected refund-routing HTTP completion, stopped the owned
server and rolled back to no active adapter. The port was released and no CUDA
process remained. This reuses an existing evaluated model; it is not another
training result or an independent new quality test.

Final follow-up `npm run check` passed: TypeScript, 1,024 CLI tests, 210 runtime
tests, 64 adapter Python tests, 40 foundation tests, four decision tests and the
build (1,342 passed; 15 explicitly skipped). The full gate ran after GPU
experiments and serving cleanup. `git diff --check` passed. This result is
separate from the original tranche's gate and retained failures above.


Post-creation PR review found a shared-store rollback edge case: activating this
project could replace another project's selection, but the project-scoped reader
then refused to restore that exact previous selection. Rollback now permits only
that recorded previous target while this project owns the current selection.
The preview identifies the target spec, its report/model identities and bytes
are verified, and an alias that disagrees with the run's recorded model is
rejected. Ordinary foreign-project inspection/activation and attempts to replace
the sealed rollback run remain blocked. A real-store workflow regression test
covers these cases; model-process execution is simulated in that test.

After the shared-store fix, the full check passed again: 1,025 CLI + 210 runtime
+ 64 adapter + 40 foundation + four decision tests = **1,343 passed**, with the
same 15 explicit skips, typechecks and build passing. The earlier final-package
consumer test passed at commit 36aa896; CI package tests verify the updated PR.
