# Testing the TT product promise

TT should let a user describe a model task, fine-tune it, assess the evidence,
and serve the intended model with confidence and little wasted work. A change
is acceptable when those workflows still work, including refusal, failure,
and recovery paths. A large unit-test count alone does not establish that.

This proposal starts with the existing Vitest, Node, and Python tests, adds a
small set of connected workflow contracts, and separates deterministic
regressions from real agent and GPU evaluation. There is no new test framework,
judge model, evaluation service, or coverage-percentage target.

## The default change gate

```bash
npm run test:workflows # Focused workflow contracts
npm run check          # Typecheck, all tests, and build
```

`test:workflows` is a subset of `test:cli`, so `npm test` already includes it.
The existing PR and npm publishing workflows run typecheck, `npm test`, and
build; no separate opt-in CI job is needed to enforce the new tests. CI also
installs the packed npm artifact on Linux/macOS and Node 22/24 to check the
shipped entrypoint. Standard tests need no provider key or GPU. The first
Python run may need network access to install its locked `uv` dependencies.

| User expectation | Executable contract | Real boundary exercised |
| --- | --- | --- |
| “Create my project, then preview fine-tuning.” | `conversation.test.ts`: prepare, approve, resume, prepare pipeline, approve | Real Pi loop, tools, file creation, durable thread/actions, session controls, and CLI subprocess plan |
| “Only run what I reviewed.” | Reject and changed spec/config cases | Real proposal fingerprints, approval checks, zero command dispatch |
| “Preview approval must not start training.” | Persisted proposal changed to real execution | Approval rejects it before command dispatch |
| “Tell me when work fails; don't run it twice.” | Execution/provider failure and duplicate approval cases | Persisted terminal action, visible failure, no model call during approval, no repeated dispatch |
| “Don't spin forever.” | Repeated tool-call case | Production Pi loop's 12-tool-call limit |
| “Did fine-tuning help on unseen examples?” | `runtime.test.ts`: full adapter lifecycle | Real dataset split, Python request/response protocol, scoring, comparison, report files and run store |
| “Serve the model that passed review.” | Verify, activate, resolve `serve active`, rollback | Real CLI commands consume artifacts produced by the same run |
| “Don't repeat expensive work unnecessarily.” | Repeat the same completed run | No additional training or evaluation process calls; one registered model |
| “Don't sacrifice general behavior for a better task score.” | Target improves, general regression fails | Activation rejected; previously active model remains the serving target |
| “Don't trust changed model files.” | Tamper with trained weights | Verification, activation, serving resolution, and resume reject corruption |
| “A rehearsal isn't a trained model.” | Runtime dry-run | No process call, registered model, activation, or serving target |

Files live in `src/__tests__/workflows/`. The conversation fixture scripts only
the provider stream; it does not replace Pi or pretend tool calls succeeded.
The runtime fixture replaces only the expensive Python model process, emitting
its protocol responses and artifact files. Splitting, scoring, manifests,
storage, and CLI consumers execute normally. Each test uses temporary state
and removes it afterwards. No user project, credentials, or real model weights
are needed.

The controlled outputs establish workflow correctness, **not learned model
quality**. Serving checks resolve and verify the launch command; they do not
load these fixture weights into a real inference server. Existing serving
tests cover the HTTP/process behavior separately.

## Real agent evaluation

The deterministic tests cannot establish whether a model interprets ordinary
language correctly. Run the small live scenario set for agent prompt, tool
schema, model-selection, or Pi dependency changes, and before a release that
changes conversational behavior:

```bash
npm run eval:agent -- --help
npm run eval:agent -- --list
npm run eval:agent
npm run eval:agent -- --scenario adapter-dry-run
npm run --silent eval:agent > agent-eval.jsonl # Save JSON lines without npm's banner
```

The runner uses TT's configured provider, model, and thinking level, including
the usual environment overrides. Configure these with `tt agent configure`
and authenticate through TT first. Live calls may incur provider charges;
they are deliberately excluded from `npm test` and CI's default gate.
Managed inference consumes TT usage allowance; BYO calls use the selected
provider's billing. The runner uses the normal model-runtime initialization,
including its saved configuration normalization and any provider credential refresh.

Four synthetic cases cover an adapter preview, a foundation preview,
an invalid spec, and an honest handoff for report inspection and serving.
Assertions check tool use, sealed proposal contents, requested execution mode,
pipeline stages and settings, persisted state, and lack of workspace changes.
They do not compare exact prose or ask a judge model to approve another model.

Every case creates a fresh temporary project and conversation. Nothing is
approved. The runner blocks tools outside pipeline description, validation,
and preparation, so it cannot initiate training, model downloads, or Hub
searches. It limits each case to 8 model requests, 12 tool calls, a 90-second
deadline, and 2,048 output tokens per model response by default. Use the
documented flags to adjust time/token limits. A provider that ignores
cancellation causes a hard failure after a five-second grace period.

JSON lines include the selected model/thinking, individual assertions,
responses, tool/proposal evidence, request counts, and elapsed time. Known
provider/environment secrets are redacted. Any failed automated assertion or
provider failure exits nonzero. The serving-handoff case also requires a
person to check command correctness and that the answer invents no results.
An automated pass is not a completed human review.

For a release decision, run the same cases three times on both the current
release and the proposed change with the same provider/model/thinking and
limits. Keep the JSON output with the review, alongside the commit and lockfile
versions. Require all supported workflow assertions to pass; do not average
away an unsafe execution, invented completion, or wrong-model selection.
Compare model-request and tool counts for wasted loops, and inspect elapsed
time without treating provider latency as a stable performance benchmark.
Investigate intermittent failures instead of retrying until one run is green.

## Real training and serving acceptance

Before changes to training, inference, model dependencies, or model support
are released, run a small real adapter workflow on a supported CUDA host.
Use a fixed spec, explicit disjoint training and held-out data, a pinned base
model revision, and a general-regression dataset. Keep its config, hardware
snapshot, logs, report, and artifact manifest with the release evidence.

The acceptance check is: complete baseline/train/candidate/compare; inspect
held-out and general scores; verify and activate the resulting model; start
`tt serve active`; send an actual `/v1/chat/completions` request; confirm the
served identity and task behavior; stop it and confirm resources are released.
Repeat the same run to check reuse, and roll back the active model. Define
task-specific score and resource limits before running, rather than inferring
success from a completed training process. For foundation changes, exercise
the corresponding foundation stages and resume behavior on real hardware too.

This is a **release acceptance procedure**, not an implemented automatic GPU
lane. A GPU lane can be added when we have a stable host, small pinned fixtures,
and measured limits. The CPU fixture results must never substitute for this
evidence or be presented as model-quality scores.

## Current gaps and extension rule

The local agent currently has no local report-reading, activation, or serving tools.
It can prepare dry-run previews of training and evaluation pipelines; users
must run explicit shell commands for real training, local report inspection,
and serving. Cloud report tools are available when signed in, but this suite
exercises the local workflow with cloud tools disabled. The live handoff case
documents that limitation; it does not certify the full chat-to-serving promise.
When those tools ship, replace the handoff with a multi-turn scenario that
assesses an actual report and serves the exact reviewed artifact.

General-regression thresholds protect activation when a regression suite is
configured. Explicit serving by model ID is not an automatic model-quality
gate. Integrity verification and model-quality assessment are separate checks.

For each new user-visible capability, add one representative success scenario
and its consequential failure/recovery case at the existing boundaries. For
each workflow bug, add the failing scenario before fixing it. Keep assertions
about observable outcomes and durable evidence, not incidental internal call
order. Add machinery only when a concrete scenario cannot be expressed cleanly
with these tools.
