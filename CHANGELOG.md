# Changelog

## Unreleased

### Added

- `tt hardware` inventories this machine (CPU, RAM, disk, NVIDIA GPU, and
  optionally the bundled torch/uv stack) and reports what certified adapter
  models and the foundation engine can train, LoRA fine-tune, or infer.
  The report is cached at `~/.tuned-tensor/hardware.json` for `tt status`,
  `/status`, `/context`, pipeline plan/validate warnings, and the local
  agent. The agent calls `examine_hardware` when you ask it to examine this
  host or decide what can run here.

### Fixed

- A quick inventory (`tt doctor`, `tt hardware --quick`, or
  `examine_hardware` with `quick: true`) does not replace a fresh full
  `tt hardware` snapshot, so nvidia-smi-only probes cannot flip CUDA after
  torch already reported it unavailable.
- nvidia-smi inventory no longer queries unused `compute_cap`, retries
  simpler CSV fields, and never treats the weekday timestamp header as the
  GPU name.
- A failed bundled Python probe marks adapter and foundation train/finetune
  as not possible. `examine_hardware` defaults to the same full probe as
  `tt hardware`.

## [0.13.1-beta.4] - 2026-08-26

### Changed

- Renamed the bundled adapter training project from `training/local-runner` to
  `training/adapter` so it sits beside `training/foundation`.
- Laptop and project state now live under one `.tuned-tensor` folder:
  `~/.tuned-tensor/` for config, agent auth/threads, run metadata, and uv
  caches, and `.tuned-tensor/` in the project for artifacts. `TUNED_TENSOR_HOME`
  relocates the laptop parent. Existing `~/.tuned-tensor-local` stores and
  `~/.cache/tuned-tensor-local` uv environments are still used when the new
  paths are absent. Existing XDG config and agent state are likewise retained
  until their new paths exist, so upgrades keep provider auth, custom models,
  model selection, and conversation threads.

### Security

- Local run-store directories and their JSON, JSONL, copied report, and
  cancellation-marker files are kept user-only (`0700`/`0600`) under the new
  shared state root. Preserved stores are repaired before use, and symbolic
  links or unsupported filesystem entries are refused rather than followed.

### Fixed

- A run is marked completed only after its canonical report has been stored;
  missing or failed report copies leave the run nonterminal for a safe retry.
- When cancellation wins during completion, partially persisted model and
  report state is removed so a cancelled adapter cannot remain serveable.

## [0.13.1-beta.3] - 2026-08-25

### Added

- The conversational agent can describe the shipped adapter and foundation
  pipeline DAGs, optional foundation RL stages, and exact `tt pipeline`
  commands from the built-in workflow tool instead of relying on model memory.
- A read-only `search_hugging_face` agent tool searches public Hugging Face
  model or dataset metadata for foundation and fine-tuning discovery. Searches
  are bounded, unauthenticated, download nothing, and exclude model-card and
  dataset-card text.
- A bounded `inspect_training_source` tool lets the agent ground educational
  explanations in the exact foundation or adapter Python source shipped with
  the running build. It exposes only a fixed component list, not arbitrary
  filesystem access, and returns the source path and SHA-256 with the code.

## [0.13.1-beta.2] - 2026-08-24

### Added

- A local-only foundation engine can build a tokenizer and GPT checkpoint from
  a foundation behavior spec, then run pretraining, supervised fine-tuning,
  optional numeric-reward RL, chat/BPB evaluation, and inference benchmarking
  as one ordered `tt pipeline` workflow.
- `tt init --engine foundation` and `tt pipeline init --engine foundation`
  create matching specs and contract-valid local DAGs. Validation, planning,
  `--only`, and `--skip` preserve typed artifact dependencies before execution.
- Foundation reports include hashed tokenizer, model, and evaluation artifacts
  plus stage metrics for reproducible local inspection.

### Changed

- Pipeline parsing and normalization now consume the immutable
  `@tuned-tensor/pipeline-contract@1.1.0` release. Foundation execution remains
  CUDA-only and single-process; unsupported compositions fail before run-state
  creation or GPU work.
- The first foundation baseline is intentionally small and inspectable rather
  than a port of NanoChat's distributed training stack. Its built-in chat
  metrics reuse spec examples and are evidence of execution/in-sample fit, not
  held-out model quality.

### Security

- Foundation run directories, prompts, configs, logs, reports, tokenizers, and
  model artifacts are private (`0700` directories and `0600` files). Symlinks,
  special nodes, missing outputs, malformed reports, and unsuccessful stage
  metrics fail closed before artifacts are accepted or hashed.

### Fixed

- Correct next-token SFT label alignment and BPB byte accounting across every
  batch row, and stop chat evaluation at the model's trained end token.
- Keep RL context bounded and rollout sampling seeded and on-policy while using
  a binary-reward advantage that corrects failed samples instead of allowing
  one-answer policy collapse. Numeric reward inputs are validated before work
  begins.
- Avoid Python stdlib `tokenize` shadowing and reject unsupported multiprocess
  plans before side effects.

## [0.13.1-beta.1] - 2026-08-23

### Fixed

- `/login` writes the API key into `~/.config/tuned-tensor/agent/auth.json`.
  Pi's `setRuntimeApiKey` only keeps the key in memory, so the file stayed
  `{}` and the next `tt` session asked for `/login` again.
- Missing-auth chat errors no longer name a leftover provider or tell you to
  open the tt shell (you are already in it). They point at `/login` or
  `/model`.
- Local-runner tests no longer fail when SIGINT and SIGTERM arrive at the
  child in either order while cleanup is still pending.

## [0.13.1-beta.0] - 2026-08-23

### Added

- `/login` asks for a provider, then prompts for that provider's API key
  (input is hidden) and stores it in TT's local agent auth file.
  `/login <provider>` skips the provider prompt.

### Changed

- Interactive `tt` is the first onboarding step. The unconfigured shell banner
  and chat hint point at `/model` instead of requiring `tt agent configure`
  first. `/model` and `/login` show OpenAI and OpenRouter.
  `/model` suggests `openai/gpt-5.6-sol` and OpenRouter DeepSeek V4 Flash.
  Other catalog providers remain available as `/model <id>` or `/login <id>`.
  `/model <provider>` lists that provider's models. `tt agent configure`
  remains the non-interactive path.
- Provider `auth.json` and `models.json` now live under
  `~/.config/tuned-tensor/agent/` (or the XDG equivalent). `tt` no longer
  reads or writes `~/.pi/agent/`. Missing-auth errors tell you to open the
  `tt` shell and run `/login` instead of pointing at environment variables
  or hand-editing that file. Unknown-provider errors point at
  `/login <id>` or `/model <id>`.

### Fixed

- `/login` no longer doubles typed characters on the provider and API key
  prompts (the shell readline was still listening while a nested prompt ran).
- Thread persistence redacts API keys and OAuth tokens stored by `/login`,
  including keys saved later in the same `tt` session.
- `/model <provider>` notes when the catalog is truncated.
- README no longer says local endpoints such as Ollama can be selected
  without a key.

## [0.13.0] - 2026-08-18

First stable local-only `tt`. Hosted account commands stay in the tree but
are unregistered. `tt local …` remains a hidden alias.

### Changed

- Unregistered `tt auth` and `tt publish` again. Modules stay on disk like
  the other hosted commands; the CLI surface is local-only.
- `tt info` reports `status: local` instead of `local-runner-preview`.
- `Qwen/Qwen3.5-2B` is pinned to Hugging Face snapshot
  `15852e8c16360a2fea060d615a32b45270f8a8fc`, matching Nemotron and Muse.
- `tt models get` now verifies the stored artifact (manifest and integrity)
  instead of returning the raw record alone.
- Root `--help` no longer duplicates the Commands list. The command catalog
  lists pipeline, agent, status, and shell, and no longer lists auth/publish.
- Hosted `tt init` next-steps now point at `tt validate` / `tt doctor` /
  `tt run`.
- README install now leads with
  `curl -fsSL https://tunedtensor.com/install.sh | sh`. Direct npm install
  remains documented. Version pins use
  `curl … | TT_VERSION=beta sh` so the variable reaches the installer.

### Fixed

- `tt models activate` fails closed with a clear error when the run has no
  `generalRegression` suite, instead of a missing-file traceback.
- `tt serve active` fails if no adapter is activated, instead of silently
  serving the protected base model.
- Doctor GPU summary uses `nvidia-smi --query-gpu=name,driver_version` so the
  first line is not a timestamp.
- Local table IDs keep the `local-` prefix and shorten the UUID:
  `local-3520da7e`, not `local-35`.
- Node no longer prints a serve URL before Python has loaded the model.
  Python still prints `Serving …` after it is ready. `GET /` matches
  `/health`.
- `uv run` is invoked with `--quiet` to hide environment chatter.
- Adapter extraction uses `tarfile.extractall(..., filter="data")` with a
  TypeError fallback on older Python.

## [0.13.0-beta.1] - 2026-08-18

### Added

- `tt publish` uploads local run evidence (spec snapshot + run report) to the
  Tuned Tensor dashboard without starting managed training.
- Restored `tt auth login|logout|status` so a personal `tt_` API key can be
  stored for publish. Logout clears only the API key and keeps agent selection.

### Changed

- `tt publish` checks auth and payload size before confirmation, rejects
  ambiguous run-id prefixes, schema-validates reports, and confines
  `report_path` reads to the local store or artifact roots.

## [0.13.0-beta.0] - 2026-08-18

### Changed

- The CLI is local-only. `tt runs`, `tt models`, `tt init`, `tt doctor`,
  `tt validate`, `tt run`, and `tt serve` now invoke the local CUDA workflow.
  Hosted commands (`tt auth`, `tt push`, `tt balance`, `tt topup`, `tt cloud`,
  and the rest of the managed API tree) are unregistered, not deleted, so they
  can be restored later.
- `tt local …` remains as a hidden alias for existing scripts.
- Root `--api-key` / `--base-url` flags and the `TT_TARGET` override are gone.
- The interactive shell no longer accepts a `cloud` one-shot prefix.
- The shell banner no longer prints `local · <project> · <spec>`; that
  context remains on `/status` and `/context`.

## [0.12.0-beta.1] - 2026-08-14

### Changed

- The interactive shell now runs the local workflow by default; hosted commands
  run one-shot via `cloud <command>` (or `tt cloud <command>` outside the
  shell). Removed the `/mode`, `/cloud`, and `/local` session slash commands
  added in beta.0.
- `/model` now controls the TT agent model instead of the fine-tuning base
  model: it shows a short suggestion list by default, searches with
  `/model <query>` (closest matches only), and switches with
  `/model <provider>/<model>`.
- `tt agent models` now lists models alphabetically.
- Aligned agent wording with the Tuned Tensor brand across the CLI, docs, and
  the npm package description.

### Internal

- Centralized agent-model read/update logic in an `agent-control` function
  layer shared by the shell and the `tt agent ...` commands.
- The lazy agent client detects selection changes via a config revision counter
  instead of re-reading the config file on every turn.
- Drop Windows from the npm smoke-test matrix; keep Ubuntu and macOS.

## [0.12.0-beta.0] - 2026-08-14

### Added

- Surface the conversational agent's model provider and underlying model in the
  shell banner, `/context`, `/status`, and `tt agent status`.
- Add `/cloud` and `/local` slash aliases to switch workflows in one word
  (aliases for `/mode cloud|local`).

### Changed

- `tt agent status` now reports the resolved provider and model display names
  alongside the configured IDs.

### Internal

- Publish beta prereleases under the `beta` npm dist-tag; stable releases stay
  on `latest`.
- Verify the GitHub release tag matches `package.json` before publishing and
  publish with npm provenance.
- Smoke-test the packed npm artifact across Ubuntu, macOS, and Windows on
  Node.js 22 and 24.

## [0.11.0] - 2026-08-12

### Added

- Before opening the interactive `tt` shell, check npm briefly for a newer
  stable CLI release and print the exact upgrade command when one is available.
  Registry failures and timeouts stay silent and never block launch.

## [0.10.0] - 2026-08-12

### Added

- **Approval-gated local spec scaffolding** — In local mode, the `tt` assistant
  can prepare creation of one new workspace folder containing a canonical,
  semantically validated `tunedtensor.json`. Filesystem mutation remains behind
  explicit `/approve`, requires no cloud authentication, and does not expose
  general shell or filesystem access.

### Security

- **Workspace-confined creation** — Bind prepared actions to the active workspace
  identity, reject absolute, nested, traversal, symlink, existing, and changed
  destinations, and use exclusive no-overwrite writes through stable Linux
  directory handles. Cloud mode remains filesystem-free, while ambiguous write
  outcomes are sealed for manual inspection instead of deleting paths that may
  belong to another writer. Non-Linux approval fails before mutation.

## [0.9.2] - 2026-08-10

### Changed

- **Brand-aligned terminal output** — Table headers, detail labels, status
  colors, markdown code spans, URLs, and the shell banner now use the Tuned
  Tensor violet palette (#8B5CF6 / #A78BFA) instead of cyan/blue. The shell
  banner renders the active workflow mode in the brand accent. No functional
  changes.

## [0.9.0] - 2026-08-09

### Added

- **Laptop-local Pi agent** — Run the interactive agent harness and conversation
  state locally with Pi 0.84.1 while routing inference through the user-selected
  local or remote provider. Reuse Pi provider catalogs, `auth.json`,
  `models.json`, and custom OpenAI-compatible providers. `tt agent status`,
  `models`, and `configure` manage non-secret provider/model/thinking selection
  metadata.
- **Typed Tuned Tensor tools** — Expose bounded, strict read tools for specs,
  runs, diagnostics, reports, estimates, datasets, models, balance, and ledger
  transactions, plus prepare-only spec create/update tools. Run start and
  cancellation stay outside the model tool loop as explicit commands.
- **Local durable conversations** — Persist transcripts and approval state
  under the TT config directory with user-only permissions and atomic writes.

### Changed

- **Local approval execution** — Execute prepared mutations only from local
  `/approve` code with API capability discovery, server-enforced spec versions,
  deterministic action IDs, and at-most-once local
  claims. Post-dispatch uncertainty remains durable as `outcome_unknown`;
  `/reject` remains non-mutating.
- **Hosted thread boundary** — Ordinary `tt` no longer calls AgentCore
  `/agent/*` endpoints. Existing hosted agent threads are not migrated into or
  shown by the local conversation store.

### Security

- **Credential separation** — Keep the Tuned Tensor API token confined to TT
  REST calls, redact it from prompts/tool results/transcripts, never copy it to
  Pi configuration, and never send it to model providers.

## 0.8.1

### Fixed

- **Contained prompt styling** — Keep the active prompt self-contained and
  repaint only submitted questions as tinted rows, preventing blank surfaces
  and prompt colors from leaking into streamed responses.
- **Readable agent responses** — Render streamed Markdown headings, emphasis,
  lists, links, quotes, and code as terminal-native formatting instead of
  exposing raw syntax markers.

## 0.8.0

### Added

- **Live reasoning** — Show model-provided reasoning as subdued italic narration
  while an agent turn is in progress, including reasoning around managed tool
  calls, before rendering the final answer.

### Changed

- **Conversation-first terminal** — Replace the large tensor-grid banner with a
  compact status header and use quieter chrome so prompts and responses carry
  the visual hierarchy.
- **Separated user input** — Render each prompt on a full-row tinted surface with
  safe reset behavior and fallbacks for basic-color, 256-color, truecolor, and
  no-color terminals.

## 0.7.0

### Added

- **Conversational shell** — Send ordinary sentences entered in the
  interactive `tt` shell to the same authenticated Tuned Tensor agent used by
  the web application, with streamed text and managed tool activity rendered
  directly in the terminal.
- **Durable agent conversations** — Start, list, and resume conversations with
  `/new`, `/threads`, and `/resume <id>`. Resumed conversations also restore
  outstanding approval requests.
- **Approval-gated actions** — Review agent-proposed mutations in the terminal
  and explicitly accept or decline them with `/approve` and `/reject`.

### Changed

- **Hybrid input routing** — Preserve direct execution for known cloud and
  local CLI commands while routing other input to the agent. Prefix a command
  with `:` to make command intent explicit.
- **Response cancellation** — Press Ctrl+C during an agent turn to stop the
  streamed response without leaving the interactive shell.

## 0.6.0

### Added

- **Shell `/model` command** — Show the model in play (active local model or
  cloud spec base model) and activate verified local models with
  `/model <id>`.
- **Shell logo banner** — Show the Tuned Tensor tensor-grid mark and the CLI
  version in the shell header.

### Changed

- **Shell styling and hints** — Align the interactive shell with the main
  CLI's output style (✓/✗ marks, bold headings and labels), use the brand
  violet accent for shell chrome, show the active model in the banner, and
  suggest corrections for mistyped session commands. Typing `exit` or `quit`
  leaves the shell.

## 0.5.0

### Added

- **Unified local workflow** — Install the complete `@tuned-tensor/local`
  runtime with the main CLI and expose its init, doctor, validation, training,
  run/model inspection, activation, rollback, prefetch, verification, and
  serving commands under `tt local`.
- **Terminal shell** — Running `tt` in a TTY now opens a scrollback-friendly
  cloud/local command shell with mode switching, context and status views,
  completion, and session navigation commands.
- **Target-specific spec projection** — Keep a shared `tunedtensor.json`
  source while sending only supported fields to the hosted API or strict local
  runner.

### Changed

- **Runtime requirement** — Require Node.js 22 or newer so the unified package
  can run the locked local training workflow.
- **License** — Align the unified CLI with TT Local under Apache License 2.0.
- **Default scaffold** — Generate two distinct examples and omit an empty
  `eval_cases` field so a new spec can be validated by either target.

### Security

- **Credential handling** — Mask interactive API-key entry, fail instead of
  prompting in non-TTY environments, and enforce user-only permissions on the
  stored cloud configuration.

## 0.4.24

### Added

- **`tt runs list --summary`** — Request compact run-list responses for agents and scripts without detailed evaluation or event payloads. Works for both account-wide and `--spec` run lists, including raw JSON output.

## 0.4.23

### Changed

- **CLI positioning** — Clarify that `tt` is the client for the optional managed Tuned Tensor service, point local CUDA and DGX Spark users to `tt-local`, and make `tt runs report` the final step in the README quickstart.

## 0.4.22

### Added

- **`tt runs report`** — Fetch sanitized run reports through the Tuned Tensor API and show side-by-side Expected, Base, and Tuned outputs for top regressions. The command also supports `--mode failures`, `--split primary|test|all`, `--limit`, and `--json` so users can inspect failed cases without cloud server access.

## 0.4.21

### Added

- **`tt models serve` multimodal inputs** — Preserve OpenAI-style image content parts, load image data URIs/URLs/local paths with Pillow, and route requests with images through the model processor so local reference serving can exercise vision-language artifacts such as Qwen3-VL.

### Changed

- **`tt models setup-runtime`** — Install and check Pillow alongside torch/transformers so the managed local runtime is ready for image inputs.
- **`tt datasets upload --format document_ocr_jsonl`** — Validate OCR asset metadata locally before upload, including image MIME/data URI shape, page numbers, non-empty OCR outputs, and invisible control characters in OCR prompts/outputs.

## 0.4.20

### Added

- **`tt label`** — Teacher-label real, unlabeled data into training datasets. `tt label upload` sends a JSONL (`{"input": ...}` per line) or CSV file (`--input-column` picks the text column; up to 50,000 rows / 50 MB) and starts a managed cloud labeling workflow that drafts an output per row with a teacher model under the behaviour spec's system prompt — no need to stay connected. `tt label watch` re-attaches to progress; `tt label rows` / `accept` / `reject` / `edit` review the teacher's drafts by row index; `tt label promote` turns accepted+edited rows into a validated dataset ready for `tt runs start --dataset`; `tt label list` / `status` / `cancel` manage jobs. Labeling reserves credits up front and settles against actual teacher token usage.

## 0.4.19

### Added

- **`tt runs estimate`** — Preview estimated training tokens, cost, and rough wall-clock duration before starting a run. The command accepts the same run-configuration flags as `tt runs start` and calls the preflight estimate API.

## 0.4.18

### Added

- **`tt models export`** — Export a fine-tuned model to GGUF and optionally package it for Ollama. Wraps llama.cpp's `convert_hf_to_gguf.py` + `llama-quantize` to turn the download → convert → quantize → Modelfile → `ollama create` flow into one command. Convert-native outtypes (`f16`/`f32`/`bf16`/`q8_0`/`tq*`) use a single conversion step; K-quants/IQ-quants (e.g. `q4_k_m`, `q5_k_m`, `q6_k`) convert to an f16 intermediate and then quantize. `--ollama` writes a Modelfile (`FROM ./model.gguf` plus the behaviour spec's system prompt as `SYSTEM`) and runs `ollama create tt-<slug>`; `--ollama-name` and `--no-ollama-create` override the tag/creation. `--print-command` shows the full plan without executing, and llama.cpp tooling is located via `--llama-cpp`/`--convert-script`/`--quantize-bin` or `LLAMA_CPP_DIR`.
- **`tt runs start --max-output-tokens` / `--eval-reserved-output-tokens`** — Pass runner eval output-budget controls for long-response tasks.

## 0.4.17

### Added

- **`tt runs start --long-examples` / `--max-seq-length`** — Pass long-example handling and training sequence length controls through to dataset-backed runs.

## 0.4.16

### Added

- **`tt runs start --parent-model`** — Continue fine-tuning from a completed Tuned Tensor model artifact instead of starting again from the original base model.

## 0.4.15

### Added

- **`tt runs get` / `tt runs diagnose`** — Show eval output diagnostics when the API provides them, including JSON validity, schema-key match, non-JSON prefix, visible reasoning prefix, average output length, and plain-language evaluation insights.
- **`tt models serve --managed`** — Run a lightweight local lifecycle wrapper in front of the reference server. Managed serving starts the model on demand, serializes generation requests, stops the model after an idle timeout, restarts after a request threshold or health failure, and logs JSONL request metrics including schema validity and gate-field results.

## 0.4.14

### Added

- **`tt models serve --json-schema`** — Enforce a default JSON Schema for local chat completions, validate model output, retry malformed JSON with `--json-repair-attempts`, and return HTTP 422 instead of a successful response when the model cannot satisfy the JSON contract.

## 0.4.13

### Changed

- **`tt runs diagnose`** — Show the live training phase and token accuracy returned by the diagnostics API, alongside epoch progress, loss, pace, ETA, and latest update time.

## 0.4.12

### Added

- **`tt models download`** — Show an interactive progress bar with transfer rate and ETA for model artifact downloads.
- **`tt models setup-runtime`** — Install an isolated local Python runtime for reference model serving.
- **`tt models serve`** — Serve fine-tuned model IDs, extracted model directories, or `.tar.gz` artifacts through a local OpenAI-compatible chat completions API. The server auto-applies the behaviour spec prompt from `tunedtensor.json` by default and supports `--device auto|cpu|cuda|mps`.

## 0.4.11

### Added

- **`tt runs diagnose`** — Show live run diagnostics from the Tuned Tensor API, including epoch progress, loss, pace, ETA, latest update time, and plain-language insights.

### Changed

- **Model details** — Label hosted model identifiers without exposing backend storage/provider names in normal CLI output.

## 0.4.10

### Added

- **Dataset upload validation** — `tt datasets upload` now rejects invisible control characters before upload, with tests covering malformed JSONL rows that previously reached the API.

### Fixed

- **`tt balance`** — Show a single `Credits` value instead of exposing available, reserved, and total balance accounting. Low-balance warnings now use total `balance_cents`.

### Docs

- Simplify README billing copy to match the single credit-balance CLI output.

## 0.4.9

### Added

- **`tt models base`** — List the supported base models accepted by local CLI spec validation, with `--json` support for scripts.

## 0.4.8

### Added

- **`tt models download`** — Download locally stored SageMaker model artifacts through the API when a model has a downloadable S3 artifact.
- **Run eval caps** — `tt runs start` now accepts `--max-eval-examples` and `--max-test-eval-examples` for small smoke tests and cost-controlled runs.

### Changed

- `tt datasets upload` now uses the signed upload URL flow: request an upload URL, PUT the JSONL file directly to S3, then finalize the dataset with the API.
- `tt eval` now validates the local `tunedtensor.json` only. It no longer requires `--model`, calls the Playground API, or runs response assertions.

## 0.4.7

### Fixed

- **Dataset format validation** — `tt datasets upload` validates JSONL locally before upload. Each line must be `{"input": "...", "output": "..."}` for the supervised format the API expects, with a short preview of the first bad row on failure.

- **Dataset ID prefixes** — `tt runs start --dataset`, `tt datasets get`, and `tt datasets delete` now resolve unambiguous dataset ID prefixes before calling the API. This matches the shortened IDs shown by `tt datasets list` while still sending the full UUID required by the API.

### Security

- **Dependencies** — Updated `package-lock.json` to clear `npm audit` findings on transitive packages.

## 0.4.6

### Added

- **`tt runs start --dataset`** — Start a run from an uploaded dataset instead of inline spec examples. Optional `--train-ratio`, `--validation-ratio`, and `--test-ratio` flags send explicit dataset split ratios.

### Fixed

- **Spec base models** — Align CLI defaults and validation with the Tuned Tensor API enum. Unsupported models now fail locally instead of bubbling up as generic API errors.

### Docs

- **Billing guide** — Remove stale signup-bonus wording. The README now says new accounts start at a zero balance and should top up before starting fine-tuning runs.

## 0.4.5

### Added

- **`tt runs start --no-llm-judge`** — Start a run without Bedrock LLM judging. The CLI now sends `use_llm_judge = false` in the request body so the API honors the opt-out.

### Fixed

- **`tt balance`** — Handle the current zero-bonus billing API response by no longer expecting signup-bonus fields. Zero-balance accounts now display cleanly without stale bonus messaging.

## 0.4.4

### Fixed

- **Spec & run ID prefixes** — `tt specs get/update/delete`, `tt runs start`, `tt runs list --spec`, `tt runs get/cancel/watch` now resolve 8+ char ID prefixes client-side via the list endpoint. Previously the truncated IDs shown in `tt specs list` and `tt runs list` produced opaque `[404] Behaviour spec not found` because the API only accepts full UUIDs.
- Help text for ID arguments now correctly says "(full UUID or 8+ char prefix)" instead of just "(full or prefix)".
- Ambiguous prefixes now error with the matching IDs and names so you can disambiguate. Prefixes shorter than 4 chars are rejected up front.

## 0.4.3

### Fixed

- **`tt specs create --file`** — Validate the file body before sending. Detects run-input payloads (with `spec_snapshot` and friends), missing `name`, and warns on unknown top-level keys. Common cause of opaque `[500] An unexpected error occurred` from the API on `tt specs create`.
- **`--json`** — Errors now respect `--json` and emit `{"error":{"status","code","message"}}` on stdout instead of a colored text message on stderr, so failures can be parsed by tooling.

## 0.4.2

### Changed

- **`tt runs get`** — Show the current run stage, progress percentage, and latest status message returned by the Tuned Tensor API.
- **`tt runs watch`** — Poll and display long-running run progress so users can follow AWS runner stage transitions from the terminal.

## 0.4.1

### Fixed

- **`tt balance`** — Show available credits and credits on hold from the reservation-aware billing API. Low-balance warnings now use spendable `available_cents` instead of total `balance_cents`.

### Docs

- Explain that active runs can reserve credits, so a positive total balance may still produce `402 insufficient_credits` when available credits are too low.

## 0.4.0

### Added

- **`tt balance`** — Show current credit balance, signup bonus status, and recent transactions. Supports `--limit <n>` and `--json`.
- **`tt topup`** — Add prepaid credits via Stripe Checkout. Use `--amount <usd>` for a specific amount (presets: $10, $25, $50, $100; min $5, max $10,000). Opens the checkout URL in your browser by default; use `--no-open` to just print it.

### Removed

- **`tt usage`** — Replaced by `tt balance`. The platform now uses prepaid credits instead of monthly run quotas.

### Changed

- New users automatically receive $5 in free credits on signup. Fine-tuning runs are charged based on `epochs × training_tokens × model_rate`. Inference, evals, and dataset operations remain free.

## 0.3.0

### Changed

- Simplified and shortened `README.md` for faster onboarding and easier scanning.
- Added a concrete `eval_cases` example to show assertion-based eval patterns.

## 0.2.0

### Added

- **`tt init`** — Scaffold a local `tunedtensor.json` behaviour spec file in your project directory. Supports `--name`, `--model`, and `--file` options.
- **`tt eval --model <id>`** — Evaluate a model against a behaviour spec. Calls the model via the Tuned Tensor Playground API and runs rule-based assertions against real responses. Works with both base and fine-tuned models.
- **`tt push`** — Push a local spec to the Tuned Tensor API. Creates on first push, updates on subsequent pushes, and writes the remote ID back to the spec file.
- **`eval_cases`** field in spec files for targeted test cases with assertion rules (`contains`, `not-contains`, `matches`, `max-length`, `min-length`, `is-json`).
- Natural-language constraint parsing for "Never X" and "Always X" patterns, with warnings for unrecognised patterns.

## 0.1.1

Initial release.
