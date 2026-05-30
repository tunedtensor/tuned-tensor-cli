# Changelog

## 0.4.13

### Added

- **`tt models serve --json-schema`** — Enforce a default JSON Schema for local chat completions, validate model output, retry malformed JSON with `--json-repair-attempts`, and return HTTP 422 instead of a successful response when the model cannot satisfy the JSON contract.

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

- **Spec base models** — Align CLI defaults and validation with the Tuned Tensor API enum: `Qwen/Qwen3.5-2B`, `google/gemma-4-E2B-it`, and `google/gemma-4-26B-A4B-it`. Unsupported models now fail locally instead of bubbling up as generic API errors.

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
