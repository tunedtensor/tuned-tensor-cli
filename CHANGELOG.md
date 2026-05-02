# Changelog

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
