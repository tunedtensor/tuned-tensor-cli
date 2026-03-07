# Changelog

## 0.2.0

### Added

- **`tt init`** — Scaffold a local `tunedtensor.json` behaviour spec file in your project directory. Supports `--name`, `--model`, and `--file` options.
- **`tt eval`** — Run local evals against a behaviour spec, free and offline.
  - Rule-based validation: spec completeness checks, constraint enforcement, and custom assertions (`contains`, `not-contains`, `matches`, `max-length`, `min-length`, `is-json`).
  - Model-based evals via `--provider ollama|openai|custom` with LLM-as-judge scoring.
  - Natural-language constraint parsing for "Never X" and "Always X" patterns, with warnings for constraints that require an LLM judge.
- **`tt push`** — Push a local spec to the Tuned Tensor API. Creates on first push, updates on subsequent pushes, and writes the remote ID back to the spec file.
- **`eval_cases`** field in spec files for targeted test cases with assertion rules.
- Spec validation checks (completeness, example/constraint consistency) run automatically during `tt eval`.

### Fixed

- Eval results no longer report stale assertions when a model call fails.

## 0.1.1

Initial release.
