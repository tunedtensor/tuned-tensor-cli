# Changelog

## 0.2.0

### Added

- **`tt init`** — Scaffold a local `tunedtensor.json` behaviour spec file in your project directory. Supports `--name`, `--model`, and `--file` options.
- **`tt eval`** — Evaluate model performance against a behaviour spec. Requires `--provider` (ollama, openai, or custom). Calls the model with each example, checks responses against constraints and assertions, and scores with LLM-as-judge. Use before and after fine-tuning to measure improvement.
- **`tt check`** — Validate a behaviour spec for free (no model needed). Checks completeness, example/constraint consistency, and reports issues.
- **`tt push`** — Push a local spec to the Tuned Tensor API. Creates on first push, updates on subsequent pushes, and writes the remote ID back to the spec file.
- **`eval_cases`** field in spec files for targeted test cases with assertion rules (`contains`, `not-contains`, `matches`, `max-length`, `min-length`, `is-json`).
- Natural-language constraint parsing for "Never X" and "Always X" patterns, with warnings for constraints that require an LLM judge.

### Fixed

- Eval results no longer report stale assertions when a model call fails.

## 0.1.1

Initial release.
