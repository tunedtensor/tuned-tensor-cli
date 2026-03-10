# Changelog

## 0.2.0

### Added

- **`tt init`** — Scaffold a local `tunedtensor.json` behaviour spec file in your project directory. Supports `--name`, `--model`, and `--file` options.
- **`tt eval`** — Run evals against a behaviour spec. Offline mode checks expected outputs; with `--model`, calls the model via the Tuned Tensor Playground API and asserts against real responses.
- **`tt push`** — Push a local spec to the Tuned Tensor API. Creates on first push, updates on subsequent pushes, and writes the remote ID back to the spec file.
- **`eval_cases`** field in spec files for targeted test cases with assertion rules (`contains`, `not-contains`, `matches`, `max-length`, `min-length`, `is-json`).
- Natural-language constraint parsing for "Never X" and "Always X" patterns, with warnings for unrecognised patterns.

## 0.1.1

Initial release.
