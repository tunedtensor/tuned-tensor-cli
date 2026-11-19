# AGENTS.md

## Development instructions

This is a single-package Node.js/TypeScript CLI (`tt`) for Tuned Tensor's
local CUDA runner and laptop-local agent. Tuned Tensor is a free, community-driven
open-source project: there is no hosted service, account, token or billing, and
nothing in this repository may depend on one. Every command runs on machines the
user controls; the optional agent uses the model provider the user configures.
Decision specs (`engine: "decision"`) fine-tune a small typed decision
model such as `convaiinnovations/laya` on CPU, MPS or CUDA; see
`docs/local-runtime/decision.md`. Training uses one local orchestrator; optional `runtime.gpu` configuration
sends GPU processes to a user-owned EC2 instance over SSH.
See `docs/local-runtime/aws-gpu.md`. No Docker or database is required for CLI development.

### Behavior spec workflow

Treat `tunedtensor.json` as the source of truth for behavior and training settings.
Read with `get_local_spec` before proposing `prepare_update_local_spec`; preserve
unrelated fields and review the diff before `/approve`. Direct `/spec` review,
validation, diff and history require no model or credentials. Pipelines derive
from its optional pipeline section or saved spec by default; only explicit `--file` loads a separate recipe,
and foundation settings must match the spec. Runtime placement and evaluation policy
also live in the spec; migrate legacy sidecars with `tt pipeline migrate`. See `docs/spec-workflow.md`.

### Key commands

All standard dev commands are in `package.json` scripts and documented in `README.md` § Development:

- `npm run build` — build via tsup → `dist/index.js`
- `npm run dev` — tsup watch mode for development
- `npm run typecheck` — TypeScript type checking
- `npm test` — CLI, workflow, local runtime, and Python regression tests
- `npm run test:workflows` — focused conversation and model lifecycle contracts
- `npm run test:windows` — CLI and portable runtime tests on native Windows
- `npm run check` — typecheck, all tests, and build
- `npm run eval:agent -- --help` — opt-in real model evaluation; see `docs/testing.md`

### Running the CLI locally

After `npm run build`, run `npm link` to make the `tt` command available globally, or invoke directly with `node dist/index.js`.

### Notes

- The CLI targets Node 22+ (`tsup.config.ts` sets `target: "node22"`)
  because the bundled local workflow and its locked Python runner require it.
- Agent selection is stored at `~/.tuned-tensor/config.json`.
  Provider auth and custom models live under
  `~/.tuned-tensor/agent/` (`auth.json`, `models.json`, threads). A saved
  selection of the retired `tunedtensor/managed` provider counts as
  unconfigured, and the next config write drops retired `api_key`/`base_url` keys.
- Standard tests need no API key or GPU. Workflow tests use temporary workspaces
  and controlled model/process responses while exercising production code.
  AWS transport tests also execute local rsync 3.2+, bash, setsid and GNU
  timeout; they substitute a local shell for SSH and do not contact AWS.
  `uv` may fetch locked Python dependencies on the first run. Real provider
  evaluation is separate and opt-in; never add it to the default test gate.

## Explaining user-owned AWS training

- Start new users at `docs/local-runtime/aws-gpu.md`; include spec creation,
  separate AWS/SSH access, `doctor`, dry-run, real run, results and shutdown.
- Use `tt pipeline run` with `runtime.gpu` in tunedtensor.json for both engines.
  Long runs can use `tt pipeline start --reason` or an approved agent background
  run; see `docs/local-runtime/background-runs.md`. Background runs require a
  completed dry-run of the same spec bytes and keep an audit record per run.
- `tt hardware` and `examine_hardware` inspect the laptop only. A missing local
  GPU does not rule out AWS training; use the configured `tt doctor` for that.
- An AWS profile does not provision a GPU. Explain quota, capacity, networking,
  SSH host verification and user-managed instance lifecycle. GPU timeout does
  not stop the instance. Keep the laptop running until outputs return.
- Keep README, AWS guide, agent prompt/tool descriptions and Codex plugin
  guidance consistent when changing the workflow. Follow `docs/testing.md`;
  simulated transport tests are not evidence of a real EC2 run.

## Releasing

Publishing is triggered by GitHub Releases (`published`) and runs the full test
gate before `npm publish`.

- **Stable** — create a normal release whose tag matches `package.json` (for
  example `v0.13.0`). It publishes under the `latest` dist-tag.
- **Beta** — bump `package.json` to a prerelease such as `0.13.0-beta.0`, tag
  the release `v0.13.0-beta.0`, and mark the release as a pre-release. It
  publishes under the `beta` dist-tag, so `npm i -g @tuned-tensor/cli` users
  stay on stable while testers opt in with `npm i -g @tuned-tensor/cli@beta`.
- The workflow refuses to publish if the release tag and `package.json` version
  do not match.
- Native Windows installation is checked in PowerShell 5.1 and 7 before npm
  publication. The release also attaches a tested x64 installer, checksum and
  stable WinGet manifests. See `docs/windows.md` for the catalog submission step;
  merging a PR does not publish a release or list it in WinGet.

## Self-hosted decision models

Tuned Tensor does not host models. `docs/local-runtime/hosting.md` explains how
users serve a complete Laya decision checkpoint on infrastructure they control.
Keep its Laya version in step with `training/decision/pyproject.toml`.
