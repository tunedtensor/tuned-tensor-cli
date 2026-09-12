# AGENTS.md

## Cursor Cloud specific instructions

This is a single-package Node.js/TypeScript CLI (`tt`) for Tuned Tensor's
local CUDA runner and laptop-local agent. Local workflow commands require no
TT token. Hosted operations are registered under `tt cloud`; `tt auth`,
`tt usage`, `tt balance`, `tt topup`, and `tt publish` handle account access and
reporting. Training uses one local orchestrator; optional `gpu` configuration
sends GPU processes to a user-owned EC2 instance over SSH. Hosted training
start/estimate are retired. See `docs/local-runtime/aws-gpu.md`. No Docker or database is required for CLI development.

### Key commands

All standard dev commands are in `package.json` scripts and documented in `README.md` § Development:

- `npm run build` — build via tsup → `dist/index.js`
- `npm run dev` — tsup watch mode for development
- `npm run typecheck` — TypeScript type checking
- `npm test` — CLI, workflow, local runtime, and Python regression tests
- `npm run test:workflows` — focused conversation and model lifecycle contracts
- `npm run check` — typecheck, all tests, and build
- `npm run eval:agent -- --help` — opt-in real model evaluation; see `docs/testing.md`

### Running the CLI locally

After `npm run build`, run `npm link` to make the `tt` command available globally, or invoke directly with `node dist/index.js`.

### Notes

- The CLI targets Node 22+ (`tsup.config.ts` sets `target: "node22"`)
  because the bundled local workflow and its locked Python runner require it.
- Agent selection is stored at `~/.tuned-tensor/config.json`.
  A TT access token automatically selects managed inference when no BYO
  selection exists. The reserved `tunedtensor/managed` provider reuses this
  token; its endpoint and model cannot be overridden through `models.json`.
  Provider auth and custom models live under
  `~/.tuned-tensor/agent/` (`auth.json`, `models.json`, threads).
- Standard tests need no API key or GPU. Workflow tests use temporary workspaces
  and controlled model/process responses while exercising production code.
  AWS transport tests also execute local rsync 3.2+, bash, setsid and GNU
  timeout; they substitute a local shell for SSH and do not contact AWS.
  `uv` may fetch locked Python dependencies on the first run. Real provider
  evaluation is separate and opt-in; never add it to the default test gate.

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
