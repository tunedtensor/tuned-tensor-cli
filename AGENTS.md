# AGENTS.md

## Cursor Cloud specific instructions

This is a single-package Node.js/TypeScript CLI (`tt`) for Tuned Tensor's
local CUDA runner and laptop-local agent. Hosted API command modules remain
in `src/commands/` but are not registered. No Docker or database is required
for CLI development.

### Key commands

All standard dev commands are in `package.json` scripts and documented in `README.md` § Development:

- `npm run build` — build via tsup → `dist/index.js`
- `npm run dev` — tsup watch mode for development
- `npm run typecheck` — TypeScript type checking
- `npm test` — vitest unit tests (all tests mock the API, no network needed)

### Running the CLI locally

After `npm run build`, run `npm link` to make the `tt` command available globally, or invoke directly with `node dist/index.js`.

### Notes

- The CLI targets Node 22+ (`tsup.config.ts` sets `target: "node22"`)
  because the bundled local workflow and its locked Python runner require it.
- Agent selection is stored at `~/.config/tuned-tensor/config.json`.
- All tests are fully self-contained with mocked API calls — no API key or network access is needed to run the test suite.

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
