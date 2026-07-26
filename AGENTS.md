# AGENTS.md

## Cursor Cloud specific instructions

This is a single-package Node.js/TypeScript CLI (`tt`) that unifies Tuned
Tensor's hosted API workflow with the separately packaged local CUDA runner.
No Docker or database is required for CLI development.

### Key commands

All standard dev commands are in `package.json` scripts and documented in `README.md` § Development:

- `npm run build` — build via tsup → `dist/index.js`
- `npm run dev` — tsup watch mode for development
- `npm run typecheck` — TypeScript type checking
- `npm test` — vitest unit tests (all tests mock the API, no network needed)

### Running the CLI locally

After `npm run build`, run `npm link` to make the `tt` command available globally, or invoke directly with `node dist/index.js`.

### Notes

- The unified CLI targets Node 22+ (`tsup.config.ts` sets `target: "node22"`)
  because the bundled local workflow and its locked Python runner require it.
- Authentication is stored at `~/.config/tuned-tensor/config.json`. The `TUNED_TENSOR_API_KEY` env var or `--api-key` flag can override it.
- All tests are fully self-contained with mocked API calls — no API key or network access is needed to run the test suite.
- `tt specs create --name "..." --model "..."` (inline flags without `--file`) may return a 500 from the API; use `--file` with a JSON spec instead.
