# AGENTS.md

## Cursor Cloud specific instructions

This is a single-package Node.js/TypeScript CLI (`tt`) — a thin API client for Tuned Tensor. No Docker, databases, or background services required.

### Key commands

All standard dev commands are in `package.json` scripts and documented in `README.md` § Development:

- `npm run build` — build via tsup → `dist/index.js`
- `npm run dev` — tsup watch mode for development
- `npm run typecheck` — TypeScript type checking
- `npm test` — vitest unit tests (all tests mock the API, no network needed)

### Running the CLI locally

After `npm run build`, run `npm link` to make the `tt` command available globally, or invoke directly with `node dist/index.js`.

### Notes

- The CLI targets Node 20+ (`tsup.config.ts` sets `target: "node20"`). Node 22 works fine.
- Authentication is stored at `~/.config/tuned-tensor/config.json`. The `TUNED_TENSOR_API_KEY` env var or `--api-key` flag can override it.
- All tests are fully self-contained with mocked API calls — no API key or network access is needed to run the test suite.
- `tt specs create --name "..." --model "..."` (inline flags without `--file`) may return a 500 from the API; use `--file` with a JSON spec instead.
