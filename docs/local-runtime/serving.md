# Local serving and coding harnesses

`tt serve` verifies TT artifacts and launches **vLLM 0.28.0**. vLLM owns model
execution, batching, KV/prefix caching, streaming, and native tool-call parsing.
There is no TT inference loop, HTTP server, or proxy between the harness and
vLLM. The serving dependencies are locked separately from training, in an
external cache environment; an npm upgrade does not mutate the training runtime.

## Requirements and scope

- The packaged serving runtime requires **Linux and NVIDIA CUDA**. CPU/macOS
  serving is not supported by this runtime; CLI/package commands still work
  there. Unsupported hardware or model/adapter combinations fail rather than
  silently falling back to the old server.
- Install `uv` and prefetch the certified model with `tt models prefetch`.
  First launch installs the locked serving dependencies and may compile kernels;
  it needs substantially more time and disk than a warm request. CUDA JIT may
  require a working CUDA toolkit/C++ toolchain. Model loading remains offline.
- Foundation checkpoints and image inputs are not supported by this path.
- Qwen3.5-2B base and a one-step, rank-16 `all-linear` TT-shaped adapter have been
  exercised on GB10. The registry's Nemotron and Muse Glimmer architectures have
  upstream support, but their exact adapters/hardware combinations are **not
  certified by that Qwen smoke test**. No 70B model was tested here.

## Serve once, compare base and tuned

```bash
tt models prefetch tunedtensor.json

tt serve local-<run-id> --config local-runner.json \
  --context-length 8192 --max-tokens 512
```

This exposes **both** `local-<run-id>` and `base:<base-model-id>` at the same
endpoint, sharing the base weights. The adapter is registered at startup; TT
never enables runtime adapter upload/loading. There is no merge/export step
and no modification of the verified weights. `tt serve base` starts base-only.
`active` still fails when no adapter is activated.

Artifact integrity, certified base revisions, and stored behavior-prompt
fingerprints retain their verification. A stored adapter's behavior prompt is
applied to both model IDs for comparison. `tt serve base` without explicit
`--spec` does not inject the adjacent spec; use `--spec tunedtensor.json` for that.
`--no-spec-prompt` is the explicit opt-out. A small generated template prefix
preserves the owner instructions while leaving tool messages and upstream chat
formatting intact; it never changes the saved tokenizer/template.

### Resource controls

- `--context-length`: total context budget, default 16,384 tokens. Keep it large
  enough for the harness's instructions and tool results, not just the question.
- `--max-tokens`: generation budget, default 512. vLLM receives this through its
  generation configuration; the upstream API owns per-request validation.
- `--gpu-memory-utilization`: fraction of device memory budgeted to vLLM,
  default 0.8. On shared-memory machines leave room for the OS and other work.
  For the small Qwen GB10 smoke we used **0.15**; this is not suitable for every
  model size. Reduce the context or memory budget when upstream reports a fit
  failure, and ensure the weights themselves still fit.
- `--max-concurrent-requests`: maximum sequences scheduled in a vLLM batch,
  default one, maximum eight. This is upstream batching, **not** the previous
  HTTP admission/429 contract; upstream may queue additional requests.

Health becomes ready after loading/warm-up. Ctrl-C terminates the process group.
A port conflict is checked before loading; upstream owns the final bind and any
race. The unreleased custom server's `--merge-adapter` flag is removed.

## Pi

Export a new isolated configuration rather than replacing global settings:

```bash
export PI_CODING_AGENT_DIR="$(mktemp -d)"
tt serve local-<run-id> --config local-runner.json \
  --context-length 8192 --max-tokens 512 \
  --print-client-config pi > "$PI_CODING_AGENT_DIR/models.json"
```

Use the same launch options as the running server. Exporting config verifies the
TT target but does not start a server. Adapter exports include both model IDs.
For example, in a disposable coding-fixture directory:

```bash
pi --provider tt-local --model 'base:Qwen/Qwen3.5-2B' \
  --tools read --no-extensions --no-skills --no-prompt-templates \
  --no-context-files --no-session --offline --thinking off \
  -p 'Read fixture.py and explain what its function returns.'
```

Repeat with `--model local-<run-id>` in a fresh session. `--tools read` is a
read-only tool allowlist, **not a filesystem sandbox**. Use a disposable account
or an enforcing harness permission boundary for untrusted tasks. Enable edits
or shell commands only when you intend to grant those permissions.

For question-only comparisons, `--no-tools` remains an optional experiment
setting. It is no longer a server requirement. `reasoning: false` and
`supportsReasoningEffort: false` in the Pi export describe this endpoint's
reasoning-control contract; **neither disables tool calls or asserts that the
model cannot reason**.

## OpenCode

OpenCode can connect directly using its existing custom provider support:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "tt-local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "TT local",
      "options": { "baseURL": "http://127.0.0.1:8000/v1" },
      "models": {
        "base:Qwen/Qwen3.5-2B": {
          "name": "TT base",
          "limit": { "context": 8192, "output": 512 }
        }
      }
    }
  }
}
```

Add the exact adapter ID from `/v1/models` to compare it. Set the actual server
budgets, configure OpenCode's permissions, and select
`opencode run --model 'tt-local/base:Qwen/Qwen3.5-2B' ...`. TT does not install a
second agent framework or change global OpenCode configuration.

## Tool support and security

TT selects upstream parsers for the registered models:

- Qwen3.5-2B: `qwen3_xml`.
- Nemotron 3.5 Lightning: `qwen3_coder` (ordinary non-thinking template mode).
- Muse Glimmer: `muse_glimmer`.

These are protocol integrations, not guarantees that a model will choose the
right tool or produce correct code. A complete test must observe a tool call,
execute an allowed tool, send its result back, and verify the final answer.
The harness owns tool execution/permissions; vLLM only generates the calls.
Fine-tuning can change tool reliability, so compare the tuned model too.

The default bind is `127.0.0.1:8000`. Non-loopback binds require both
`--allow-remote` and `--api-key-env NAME`. An authentication-only middleware
extends the bearer check to **all** upstream HTTP routes, including health,
tokenization, and administrative routes; vLLM's built-in guard covers only
selected prefixes. No token is put in argv or exported client configuration.
Pi uses `${NAME}` interpolation; OpenCode can use `"apiKey": "{env:NAME}"`.
Provide the variable separately to each process. Without `--api-key-env`, local
serving is unauthenticated. Use a trusted host or encrypted tunnel, never expose
plain HTTP directly to the Internet. Request/output logging and vLLM usage
telemetry are disabled by the TT launcher.

## API and measurement

Use `/health`, `/v1/models`, and `/v1/chat/completions`. API bodies, SSE events,
usage, tool calls, cancellation, and error semantics come from the pinned
upstream server, not a TT approximation. The old custom server's message/byte
limits, HTTP errors, and health JSON are not preserved as an API contract.

Measure startup and first-request compilation separately from steady-state
first-text latency and tokens/second. Compare identical prompts, templates,
budgets, and fresh harness sessions on an otherwise-idle machine. Test generated
code independently; a tiny one-step adapter proves integration, not better
coding. Pi's zero provider token prices do not mean zero hardware cost.

References: [vLLM LoRA](https://docs.vllm.ai/en/v0.28.0/features/lora/),
[Nemotron tool configuration](https://docs.nvidia.com/nim/large-language-models/latest/get-started/advanced/get-started-nemotron-3.5-lightning.html),
[OpenCode custom providers](https://opencode.ai/docs/providers/#custom-provider).
