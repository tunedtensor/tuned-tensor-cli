# Local serving and coding-question comparisons

`tt serve` hosts one certified Hugging Face base model or verified LoRA adapter
per process. It uses the existing locked Transformers/PyTorch runtime; there is
no extra inference engine, proxy, agent loop, or global harness configuration.
Foundation checkpoints and image inputs are not supported by this server.

## Serve base or tuned

```bash
# Prefetch once; serving itself remains offline.
tt models prefetch tunedtensor.json

# Base plus the same behavior instructions used for the tuned model:
tt serve base --spec tunedtensor.json --config local-runner.json --max-tokens 512

# Stop the base server with Ctrl-C before switching to the tuned model:
tt serve local-<run-id> --config local-runner.json --max-tokens 512
```

`active` resolves the activated adapter and fails if none is active. Adapter
artifacts, pinned base revisions, and stored system-prompt fingerprints retain
their existing verification. `base` without explicit `--spec` does not inject
the adjacent behavior prompt; use explicit `--spec` for a fair comparison.

- Use CUDA when available (`tt hardware` reports suitability). The loader uses
  BF16 on supported CUDA hardware, FP16 otherwise, and keeps weights resident.
- For a static adapter with spare memory, add `--merge-adapter` to fold LoRA
  into the resident base weights once, removing the separate LoRA operations
  during decoding. This uses PEFT's safe merge and never changes on-disk weights.
  It needs extra startup memory and can change floating-point rounding; measure
  latency and check quality on your workload. Omit the flag to keep the existing
  unmerged path. It is invalid for `base`.
- Generation explicitly enables the KV cache and runs in inference mode.
  Streaming emits decoded text during generation, not after a full response.
- The default admission limit is one request; excess work returns HTTP 429.
  `--max-concurrent-requests` admits up to eight requests but generation remains
  serialized: it is **not** GPU batching or a throughput tuning knob.
- Prompts are limited to 128 messages, 100,000 characters, and 16,384 tokens;
  requested output is limited to 8,192 tokens. Prompt plus output must also fit
  the model's context. Lower output budgets reduce worst-case latency/memory.
- A port conflict fails before loading weights. Health becomes available only
  after loading. Ctrl-C terminates the process group. A failed stream write
  unwinds generation and releases its slot; there is no orphan generation worker.

The default bind is `127.0.0.1:8000`. Non-loopback binds still require both
`--allow-remote` and `--api-key-env NAME`. Never put the key itself on the command
line. Plain HTTP is intended for a trusted host/network or an encrypted tunnel,
not direct Internet exposure.

## Pi harness: question-only mode

Install [Pi](https://github.com/earendil-works/pi-mono) separately. The integration
is tested against the CLI's pinned Pi dependency. Export a **new, isolated** Pi
config rather than replacing `~/.pi/agent/models.json`:

```bash
export PI_CODING_AGENT_DIR="$(mktemp -d)"

tt serve base --spec tunedtensor.json --config local-runner.json \
  --max-tokens 512 --print-client-config pi > "$PI_CODING_AGENT_DIR/models.json"
```

This prints valid Pi `models.json` and exits without starting a server. It
includes the resolved model ID, endpoint, output budget, and OpenAI compatibility
settings. With `--api-key-env NAME`, it exports an explicit **`${NAME}` environment
reference**, never the secret; provide the same variable to Pi. Wildcard bind addresses
are rendered as loopback client addresses for use on the same machine.

Start the matching `tt serve base ...` command in another terminal. Then:

```bash
pi --provider tt-local --model 'base:Qwen/Qwen3.5-2B' \
  --no-tools --no-extensions --no-skills --no-prompt-templates \
  --no-context-files --no-session --offline --thinking off \
  --system-prompt 'Answer coding questions concisely.' \
  -p 'Write a Python function that removes duplicates while preserving order.'
```

For the tuned pass, stop the base server, export configuration with
`tt serve local-<run-id> ... --print-client-config pi` to the same isolated config,
start that exact tuned target, and replace Pi's `--model` with `local-<run-id>`.
Use a fresh Pi invocation, the same question, system prompt, output budget, and
behavior spec for each pass. Explicit model IDs prevent accidentally comparing
the base model against itself. Save outputs locally and run the same independent
tests on generated code; a plausible answer is not proof of correctness.

**This is coding Q&A, not autonomous repository editing.** The bundled server
has no tool-call parser. Non-empty tools, tool histories, stop sequences,
structured-output requests, and multiple completions are rejected rather than
silently ignored. Pi's `--no-tools` is required. OpenCode's normal coding-agent
mode is not supported by this integration; it would require a separately tested
tool-calling contract. No changes to TT's own agent permissions are made.

## API and measurements

Discover the exact model ID with `GET /v1/models`, check `GET /health`, and send
`POST /v1/chat/completions` with `model` matching that ID. Both normal JSON and
`stream: true` SSE responses are supported. Use
`stream_options: {"include_usage": true}` for a final streamed token-usage chunk.
`max_tokens` or `max_completion_tokens` is accepted, but not both. Token-budget
exhaustion reports `finish_reason: "length"`. A generation error after streaming
starts produces an error event and closes without a successful `[DONE]` marker.

Measure model load separately from first-request warm-up. For steady-state
latency, repeat the same bounded workload after warm-up and record first-text
latency, total elapsed time, and completion tokens. Compare base and adapter
sequentially on the same otherwise-idle machine. Pi's zero provider token prices
mean no billed API usage, **not** zero hardware or electricity cost. This change
does not claim optimal throughput across all model/hardware combinations.
