from __future__ import annotations

import hmac
import json
import math
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from evaluate import (
    configure_hugging_face_cache,
    import_runtime_dependencies,
    load_text_model,
    resolve_adapter_path,
    sampling_kwargs,
)
from model_contract import assert_certified_base_model_revision, chat_template_kwargs


MODEL_ARTIFACT = os.environ.get("TT_MODEL_ARTIFACT")
BASE_MODEL = os.environ.get("TT_BASE_MODEL", "")
MODEL_SOURCE = os.environ.get("TT_MODEL_SOURCE", BASE_MODEL)
BASE_MODEL_REVISION = os.environ.get("TT_BASE_MODEL_REVISION")
MODEL_NAME = os.environ.get("TT_MODEL_NAME", "tuned-tensor-local")
MODEL_LOADER = os.environ.get("TT_MODEL_LOADER", "causal_lm")
SYSTEM_PROMPT = os.environ.get("TT_SYSTEM_PROMPT", "").strip()
HOST = os.environ.get("TT_HOST", "127.0.0.1")
PORT = int(os.environ.get("TT_PORT", "8000"))
DEVICE_REQUEST = os.environ.get("TT_DEVICE", "cuda")
DEFAULT_MAX_TOKENS = int(os.environ.get("TT_MAX_TOKENS", "512"))
DEFAULT_TEMPERATURE = float(os.environ.get("TT_TEMPERATURE", "0"))
DEFAULT_TOP_P = float(os.environ.get("TT_TOP_P", "1"))
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_PROMPT_CHARS = 100_000
MAX_PROMPT_TOKENS = 16_384
MAX_MESSAGES = 128
API_KEY = os.environ.get("TT_API_KEY", "")
MAX_CONCURRENT_REQUESTS = int(os.environ.get("TT_MAX_CONCURRENT_REQUESTS", "1"))


MODEL: Any = None
TOKENIZER: Any = None
DEVICE = DEVICE_REQUEST


def load_runtime(temp_dir: str) -> None:
    global MODEL, TOKENIZER, DEVICE
    if MODEL_LOADER not in ("causal_lm", "image_text_to_text"):
        raise ValueError(f"Unsupported TT_MODEL_LOADER={MODEL_LOADER!r}")
    assert_certified_base_model_revision(BASE_MODEL, BASE_MODEL_REVISION, "Serving base model revision")
    configure_hugging_face_cache(os.environ.get("HF_HOME"))
    import_runtime_dependencies()
    adapter_path = resolve_adapter_path(MODEL_ARTIFACT, Path(temp_dir)) if MODEL_ARTIFACT else None
    MODEL, TOKENIZER, DEVICE = load_text_model({
        "base_model": BASE_MODEL,
        "model_source": MODEL_SOURCE,
        "base_model_revision": BASE_MODEL_REVISION,
        "device": DEVICE_REQUEST,
        "model_loader": MODEL_LOADER,
    }, adapter_path, MODEL_LOADER)
    if os.environ.get("TT_MERGE_ADAPTER", "false") == "true":
        if not adapter_path:
            raise ValueError("Adapter merging requires an adapter target.")
        # In-memory only. Safe merge refuses non-finite weights; never overwrite
        # the verified base snapshot or adapter artifact.
        MODEL = MODEL.merge_and_unload(safe_merge=True)
        MODEL.eval()

GENERATION_LOCK = threading.Lock()
REQUEST_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_REQUESTS)


def text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if not isinstance(content, list):
        raise ValueError("Message content must be text or an array of text parts.")
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "text":
            raise ValueError("The bundled model server accepts text content only.")
        text = part.get("text")
        if not isinstance(text, str):
            raise ValueError("Every text content part must contain a string text field.")
        parts.append(text)
    return "\n".join(parts)


def normalize_messages(raw_messages: Any) -> list[dict[str, str]]:
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("Request must include a non-empty messages array.")
    if len(raw_messages) > MAX_MESSAGES:
        raise ValueError(f"Request exceeds the {MAX_MESSAGES}-message limit.")

    system_parts = [SYSTEM_PROMPT] if SYSTEM_PROMPT else []
    conversation: list[dict[str, str]] = []
    for raw in raw_messages:
        if not isinstance(raw, dict) or raw.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("Each message must have a supported role and text content.")
        if raw.get("tool_calls") or raw.get("function_call") or "tool_call_id" in raw:
            raise ValueError("Tool calling is not supported; use a question-only harness with tools disabled.")
        role = str(raw["role"])
        content = text_content(raw.get("content", ""))
        if role == "system":
            if content.strip():
                system_parts.append(content.strip())
        else:
            conversation.append({"role": role, "content": content})

    if not conversation:
        raise ValueError("Request must contain at least one non-system message.")
    messages: list[dict[str, str]] = []
    if system_parts:
        # The certified chat templates accept one leading system message. Merge the invariant owner
        # prompt and any client context instead of triggering an implicit
        # template fallback with duplicate system turns.
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})
    messages.extend(conversation)
    if sum(len(message["content"]) for message in messages) > MAX_PROMPT_CHARS:
        raise ValueError(f"Prompt exceeds the {MAX_PROMPT_CHARS}-character limit.")
    return messages


def generate_text(messages: list[dict[str, str]], generation: dict[str, Any], on_text=None) -> tuple[str, int, int]:
    try:
        prompt = TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs(BASE_MODEL),
        )
    except Exception as exc:
        raise ValueError(f"Model chat-template rendering failed: {exc}") from exc
    inputs = TOKENIZER(prompt, return_tensors="pt")
    if int(inputs["input_ids"].shape[-1]) > MAX_PROMPT_TOKENS:
        raise ValueError(f"Prompt exceeds the {MAX_PROMPT_TOKENS}-token limit.")
    config = getattr(MODEL.config, "text_config", MODEL.config)
    context_limit = getattr(config, "max_position_embeddings", None)
    if isinstance(context_limit, int) and inputs["input_ids"].shape[-1] + generation["max_new_tokens"] > context_limit:
        raise ValueError(f"Prompt plus requested output exceeds the model's {context_limit}-token context.")
    target_device = next(MODEL.parameters()).device
    inputs = {key: value.to(target_device) for key, value in inputs.items()}
    streamer = None
    if on_text is not None:
        from transformers import TextStreamer

        class ResponseStreamer(TextStreamer):
            def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
                if text:
                    on_text(text)

        streamer = ResponseStreamer(TOKENIZER, skip_prompt=True, skip_special_tokens=True)
    with GENERATION_LOCK:
        import torch

        with torch.inference_mode():
            generated = MODEL.generate(
                **inputs,
                max_new_tokens=int(generation["max_new_tokens"]),
                pad_token_id=TOKENIZER.eos_token_id,
                use_cache=True,
                **({"streamer": streamer} if streamer else {}),
                **sampling_kwargs(generation),
            )
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    completion_tokens = generated[0][prompt_tokens:]
    content = TOKENIZER.decode(completion_tokens, skip_special_tokens=True).strip()
    return content, prompt_tokens, int(completion_tokens.shape[-1])


def bounded_number(value: Any, default: float, minimum: float, maximum: float) -> float:
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise ValueError("Generation values must be JSON numbers.")
    number = float(default if value is None else value)
    if not math.isfinite(number) or number < minimum or number > maximum:
        raise ValueError(f"Generation value must be between {minimum} and {maximum}.")
    return number


def bounded_integer(value: Any, default: int, minimum: int, maximum: int) -> int:
    number = bounded_number(value, default, minimum, maximum)
    if not number.is_integer():
        raise ValueError(f"Generation value must be an integer between {minimum} and {maximum}.")
    return int(number)


class Handler(BaseHTTPRequestHandler):
    server_version = "TunedTensorLocalServer/0.1"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(30)

    def authorized(self) -> bool:
        if not API_KEY:
            return True
        supplied = self.headers.get("authorization", "")
        return hmac.compare_digest(supplied, "Bearer " + API_KEY)

    def send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if not self.authorized():
            self.send_json(401, {"error": {"message": "Unauthorized"}})
            return
        if self.path in {"/", "/health"}:
            self.send_json(200, {"status": "ok", "model": MODEL_NAME, "device": DEVICE})
            return
        if self.path == "/v1/models":
            self.send_json(200, {
                "object": "list",
                "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "tuned-tensor-local"}],
            })
            return
        self.send_json(404, {"error": {"message": "Not found"}})

    def send_event(self, payload: Any) -> None:
        data = payload if isinstance(payload, str) else json.dumps(payload)
        self.wfile.write(("data: " + data + "\n\n").encode("utf-8"))
        self.wfile.flush()

    def do_POST(self) -> None:  # noqa: N802
        if not self.authorized():
            self.send_json(401, {"error": {"message": "Unauthorized"}})
            return
        if self.path not in {"/v1/chat/completions", "/chat/completions"}:
            self.send_json(404, {"error": {"message": "Not found"}})
            return
        stream_started = False
        response_id = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())

        def chunk(delta=None, finish_reason=None, usage=None):
            payload = {
                "id": response_id, "object": "chat.completion.chunk",
                "created": created, "model": MODEL_NAME,
                "choices": [] if usage is not None else [{
                    "index": 0, "delta": delta or {}, "finish_reason": finish_reason,
                }],
            }
            if usage is not None:
                payload["usage"] = usage
            return payload

        def on_text(text):
            nonlocal stream_started
            if not stream_started:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                stream_started = True
                self.send_event(chunk({"role": "assistant", "content": ""}))
            if text:
                self.send_event(chunk({"content": text}))

        try:
            length = int(self.headers.get("content-length", "0"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("Request body is empty or too large.")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Request body must be a JSON object.")
            if "model" in body and body["model"] != MODEL_NAME:
                raise ValueError(f"This endpoint serves {MODEL_NAME!r}; request that exact model id.")
            streaming = body.get("stream", False)
            if not isinstance(streaming, bool):
                raise ValueError("stream must be a boolean.")
            stream_options = body.get("stream_options")
            if stream_options is not None and (
                not isinstance(stream_options, dict)
                or not isinstance(stream_options.get("include_usage", False), bool)
            ):
                raise ValueError("stream_options must be an object with boolean include_usage.")
            for field in ("tools", "functions", "function_call", "stop"):
                if body.get(field):
                    raise ValueError(f"{field} is not supported by the bundled text server; disable it in the client.")
            if body.get("tool_choice") not in (None, "none"):
                raise ValueError("Tool calling is not supported; disable tools in the client.")
            if body.get("response_format") not in (None, {"type": "text"}):
                raise ValueError("Only text response_format is supported.")
            if body.get("n", 1) != 1:
                raise ValueError("Only n=1 is supported.")
            if body.get("max_tokens") is not None and body.get("max_completion_tokens") is not None:
                raise ValueError("Use only one of max_tokens or max_completion_tokens.")
            messages = normalize_messages(body.get("messages"))
            generation = {
                "max_new_tokens": bounded_integer(body.get("max_tokens", body.get("max_completion_tokens")), DEFAULT_MAX_TOKENS, 1, 8192),
                "temperature": bounded_number(body.get("temperature"), DEFAULT_TEMPERATURE, 0, 5),
                "top_p": bounded_number(body.get("top_p"), DEFAULT_TOP_P, 0, 1),
            }
            if not REQUEST_SLOTS.acquire(blocking=False):
                self.send_json(429, {"error": {"message": "The local model is busy; retry shortly."}})
                return
            started = time.perf_counter()
            try:
                content, prompt_tokens, completion_tokens = generate_text(
                    messages, generation, on_text if streaming else None,
                )
            finally:
                REQUEST_SLOTS.release()
            latency_ms = round((time.perf_counter() - started) * 1000)
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            finish_reason = "length" if completion_tokens >= generation["max_new_tokens"] else "stop"
            if streaming:
                on_text("")  # EOS-only responses still need a valid stream.
                self.send_event(chunk(finish_reason=finish_reason))
                if (body.get("stream_options") or {}).get("include_usage"):
                    self.send_event(chunk(usage=usage))
                self.send_event("[DONE]")
                return
            self.send_json(200, {
                "id": response_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_NAME,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "tt_local": {"latency_ms": latency_ms, "device": DEVICE},
            })
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            # Generation runs in this request thread: a failed stream write unwinds
            # generate() immediately, releases the GPU lock/slot, and leaves no worker.
            self.close_connection = True
        except Exception as exc:
            client_error = isinstance(exc, (ValueError, TypeError))
            message = str(exc) if client_error else "Internal model server error."
            if not client_error:
                print(f"Model server error: {type(exc).__name__}", file=sys.stderr, flush=True)
            error = {"error": {"message": message}}
            if stream_started:
                try:
                    self.send_event(error)
                except (OSError, TimeoutError):
                    pass
                self.close_connection = True
            else:
                self.send_json(400 if client_error else 500, error)

    def log_message(self, fmt: str, *args: Any) -> None:
        print("%s - %s" % (self.address_string(), fmt % args), flush=True)


def main() -> None:
    # Reserve the port before allocating model memory. Context managers also
    # close the socket and extracted adapter if loading fails.
    with ThreadingHTTPServer((HOST, PORT), Handler) as server, TemporaryDirectory(prefix="tt-local-serve-") as temp_dir:
        load_runtime(temp_dir)
        print(f"Serving {MODEL_NAME} on http://{HOST}:{PORT}", flush=True)
        print(f"OpenAI-compatible endpoint: http://{HOST}:{PORT}/v1/chat/completions", flush=True)
        print(f"Device: {DEVICE}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Model server stopped.", flush=True)


if __name__ == "__main__":
    main()
