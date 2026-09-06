"""Artifact preparation only; vLLM owns HTTP, batching, streaming and tools."""
from __future__ import annotations

import json
import hmac
import os
import signal
import socket
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluate import configure_hugging_face_cache, resolve_adapter_path
from model_contract import assert_certified_base_model_revision, assert_certified_model_config

class BearerAuthMiddleware:
    """Keep TT's auth boundary on ALL routes, including upstream admin APIs.

    vLLM's default bearer guard only covers selected API prefixes. This is
    transport admission only; requests/responses and generation stay upstream.
    """
    def __init__(self, app):
        self.app = app
        self.expected = os.environ["TT_API_KEY"].encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] in {"http", "websocket"}:
            authorization = dict(scope.get("headers", [])).get(b"authorization", b"")
            scheme, _, token = authorization.partition(b" ")
            if scheme.lower() != b"bearer" or not hmac.compare_digest(token, self.expected):
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                else:
                    await send({"type": "http.response.start", "status": 401,
                                "headers": [(b"content-type", b"application/json")]})
                    await send({"type": "http.response.body", "body": b'{"error":"Unauthorized"}'})
                return
        await self.app(scope, receive, send)


def build_vllm_args(model_source: str, adapter_path: str | None, temp_dir: Path) -> list[str]:
    args = [
        "serve", model_source,
        "--served-model-name", os.environ["TT_MODEL_NAME"],
        "--host", os.environ.get("TT_HOST", "127.0.0.1"),
        "--port", os.environ.get("TT_PORT", "8000"),
        "--max-model-len", os.environ.get("TT_CONTEXT_LENGTH", "16384"),
        "--max-num-seqs", os.environ.get("TT_MAX_CONCURRENT_REQUESTS", "1"),
        "--gpu-memory-utilization", os.environ.get("TT_GPU_MEMORY_UTILIZATION", "0.8"),
        "--generation-config", "vllm",
        "--override-generation-config", json.dumps({
            "max_new_tokens": int(os.environ.get("TT_MAX_TOKENS", "512")),
            "temperature": float(os.environ.get("TT_TEMPERATURE", "0")),
            "top_p": float(os.environ.get("TT_TOP_P", "1")),
        }),
        "--language-model-only",
        "--no-enable-log-requests", "--no-enable-log-outputs",
    ]
    if os.environ.get("TT_API_KEY"):
        args += ["--middleware", "serve.BearerAuthMiddleware"]
    base = os.environ["TT_BASE_MODEL"]
    # Upstream model-specific parsers, not a TT parser or model-size heuristic.
    parser = {
        "Qwen/Qwen3.5-2B": "qwen3_xml",
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16": "qwen3_coder",
        "meta-models/Muse-Glimmer-30B": "muse_glimmer",
    }[base]
    args += ["--enable-auto-tool-choice", "--tool-call-parser", parser]
    if base != "meta-models/Muse-Glimmer-30B":
        args += ["--default-chat-template-kwargs", '{"enable_thinking":false}']
    if adapter_path:
        config = json.loads((Path(adapter_path) / "adapter_config.json").read_text())
        adapter_base = config.get("base_model_name_or_path")
        if adapter_base not in {os.environ["TT_BASE_MODEL"], model_source}:
            raise ValueError("The adapter base does not match the verified serving model.")
        base_name = "base:" + os.environ["TT_BASE_MODEL"]
        args[args.index("--served-model-name") + 1] = base_name
        args += ["--enable-lora", "--max-lora-rank", str(max(8, int(config["r"]))),
                 "--lora-modules", json.dumps({"name": os.environ["TT_MODEL_NAME"],
                 "path": adapter_path, "base_model_name": base_name})]
    prompt = os.environ.get("TT_SYSTEM_PROMPT", "").strip()
    if prompt:
        source = Path(model_source)
        template_file = source / "chat_template.jinja"
        if template_file.is_file():
            template = template_file.read_text(encoding="utf-8")
        else:
            template = json.loads((source / "tokenizer_config.json").read_text())["chat_template"]
        if not isinstance(template, str):
            raise ValueError("Serving requires an unambiguous string chat template.")
        # Literal data, never prompt text interpolated as Jinja source. Preserve
        # tool calls/results verbatim while merging leading system context.
        prefix = "{%- set tt = namespace(system=" + json.dumps(prompt) + ", history=[]) -%}"
        prefix += "{%- for message in messages -%}{%- if message.role == 'system' -%}"
        prefix += "{%- set tt.system = tt.system + '\\n\\n' + message.content -%}"
        prefix += "{%- else -%}{%- set tt.history = tt.history + [message] -%}{%- endif -%}{%- endfor -%}"
        prefix += "{%- set messages = [{'role': 'system', 'content': tt.system}] + tt.history -%}"
        path = temp_dir / "chat-template.jinja"
        path.write_text(prefix + template, encoding="utf-8")
        args += ["--chat-template", str(path)]
    return args


def prepare_model_source() -> str:
    base = os.environ["TT_BASE_MODEL"]
    revision = os.environ.get("TT_BASE_MODEL_REVISION")
    assert_certified_base_model_revision(base, revision, "Serving base model revision")
    configure_hugging_face_cache(os.environ.get("HF_HOME"))
    source = os.environ.get("TT_MODEL_SOURCE", base)
    if not Path(source).is_dir():
        if source != base:
            raise ValueError("Explicit serving snapshot is not an existing directory.")
        from huggingface_hub import snapshot_download
        source = snapshot_download(repo_id=base, revision=revision, local_files_only=True)
    config = json.loads((Path(source) / "config.json").read_text(encoding="utf-8"))
    assert_certified_model_config(config, "Serving model config", base)
    return str(Path(source).resolve())


def run_server() -> None:
    host = os.environ.get("TT_HOST", "127.0.0.1")
    port = int(os.environ.get("TT_PORT", "8000"))
    # Early diagnostic; upstream still owns the real bind and handles races.
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        probe.bind((host, port))
    model_source = prepare_model_source()
    if os.environ.get("TT_API_KEY"):
        os.environ["VLLM_API_KEY"] = os.environ["TT_API_KEY"]
    with TemporaryDirectory(prefix="tt-serve-") as temp:
        adapter = resolve_adapter_path(os.environ.get("TT_MODEL_ARTIFACT"), Path(temp))
        args = build_vllm_args(model_source, adapter, Path(temp))
        from vllm.entrypoints.cli.main import main
        previous = sys.argv
        try:
            sys.argv = ["vllm", *args]
            main()
        finally:
            sys.argv = previous


if __name__ == "__main__":
    # Ensure temporary artifacts unwind if stopped before upstream installs its
    # own graceful shutdown handlers. Node forwards signals to the whole group.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    run_server()
