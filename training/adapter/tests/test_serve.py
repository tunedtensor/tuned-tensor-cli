from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import serve


class UpstreamLaunchTests(unittest.TestCase):

    def test_auth_covers_upstream_admin_routes_not_only_v1(self):
        import asyncio
        calls = []
        async def app(scope, receive, send):
            calls.append(scope["path"])
        async def receive():
            return {"type": "http.request"}
        async def exercise():
            with patch.dict(os.environ, {"TT_API_KEY": "fixture-token"}):
                middleware = serve.BearerAuthMiddleware(app)
            for path in ("/health", "/tokenize", "/sleep", "/v1/models"):
                for auth in (False, True):
                    sent = []
                    async def send(message):
                        sent.append(message)
                    scope = {"type": "http", "path": path, "headers": [(b"authorization", b"bearer fixture-token")] if auth else []}
                    await middleware(scope, receive, send)
                    if auth:
                        self.assertEqual(calls[-1], path)
                    else:
                        self.assertEqual(sent[0]["status"], 401)
            self.assertEqual(len(calls), 4)
        asyncio.run(exercise())


    def test_larger_models_use_their_upstream_tool_parsers(self):
        for model, parser in [
            ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", "qwen3_coder"),
            ("meta-models/Muse-Glimmer-30B", "muse_glimmer"),
        ]:
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                with patch.dict(os.environ, {"TT_BASE_MODEL": model, "TT_MODEL_NAME": "base"}, clear=True):
                    args = serve.build_vllm_args("/verified/snapshot", None, Path(tmp))
                self.assertEqual(args[args.index("--tool-call-parser") + 1], parser)
                self.assertIn("--enable-auto-tool-choice", args)

    def test_launch_uses_upstream_server_and_native_qwen_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {
                "TT_BASE_MODEL": "Qwen/Qwen3.5-2B",
                "TT_BASE_MODEL_REVISION": "15852e8c16360a2fea060d615a32b45270f8a8fc",
                "TT_MODEL_NAME": "base:Qwen/Qwen3.5-2B",
                "TT_HOST": "127.0.0.1", "TT_PORT": "8123",
                "TT_MAX_TOKENS": "128", "TT_MAX_CONCURRENT_REQUESTS": "2",
            }, clear=True):
                args = serve.build_vllm_args("/verified/snapshot", None, Path(tmp))
        self.assertEqual(args[:2], ["serve", "/verified/snapshot"])
        self.assertEqual(args[args.index("--served-model-name") + 1], "base:Qwen/Qwen3.5-2B")
        self.assertEqual(args[args.index("--tool-call-parser") + 1], "qwen3_xml")
        self.assertIn("--enable-auto-tool-choice", args)
        self.assertIn("--language-model-only", args)
        self.assertIn("--generation-config", args)
        self.assertNotIn("--trust-remote-code", args)
        self.assertIn("--no-enable-log-requests", args)
        self.assertNotIn("--enable-lora", args)

    def test_adapter_is_registered_alongside_base_without_loading_weights(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(json.dumps({
                "peft_type": "LORA", "r": 16, "base_model_name_or_path": "Qwen/Qwen3.5-2B"
            }))
            with patch.dict(os.environ, {"TT_BASE_MODEL": "Qwen/Qwen3.5-2B", "TT_MODEL_NAME": "local-tuned"}, clear=True):
                args = serve.build_vllm_args("/verified/snapshot", str(adapter), root)
            self.assertEqual(args[args.index("--served-model-name") + 1], "base:Qwen/Qwen3.5-2B")
            self.assertEqual(json.loads(args[args.index("--lora-modules") + 1]), {
                "name": "local-tuned", "path": str(adapter), "base_model_name": "base:Qwen/Qwen3.5-2B"
            })
            self.assertEqual(args[args.index("--max-lora-rank") + 1], "16")

    def test_owner_prompt_is_literal_and_tool_history_is_preserved(self):
        import json
        try:
            from jinja2 import Environment
        except ImportError:
            self.skipTest("Template rendering is also exercised in the locked-runtime integration smoke")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "snapshot"
            source.mkdir()
            original = "{{ messages | tojson }}"
            (source / "chat_template.jinja").write_text(original)
            owner = 'Owner: {{ dangerous }} "quoted"'
            with patch.dict(os.environ, {"TT_BASE_MODEL": "Qwen/Qwen3.5-2B", "TT_MODEL_NAME": "base", "TT_SYSTEM_PROMPT": owner}, clear=True):
                args = serve.build_vllm_args(str(source), None, root)
            template = Path(args[args.index("--chat-template") + 1]).read_text()
            history = [{"role": "system", "content": "Client context"}, {"role": "user", "content": "hi"},
                {"role": "assistant", "tool_calls": [{"id": "call1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "call1", "content": "result"}]
            messages = json.loads(Environment().from_string(template).render(messages=history))
            self.assertEqual(messages[0], {"role": "system", "content": owner + "\n\nClient context"})
            self.assertEqual(messages[1:], history[1:])
            self.assertEqual((source / "chat_template.jinja").read_text(), original)

    def test_bootstrap_delegates_to_vllm_and_keeps_credentials_out_of_argv(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        upstream = Mock()
        with patch.dict(os.environ, {"TT_BASE_MODEL": "Qwen/Qwen3.5-2B", "TT_BASE_MODEL_REVISION": "15852e8c16360a2fea060d615a32b45270f8a8fc", "TT_MODEL_NAME": "base", "TT_PORT": "0", "TT_API_KEY": "fixture-secret"}, clear=True):
            with patch.object(serve, "prepare_model_source", return_value="/verified/snapshot"):
                with patch.dict(sys.modules, {"vllm.entrypoints.cli.main": SimpleNamespace(main=upstream)}):
                    serve.run_server()
            upstream.assert_called_once()
            self.assertEqual(os.environ["VLLM_API_KEY"], "fixture-secret")
            self.assertNotIn("fixture-secret", " ".join(sys.argv))
        self.assertFalse(hasattr(serve, "Handler"), "TT must not retain a second HTTP server")

    def test_import_is_safe_without_runtime_or_environment(self):
        import subprocess
        subprocess.run([sys.executable, "-c", "import sys;sys.path.insert(0,sys.argv[1]);import serve;assert 'torch' not in sys.modules;assert 'vllm' not in sys.modules", str(Path(serve.__file__).parent)], env={"PATH": os.environ.get("PATH", "")}, check=True)

    def test_port_conflict_fails_before_model_preparation(self):
        import socket
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            with patch.dict(os.environ, {"TT_HOST": "127.0.0.1", "TT_PORT": str(listener.getsockname()[1])}):
                with patch.object(serve, "prepare_model_source") as prepare:
                    with self.assertRaises(OSError):
                        serve.run_server()
                    prepare.assert_not_called()

    def test_missing_explicit_snapshot_never_falls_back_to_hub(self):
        with patch.dict(os.environ, {"TT_BASE_MODEL": "Qwen/Qwen3.5-2B", "TT_BASE_MODEL_REVISION": "15852e8c16360a2fea060d615a32b45270f8a8fc", "TT_MODEL_SOURCE": "/nonexistent-tt-serving-snapshot"}, clear=True):
            with self.assertRaisesRegex(ValueError, "Explicit serving snapshot"):
                serve.prepare_model_source()

    def test_revision_mismatch_fails_before_any_snapshot_lookup(self):
        with patch.dict(os.environ, {"TT_BASE_MODEL": "Qwen/Qwen3.5-2B", "TT_BASE_MODEL_REVISION": "wrong"}, clear=True):
            with self.assertRaisesRegex(ValueError, "Serving base model revision must be"):
                serve.prepare_model_source()

    def test_adapter_identity_mismatch_is_rejected(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "adapter_config.json").write_text(json.dumps({"r": 16, "peft_type": "LORA", "base_model_name_or_path": "wrong-model"}))
            with patch.dict(os.environ, {"TT_BASE_MODEL": "Qwen/Qwen3.5-2B", "TT_MODEL_NAME": "tuned"}, clear=True):
                with self.assertRaisesRegex(ValueError, "adapter base"):
                    serve.build_vllm_args("/verified/snapshot", tmp, root)
