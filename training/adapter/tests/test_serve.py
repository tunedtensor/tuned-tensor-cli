from __future__ import annotations

import os
import contextlib
import http.client
import json
import threading
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import subprocess
import sys
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"


sys.path.insert(0, str(SRC))
import serve


class Tensor:
    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values))

    def to(self, device):
        return self

    def __getitem__(self, key):
        if key == 0:
            return self
        return Tensor(self.values[key])


class Tokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        return "prompt"

    def __call__(self, prompt, **kwargs):
        return {"input_ids": Tensor([10, 11])}

    def decode(self, tokens, **kwargs):
        return "hello world"


class Model:
    config = SimpleNamespace(max_position_embeddings=32768)

    def parameters(self):
        yield SimpleNamespace(device="cpu")

    def generate(self, **kwargs):
        self.kwargs = kwargs
        streamer = kwargs.get("streamer")
        if streamer:
            streamer.on_finalized_text("hello ")
            if not self.allow_finish.wait(3):
                raise RuntimeError("stream was not delivered before generation ended")
            streamer.on_finalized_text("world", stream_end=True)
        return [Tensor([10, 11, 20, 21, 2])]


class ServeHTTPTests(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        self.model.allow_finish = threading.Event()
        self.patches = contextlib.ExitStack()
        self.patches.enter_context(patch.multiple(serve, MODEL=self.model, TOKENIZER=Tokenizer(),
            MODEL_NAME="base:test", SYSTEM_PROMPT="", API_KEY="", BASE_MODEL="Qwen/Qwen3.5-2B",
            REQUEST_SLOTS=threading.BoundedSemaphore(1)))
        class TextStreamer:
            def __init__(self, *args, **kwargs):
                pass
        self.patches.enter_context(patch.dict(sys.modules, {
            "torch": SimpleNamespace(inference_mode=contextlib.nullcontext),
            "transformers": SimpleNamespace(TextStreamer=TextStreamer),
        }))
        self.server = serve.ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = http.client.HTTPConnection(*self.server.server_address, timeout=5)

    def tearDown(self):
        self.model.allow_finish.set()
        self.client.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.patches.close()

    def request(self, **extra):
        body = {"model": "base:test", "messages": [{"role": "user", "content": "hi"}], **extra}
        self.client.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        return self.client.getresponse()

    def test_generation_error_sends_no_false_success_and_releases_capacity(self):
        def fail(**kwargs):
            kwargs["streamer"].on_finalized_text("partial ")
            raise RuntimeError("private model data must not leak")
        with patch.object(self.model, "generate", side_effect=fail):
            response = self.request(stream=True)
            data = response.read().decode()
            self.assertEqual(response.status, 200)
            self.assertIn('"error"', data)
            self.assertNotIn("private model data", data)
            self.assertNotIn("[DONE]", data)
        self.assertFalse(serve.GENERATION_LOCK.locked())
        response = self.request()
        self.assertEqual(response.status, 200, response.read())

    def test_disconnect_unwinds_generation_and_releases_capacity(self):
        with patch.object(serve.Handler, "send_event", side_effect=BrokenPipeError) as send:
            response = self.request(stream=True)
            self.assertEqual(response.status, 200)
            self.assertEqual(response.read(), b"")
            send.assert_called_once()
        self.assertFalse(serve.GENERATION_LOCK.locked())
        response = self.request()
        self.assertEqual(response.status, 200, response.read())

    def test_busy_returns_429_and_health_remains_available(self):
        serve.REQUEST_SLOTS.acquire()
        try:
            response = self.request()
            self.assertEqual(response.status, 429, response.read())
            self.client.request("GET", "/health")
            response = self.client.getresponse()
            self.assertEqual(response.status, 200, response.read())
        finally:
            serve.REQUEST_SLOTS.release()

    def test_auth_applies_to_discovery_and_generation(self):
        with patch.object(serve, "API_KEY", "test-secret"):
            response = self.request()
            self.assertEqual(response.status, 401, response.read())
            self.client.request("GET", "/v1/models", headers={"Authorization": "Bearer test-secret"})
            response = self.client.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.read())["data"][0]["id"], "base:test")

    def test_rejects_context_overflow_before_generation(self):
        self.model.config = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=4))
        response = self.request(max_tokens=3)
        self.assertEqual(response.status, 400, response.read())
        self.assertFalse(hasattr(self.model, "kwargs"))

    def test_reports_length_finish_for_both_response_modes(self):
        self.model.allow_finish.set()
        for streaming in (False, True):
            response = self.request(stream=streaming, max_completion_tokens=3)
            self.assertEqual(response.status, 200)
            body = response.read().decode()
            self.assertIn('"finish_reason": "length"', body)
            self.assertEqual(self.model.kwargs["max_new_tokens"], 3)

    def test_rejects_unsupported_or_malformed_requests_before_generation(self):
        cases = [
            {"model": "other-model"}, {"stream": "true"}, {"stream_options": []},
            {"stream_options": {"include_usage": "yes"}}, {"tools": [{"type": "function"}]},
            {"tool_choice": "auto"}, {"functions": [{"name": "read"}]},
            {"response_format": {"type": "json_object"}}, {"n": 2}, {"stop": ["END"]},
            {"messages": [{"role": "tool", "content": "result", "tool_call_id": "1"}]},
            {"messages": [{"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]}]},
            {"max_tokens": True}, {"max_tokens": "12"}, {"temperature": "NaN"},
            {"max_tokens": 2, "max_completion_tokens": 3},
        ]
        for body in cases:
            with self.subTest(body=body):
                response = self.request(**body)
                self.assertEqual(response.status, 400, response.read())
        self.assertFalse(hasattr(self.model, "kwargs"))

    def test_stream_arrives_before_generation_finishes_and_reports_usage(self):
        response = self.request(stream=True, stream_options={"include_usage": True})
        self.assertEqual(response.status, 200, response.read() if response.status != 200 else "")
        self.assertEqual(response.getheader("Content-Type"), "text/event-stream")
        chunks = []
        while True:
            line = response.readline()
            self.assertTrue(line, "stream closed before first content")
            if line.startswith(b"data: {"):
                chunk = json.loads(line[6:])
                chunks.append(chunk)
                if chunk["choices"] and chunk["choices"][0]["delta"].get("content"):
                    break
        self.assertEqual(chunks[-1]["choices"][0]["delta"]["content"], "hello ")
        self.model.allow_finish.set()
        rest = response.read().decode()
        self.assertIn("[DONE]", rest)
        chunks += [json.loads(line[6:]) for line in rest.splitlines() if line.startswith("data: {")]
        self.assertEqual(chunks[-1]["usage"], {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5})
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "stop")
        self.assertTrue(self.model.kwargs["use_cache"])
        self.assertEqual(len({c["id"] for c in chunks}), 1)


class ServeImportTests(unittest.TestCase):
    def test_merge_is_explicit_safe_and_does_not_save_artifacts(self):
        from unittest.mock import Mock
        model = Mock()
        merged = Mock()
        model.merge_and_unload.return_value = merged
        with patch.dict(os.environ, {"TT_MERGE_ADAPTER": "true"}), \
                patch.multiple(serve, BASE_MODEL="Qwen/Qwen3.5-2B", BASE_MODEL_REVISION="15852e8c16360a2fea060d615a32b45270f8a8fc", MODEL_ARTIFACT="/adapter"), \
                patch.object(serve, "import_runtime_dependencies"), \
                patch.object(serve, "configure_hugging_face_cache"), \
                patch.object(serve, "resolve_adapter_path", return_value="/verified-adapter"), \
                patch.object(serve, "load_text_model", return_value=(model, object(), "cpu")):
            serve.load_runtime("/tmp/extraction")
            model.merge_and_unload.assert_called_once_with(safe_merge=True)
            merged.eval.assert_called_once()
            self.assertIs(serve.MODEL, merged)
            model.save_pretrained.assert_not_called()
            merged.save_pretrained.assert_not_called()

    def test_load_runtime_preserves_base_and_adapter_identity(self):
        with patch.multiple(serve, BASE_MODEL="Qwen/Qwen3.5-2B", BASE_MODEL_REVISION="15852e8c16360a2fea060d615a32b45270f8a8fc"), \
                patch.object(serve, "import_runtime_dependencies"), \
                patch.object(serve, "configure_hugging_face_cache"), \
                patch.object(serve, "resolve_adapter_path", return_value="/verified-adapter") as resolve, \
                patch.object(serve, "load_text_model", return_value=(object(), object(), "cpu")) as load:
            for artifact in (None, "/adapter.tar.gz"):
                with patch.object(serve, "MODEL_ARTIFACT", artifact):
                    serve.load_runtime("/tmp/extraction")
                    self.assertEqual(load.call_args.args[0]["base_model"], "Qwen/Qwen3.5-2B")
                    self.assertEqual(load.call_args.args[1], "/verified-adapter" if artifact else None)
            resolve.assert_called_once_with("/adapter.tar.gz", Path("/tmp/extraction"))


    def test_occupied_port_fails_before_loading_weights(self):
        server = serve.ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
        try:
            with patch.multiple(serve, HOST="127.0.0.1", PORT=server.server_address[1]), patch.object(serve, "load_runtime") as load:
                with self.assertRaises(OSError):
                    serve.main()
                load.assert_not_called()
        finally:
            server.server_close()


    def test_import_does_not_load_weights_or_require_environment(self):
        result = subprocess.run(
            [sys.executable, "-c", "import serve; assert serve.MODEL is None"],
            cwd=SRC, env={k: v for k, v in os.environ.items() if not k.startswith("TT_")},
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
