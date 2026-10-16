from __future__ import annotations

import contextlib
import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import evaluate


class FakeTensor:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def to(self, device):
        return self


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    padding_side = "left"

    def apply_chat_template(self, messages, **kwargs):
        return messages[-1]["content"]

    def __call__(self, prompts, **kwargs):
        assert kwargs == {"return_tensors": "pt", "padding": True, "add_special_tokens": False}
        width = max(map(len, prompts))
        rows = [[0] * (width - len(prompt)) + list(map(ord, prompt)) for prompt in prompts]
        return {"input_ids": FakeTensor(rows),
                "attention_mask": FakeTensor([[int(token != 0) for token in row] for row in rows])}

    def decode(self, tokens, **kwargs):
        return "".join(chr(token) for token in tokens if token)


class OutOfMemoryError(RuntimeError):
    pass


class FakeModel:
    def __init__(self, fail_call=None, max_batch=None):
        self.calls = []
        self.fail_call = fail_call
        self.max_batch = max_batch

    def parameters(self):
        yield SimpleNamespace(device="cpu")

    def generate(self, **kwargs):
        rows = kwargs["input_ids"].rows
        self.calls.append(len(rows))
        if self.fail_call == len(self.calls):
            raise RuntimeError("interrupted")
        if self.max_batch is not None and len(rows) > self.max_batch:
            raise OutOfMemoryError("fixture allocation failure")
        assert kwargs["pad_token_id"] == 0
        assert kwargs["do_sample"] is False
        assert kwargs["max_new_tokens"] == 8
        for row, mask in zip(rows, kwargs["attention_mask"].rows):
            assert row[-1] != 0  # Every prompt must end on a real token.
            assert mask == [int(token != 0) for token in row]
        return [row + list(map(ord, "reply:")) + [token for token in row if token] for row in rows]


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.checkpoint = Path(self.tmp.name) / "checkpoints" / "eval.json"
        self.payload = {
            "base_model": "Qwen/Qwen3.5-2B", "batch_size": 2, "checkpoint_key": "pinned-inputs",
            "generation": {"max_new_tokens": 8, "temperature": 0},
            "examples": [{"id": str(i), "input": text} for i, text in enumerate(
                ["a", "a much longer prompt", "third", "fourth", "last"])],
        }
        runtime = SimpleNamespace(inference_mode=contextlib.nullcontext,
                                  cuda=SimpleNamespace(OutOfMemoryError=OutOfMemoryError, empty_cache=lambda: None))
        self.runtime = runtime
        self.torch_patch = patch.object(evaluate, "torch", runtime, create=True)
        self.torch_patch.start()
        self.addCleanup(self.torch_patch.stop)
        self.progress = []
        progress_patch = patch.object(evaluate, "emit_progress", side_effect=lambda *args: self.progress.append(args))
        progress_patch.start()
        self.addCleanup(progress_patch.stop)

    def run_batches(self, model, payload=None, adapter=None, device="cpu"):
        with patch.object(evaluate, "load_text_model", return_value=(model, FakeTokenizer(), device)):
            return evaluate.evaluate_batches(payload or self.payload, adapter, self.checkpoint, "causal_lm")

    def test_variable_length_prompts_are_batched_without_prompt_leakage(self):
        model = FakeModel()
        results = self.run_batches(model)
        self.assertEqual(model.calls, [2, 2, 1])
        self.assertEqual([item["id"] for item in results], ["0", "1", "2", "3", "4"])
        self.assertEqual([item["actual"] for item in results],
                         ["reply:" + example["input"] for example in self.payload["examples"]])
        self.assertEqual(self.progress[-1][1:3], (5, 5))
        self.assertEqual(self.progress[-1][-1], 0)
        if os.name != "nt":
            self.assertEqual(self.checkpoint.stat().st_mode & 0o777, 0o600)

    def test_interrupted_evaluation_resumes_only_completed_batches(self):
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.run_batches(FakeModel(fail_call=2))
        self.assertEqual(len(json.loads(self.checkpoint.read_text())["results"]), 2)
        retry = FakeModel()
        results = self.run_batches(retry)
        self.assertEqual(retry.calls, [2, 1])
        self.assertEqual(len(results), 5)
        self.assertIn(("resuming", 2, 5, 2), self.progress)
        with patch.object(evaluate, "load_text_model", side_effect=AssertionError("must not load")):
            self.assertEqual(evaluate.evaluate_batches(self.payload, None, self.checkpoint, "causal_lm"), results)

    def test_changed_inputs_or_adapter_bytes_invalidate_checkpoint(self):
        adapter = Path(self.tmp.name) / "adapter"
        adapter.mkdir()
        weights = adapter / "adapter_model.safetensors"
        weights.write_bytes(b"original")
        self.run_batches(FakeModel(), adapter=str(adapter))
        for payload, content in [(self.payload, b"retrained"),
                                 ({**self.payload, "checkpoint_key": "new-inputs"}, b"retrained")]:
            weights.write_bytes(content)
            model = FakeModel()
            self.run_batches(model, payload, str(adapter))
            self.assertEqual(model.calls, [2, 2, 1])

    def test_corrupt_or_invalid_checkpoints_never_supply_predictions(self):
        for content in ["{", "[]", json.dumps({"identity": "wrong", "results": []})]:
            self.checkpoint.parent.mkdir(exist_ok=True)
            self.checkpoint.write_text(content)
            model = FakeModel()
            self.run_batches(model)
            self.assertEqual(model.calls, [2, 2, 1])
        original = json.loads(self.checkpoint.read_text())
        for result in [{"id": "wrong", "actual": "bad", "latency_ms": 0},
                       {"id": "0", "actual": 7, "latency_ms": 0},
                       {"id": "0", "actual": "bad", "latency_ms": -1}]:
            self.checkpoint.write_text(json.dumps({**original, "results": [result]}))
            model = FakeModel()
            self.run_batches(model)
            self.assertEqual(model.calls, [2, 2, 1])

    def test_cuda_oom_reduces_batch_and_retries_without_skipping_examples(self):
        model = FakeModel(max_batch=2)
        results = self.run_batches(model, {**self.payload, "batch_size": 4}, device="cuda")
        self.assertEqual(model.calls, [4, 2, 2, 1])
        self.assertEqual([item["id"] for item in results], ["0", "1", "2", "3", "4"])
        self.assertIn(("reducing_batch", 0, 5, 2), self.progress)

    def test_single_example_and_cpu_oom_fail_without_retry_loops(self):
        for device, size in [("cuda", 1), ("cpu", 2)]:
            model = FakeModel(max_batch=0)
            with self.assertRaises(OutOfMemoryError):
                self.run_batches(model, {**self.payload, "batch_size": size}, device=device)
            self.assertEqual(model.calls, [size])

    def test_failed_atomic_replace_preserves_previous_checkpoint(self):
        self.run_batches(FakeModel())
        before = self.checkpoint.read_bytes()
        with patch.object(evaluate.os, "replace", side_effect=OSError("disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                evaluate.write_json_atomic(self.checkpoint, {"results": []})
        self.assertEqual(self.checkpoint.read_bytes(), before)
        self.assertEqual(list(self.checkpoint.parent.iterdir()), [self.checkpoint])

    def test_unstable_sources_do_not_resume_and_invalid_batch_sizes_are_rejected(self):
        self.run_batches(FakeModel())
        model = FakeModel()
        self.run_batches(model, {**self.payload, "checkpoint_key": None})
        self.assertEqual(model.calls, [2, 2, 1])
        for size in [0, -1, 129, 1.5, True]:
            with self.assertRaisesRegex(ValueError, "batch_size"):
                self.run_batches(FakeModel(), {**self.payload, "batch_size": size})


if __name__ == "__main__":
    unittest.main()
