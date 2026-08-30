from __future__ import annotations

import stat
import subprocess
import sys
import tempfile
import unittest
import json
import os
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

import evaluate
import pretrain
import rl
import train_tokenizer as tokenizer_entrypoint
from checkpoint import CheckpointManager
from common import write_json
from data import (
    END,
    IGNORE_INDEX,
    decoded_byte_count,
    encode_ids,
    encode_sft_example,
    format_chat,
    format_prompt,
    numeric_reward,
    parse_numeric_answer,
    train_tokenizer,
    iter_corpus_documents,
)
from model import FoundationGPT, derived_heads, derived_width, generate, model_config_from_depth, save_model


class FoundationOutputTests(unittest.TestCase):
    def test_json_outputs_are_private(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "metrics.json"
            write_json(path, {"ok": True})
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_tokenizer_entrypoint_writes_a_private_artifact_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "tokenizer"
            config = root / "config.json"
            config.write_text(
                '{"output_dir":"%s","vocab_size":64,"max_chars":2000,'
                '"system_prompt":"Answer briefly.","examples":['
                '{"input":"hello","output":"world"}]}' % output,
                encoding="utf-8",
            )

            tokenizer_entrypoint.main(["--config", str(config)])

            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE((output / "tokenizer.json").stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE((output / "metrics.json").stat().st_mode), 0o600)

    def test_model_writer_writes_a_private_artifact_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model"
            model = FoundationGPT(model_config_from_depth(1, vocab_size=64, sequence_length=16))
            model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())

            save_model(model, output)

            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE((output / "model.safetensors").stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE((output / "config.json").stat().st_mode), 0o600)


class FoundationEntrypointTests(unittest.TestCase):
    def test_source_directory_does_not_shadow_python_tokenize_module(self) -> None:
        source_dir = Path(__file__).resolve().parents[1] / "src"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import tokenize; assert hasattr(tokenize, 'open'), tokenize.__file__",
            ],
            cwd=source_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class FoundationModelTests(unittest.TestCase):
    def test_depth_derives_width_and_heads(self) -> None:
        self.assertEqual(derived_width(2), 64)
        self.assertEqual(derived_heads(2), 8)
        config = model_config_from_depth(2, vocab_size=128, sequence_length=16)
        model = FoundationGPT(config)
        logits = model(torch.zeros(2, 8, dtype=torch.long))
        self.assertEqual(tuple(logits.shape), (2, 8, 128))

    def test_generation_stops_when_every_row_emits_the_stop_token(self) -> None:
        class FakeModel:
            config = {"sequence_length": 8}

            def __init__(self) -> None:
                self.calls = 0

            def eval(self) -> None:
                return None

            def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
                logits = torch.zeros(tokens.size(0), tokens.size(1), 8)
                next_id = 7 if self.calls == 0 else 5
                logits[:, -1, next_id] = 1
                self.calls += 1
                return logits

        model = FakeModel()
        prompt = torch.tensor([[1, 2]], dtype=torch.long)

        output = generate(model, prompt, max_new_tokens=8, stop_token_id=5)

        self.assertEqual(output.tolist(), [[1, 2, 7, 5]])
        self.assertEqual(model.calls, 2)


class FoundationEvaluationTests(unittest.TestCase):
    def test_chat_evaluation_stops_at_the_tokenizer_end_token(self) -> None:
        class FakeEncoding:
            ids = [1, 2]

        class FakeTokenizer:
            def encode(self, _text: str) -> FakeEncoding:
                return FakeEncoding()

            def token_to_id(self, _token: str) -> int:
                return 5

            def decode(self, ids: list[int]) -> str:
                return "0" if ids == [7, 5] else "0 extra"

        class FakeModel:
            config = {"sequence_length": 8}

            def __init__(self) -> None:
                self.calls = 0

            def eval(self) -> None:
                return None

            def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
                logits = torch.zeros(tokens.size(0), tokens.size(1), 8)
                next_id = 7 if self.calls == 0 else 5
                logits[:, -1, next_id] = 1
                self.calls += 1
                return logits

        metrics = cast(
            dict[str, Any],
            evaluate.evaluate_chat(
                cast(FoundationGPT, FakeModel()),
                cast(Any, FakeTokenizer()),
                "Return one digit.",
                [{"input": "parity digits 1 0", "output": "0"}],
                torch.device("cpu"),
            ),
        )

        self.assertEqual(metrics["correct"], 1)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["predictions"][0]["actual"], "0")


class FoundationTokenizerTests(unittest.TestCase):
    def test_whitespace_bpe_round_trips_words(self) -> None:
        tokenizer = train_tokenizer("hello world hello there hello world\n" * 32, vocab_size=64)
        encoded = tokenizer.encode("hello world")
        self.assertGreater(len(encoded.ids), 0)
        self.assertIn("hello", tokenizer.decode(encoded.ids))

    def test_byte_count_sums_every_target_row(self) -> None:
        class FakeTokenizer:
            def decode(self, ids: list[int]) -> str:
                return "a" if ids[0] == 1 else "éé"

        self.assertEqual(decoded_byte_count(FakeTokenizer(), [[1, 9], [2, 9]]), 5)

    def test_streams_text_and_jsonl_corpora_with_a_global_character_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("alpha", encoding="utf-8")
            (root / "b.jsonl").write_text(
                '{"text":"beta"}\n{"text":"gamma"}\n',
                encoding="utf-8",
            )
            self.assertEqual("".join(iter_corpus_documents(root, 8)), "alphabet")

    def test_rejects_jsonl_without_a_text_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text('{"content":"wrong"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, 'string "text" field'):
                list(iter_corpus_documents(path))


class FoundationPretrainReliabilityTests(unittest.TestCase):
    def test_token_cache_resumes_from_published_document_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            import hashlib
            import numpy as np
            root = Path(directory)
            corpus = root / "corpus.jsonl"
            corpus.write_text(
                '{"text":"alpha beta"}\n{"text":"gamma delta"}\n{"text":"epsilon zeta"}\n',
                encoding="utf-8",
            )
            tokenizer = train_tokenizer("alpha beta gamma delta epsilon zeta " * 16, vocab_size=64)
            tokenizer_path = root / "tokenizer.json"
            tokenizer.save(str(tokenizer_path))
            output = root / "work"
            output.mkdir()
            first_ids = tokenizer.encode("alpha beta").ids
            end_id = tokenizer.token_to_id("<|end|>")
            if end_id is not None:
                first_ids.append(end_id)
            first = np.asarray(first_ids, dtype=np.uint32).tobytes()
            partial = output / ".pretrain-tokens.partial"
            partial.write_bytes(first + b"discarded-unpublished-tail")
            config = {"corpus_path": str(corpus), "max_chars": 1000}
            signature = {
                "source": pretrain.source_signature(config),
                "tokenizer_sha256": pretrain.file_sha256(tokenizer_path),
            }
            (output / ".pretrain-tokens.partial.json").write_text(json.dumps({
                "signature": signature,
                "tokens": len(first_ids),
                "documents": 1,
                "bytes": len(first),
                "tokens_sha256": hashlib.sha256(first).hexdigest(),
            }), encoding="utf-8")
            tokens, metadata = pretrain.prepare_token_cache(tokenizer, tokenizer_path, config, output)
            expected: list[int] = []
            for text in ["alpha beta", "gamma delta", "epsilon zeta"]:
                expected.extend(tokenizer.encode(text).ids)
                if end_id is not None:
                    expected.append(end_id)
            self.assertEqual(tokens.tolist(), expected)
            self.assertEqual(metadata["documents"], 3)

    def test_token_cache_rebuilds_same_size_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.txt"
            corpus.write_text("alpha beta gamma delta", encoding="utf-8")
            tokenizer = train_tokenizer("alpha beta gamma delta " * 16, vocab_size=64)
            tokenizer_path = root / "tokenizer.json"
            tokenizer.save(str(tokenizer_path))
            output = root / "work"
            output.mkdir()
            config = {"corpus_path": str(corpus), "max_chars": 1000}

            tokens, _metadata = pretrain.prepare_token_cache(tokenizer, tokenizer_path, config, output)
            expected = tokens.tolist()
            del tokens
            cache_path = output / "pretrain-tokens.bin"
            original = cache_path.read_bytes()
            cache_path.write_bytes(b"\xff\xff\xff\xff" + original[4:])

            rebuilt, metadata = pretrain.prepare_token_cache(tokenizer, tokenizer_path, config, output)

            self.assertEqual(rebuilt.tolist(), expected)
            self.assertEqual(metadata["tokens_sha256"], pretrain.file_sha256(cache_path))

    def test_token_cache_discards_corrupt_published_partial_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            import hashlib
            import numpy as np
            root = Path(directory)
            corpus = root / "corpus.jsonl"
            corpus.write_text(
                '{"text":"alpha beta"}\n{"text":"gamma delta"}\n',
                encoding="utf-8",
            )
            tokenizer = train_tokenizer("alpha beta gamma delta " * 16, vocab_size=64)
            tokenizer_path = root / "tokenizer.json"
            tokenizer.save(str(tokenizer_path))
            output = root / "work"
            output.mkdir()
            first_ids = tokenizer.encode("alpha beta").ids
            end_id = tokenizer.token_to_id(END)
            if end_id is not None:
                first_ids.append(end_id)
            first = np.asarray(first_ids, dtype=np.uint32).tobytes()
            partial = output / ".pretrain-tokens.partial"
            partial.write_bytes(b"\xff\xff\xff\xff" + first[4:])
            config = {"corpus_path": str(corpus), "max_chars": 1000}
            signature = {
                "source": pretrain.source_signature(config),
                "tokenizer_sha256": pretrain.file_sha256(tokenizer_path),
            }
            (output / ".pretrain-tokens.partial.json").write_text(json.dumps({
                "signature": signature,
                "tokens": len(first_ids),
                "documents": 1,
                "source_documents": 1,
                "bytes": len(first),
                "tokens_sha256": hashlib.sha256(first).hexdigest(),
            }), encoding="utf-8")

            rebuilt, metadata = pretrain.prepare_token_cache(tokenizer, tokenizer_path, config, output)
            expected: list[int] = []
            for text in ["alpha beta", "gamma delta"]:
                expected.extend(tokenizer.encode(text).ids)
                if end_id is not None:
                    expected.append(end_id)

            self.assertEqual(rebuilt.tolist(), expected)
            self.assertEqual(metadata["documents"], 2)

    def test_token_cache_rebuilds_corrupt_cache_from_interrupted_publish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            import hashlib
            import numpy as np
            root = Path(directory)
            corpus = root / "corpus.txt"
            corpus.write_text("alpha beta", encoding="utf-8")
            tokenizer = train_tokenizer("alpha beta " * 16, vocab_size=64)
            tokenizer_path = root / "tokenizer.json"
            tokenizer.save(str(tokenizer_path))
            output = root / "work"
            output.mkdir()
            ids = tokenizer.encode("alpha beta").ids
            end_id = tokenizer.token_to_id(END)
            if end_id is not None:
                ids.append(end_id)
            encoded = np.asarray(ids, dtype=np.uint32).tobytes()
            (output / "pretrain-tokens.bin").write_bytes(b"\xff\xff\xff\xff" + encoded[4:])
            config = {"corpus_path": str(corpus), "max_chars": 1000}
            signature = {
                "source": pretrain.source_signature(config),
                "tokenizer_sha256": pretrain.file_sha256(tokenizer_path),
            }
            (output / ".pretrain-tokens.partial.json").write_text(json.dumps({
                "signature": signature,
                "tokens": len(ids),
                "documents": 1,
                "source_documents": 1,
                "bytes": len(encoded),
                "tokens_sha256": hashlib.sha256(encoded).hexdigest(),
            }), encoding="utf-8")

            rebuilt, metadata = pretrain.prepare_token_cache(tokenizer, tokenizer_path, config, output)

            self.assertEqual(rebuilt.tolist(), ids)
            self.assertEqual(metadata["tokens_sha256"], pretrain.file_sha256(output / "pretrain-tokens.bin"))

    def test_token_batches_are_deterministic_and_advance_with_the_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.bin"
            import numpy as np
            np.arange(256, dtype=np.uint32).tofile(path)
            tokens = np.memmap(path, dtype=np.uint32, mode="r")
            first = pretrain.token_batch(tokens, samples_consumed=0, batch_size=2, sequence_length=8, seed=7)
            repeated = pretrain.token_batch(tokens, samples_consumed=0, batch_size=2, sequence_length=8, seed=7)
            advanced = pretrain.token_batch(tokens, samples_consumed=2, batch_size=2, sequence_length=8, seed=7)
            self.assertTrue(np.array_equal(first, repeated))
            self.assertFalse(np.array_equal(first, advanced))

    def test_learning_rate_warms_up_and_cosine_decays_to_the_floor(self) -> None:
        values = [
            pretrain.learning_rate_multiplier(step, warmup_steps=2, total_steps=10, min_ratio=0.1)
            for step in range(11)
        ]
        self.assertLess(values[0], values[1])
        self.assertAlmostEqual(values[1], 1.0)
        self.assertAlmostEqual(values[-1], 0.1)

    def test_checkpoint_round_trip_restores_training_and_cursor_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(4, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            loss = model(torch.ones(2, 4)).sum()
            loss.backward()
            optimizer.step()
            scheduler.step()
            expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
            manager = CheckpointManager(directory, "config-a", keep=2)
            manager.save(
                model,
                optimizer,
                scheduler,
                step=3,
                samples_consumed=12,
                tokens_seen=384,
                last_loss=1.25,
                reason="test",
            )
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            resumed = manager.load(model, optimizer, scheduler, torch.device("cpu"))
            self.assertIsNotNone(resumed)
            assert resumed is not None
            self.assertEqual((resumed.step, resumed.samples_consumed, resumed.tokens_seen), (3, 12, 384))
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, expected[key]))

    def test_checkpoint_resume_loads_verified_backup_when_primary_is_lost(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as backup:
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(model.parameters())
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            manager = CheckpointManager(directory, "stable", backup_dir=backup)
            manager.save(
                model, optimizer, scheduler,
                step=2, samples_consumed=4, tokens_seen=8, last_loss=1.0, reason="periodic",
            )
            __import__("shutil").rmtree(Path(directory) / "checkpoints")
            replacement = CheckpointManager(directory, "stable", backup_dir=backup)

            resumed = replacement.load(model, optimizer, scheduler, torch.device("cpu"))

            self.assertIsNotNone(resumed)
            assert resumed is not None
            self.assertEqual(resumed.step, 2)
            self.assertEqual(resumed.path.parent, Path(backup).resolve())

    def test_checkpoint_save_is_idempotent_for_an_existing_verified_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(model.parameters())
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            manager = CheckpointManager(directory, "stable")
            first = manager.save(
                model, optimizer, scheduler,
                step=1, samples_consumed=1, tokens_seen=2, last_loss=1.0, reason="periodic",
            )
            original = (first / "model.safetensors").read_bytes()
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(1)

            repeated = manager.save(
                model, optimizer, scheduler,
                step=1, samples_consumed=1, tokens_seen=2, last_loss=2.0, reason="non-finite-loss",
            )

            self.assertEqual(repeated, first)
            self.assertEqual((first / "model.safetensors").read_bytes(), original)

    def test_checkpoint_backup_rejects_relative_alias_of_primary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            os.chdir(directory)
            try:
                with self.assertRaisesRegex(ValueError, "outside the primary"):
                    CheckpointManager("run", "stable", backup_dir="run/checkpoints")
            finally:
                os.chdir(previous)

    def test_checkpoint_resume_rejects_configuration_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(model.parameters())
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            CheckpointManager(directory, "original").save(
                model,
                optimizer,
                scheduler,
                step=1,
                samples_consumed=1,
                tokens_seen=2,
                last_loss=1.0,
                reason="test",
            )
            with self.assertRaisesRegex(ValueError, "do not match"):
                CheckpointManager(directory, "changed").load(
                    model, optimizer, scheduler, torch.device("cpu"),
                )

    def test_checkpoint_resume_rejects_unreadable_only_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = CheckpointManager(directory, "stable")
            checkpoint = Path(directory) / "checkpoints" / "step-000000001"
            checkpoint.mkdir()
            (checkpoint / "metadata.json").write_text("{", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "No intact foundation checkpoint"):
                manager.latest()

    def test_checkpoint_resume_rejects_incomplete_only_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = CheckpointManager(directory, "stable")
            checkpoint = Path(directory) / "checkpoints" / "step-000000001"
            checkpoint.mkdir()
            (checkpoint / "metadata.json").write_text(
                json.dumps({"complete": False, "config_fingerprint": "stable"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "No intact foundation checkpoint"):
                manager.latest()

    def test_checkpoint_resume_rejects_checkpoint_with_missing_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = CheckpointManager(directory, "stable")
            checkpoint = Path(directory) / "checkpoints" / "step-000000001"
            checkpoint.mkdir()
            (checkpoint / "metadata.json").write_text(json.dumps({
                "complete": True,
                "config_fingerprint": "stable",
                "files": {},
            }), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "No intact foundation checkpoint"):
                manager.latest()

    def test_checkpoint_resume_falls_back_when_latest_bytes_are_corrupt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(model.parameters())
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            manager = CheckpointManager(directory, "stable", keep=2)
            manager.save(
                model, optimizer, scheduler,
                step=1, samples_consumed=1, tokens_seen=2, last_loss=2.0, reason="periodic",
            )
            manager.save(
                model, optimizer, scheduler,
                step=2, samples_consumed=2, tokens_seen=4, last_loss=1.0, reason="periodic",
            )
            latest_model = Path(directory) / "checkpoints" / "step-000000002" / "model.safetensors"
            latest_model.write_bytes(b"corrupt")
            resumed = manager.load(model, optimizer, scheduler, torch.device("cpu"))
            self.assertIsNotNone(resumed)
            assert resumed is not None
            self.assertEqual(resumed.step, 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA failure-injection test")
    def test_cuda_pretrain_handles_sigterm_and_resumes_to_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer_dir = root / "tokenizer"
            tokenizer_dir.mkdir()
            tokenizer = train_tokenizer("alpha beta gamma delta " * 256, vocab_size=64)
            tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
            config = root / "config.json"
            payload = {
                "tokenizer_dir": str(tokenizer_dir),
                "output_dir": str(root / "output"),
                "work_dir": str(root / "recovery"),
                "depth": 2,
                "steps": 80,
                "batch_size": 2,
                "sequence_length": 32,
                "max_chars": 20_000,
                "system_prompt": "alpha beta",
                "examples": [{"input": "alpha", "output": "beta"}],
                "checkpoint_interval_steps": 5,
                "checkpoint_interval_seconds": 3600,
                "log_interval_steps": 2,
                "keep_checkpoints": 2,
                "resume": True,
            }
            config.write_text(json.dumps(payload), encoding="utf-8")
            command = [sys.executable, str(Path(__file__).parents[1] / "src" / "pretrain.py"), "--config", str(config)]
            first = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            deadline = __import__("time").monotonic() + 30
            checkpoint_root = root / "recovery" / "checkpoints"
            while __import__("time").monotonic() < deadline and first.poll() is None:
                if checkpoint_root.exists() and list(checkpoint_root.glob("step-*")):
                    first.send_signal(__import__("signal").SIGTERM)
                    break
                __import__("time").sleep(0.05)
            first.communicate(timeout=30)
            self.assertNotEqual(first.returncode, 0)
            second = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn('"event": "resumed"', second.stdout)
            metrics = json.loads((root / "output" / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["steps"], 80)
            self.assertGreater(metrics["tokens_seen"], 0)


class FoundationSftMaskTests(unittest.TestCase):
    def test_assistant_targets_are_shifted_for_next_token_prediction(self) -> None:
        system_prompt = "You are helpful."
        user = "hello"
        assistant = "world"
        tokenizer = train_tokenizer(
            "<|system|> You are helpful. <|user|> hello <|assistant|> world <｜end｜>\n" * 16,
            vocab_size=96,
        )
        input_ids, labels = encode_sft_example(
            tokenizer,
            system_prompt,
            user,
            assistant,
            32,
        )
        prefix_ids = encode_ids(tokenizer, format_prompt(system_prompt, user))
        full_ids = encode_ids(tokenizer, format_chat(system_prompt, user, assistant))
        first_assistant_target = len(prefix_ids) - 1

        self.assertEqual(len(input_ids), 32)
        self.assertEqual(len(labels), 32)
        self.assertEqual(input_ids[: len(full_ids) - 1], full_ids[:-1])
        self.assertTrue(all(label == IGNORE_INDEX for label in labels[:first_assistant_target]))
        self.assertEqual(labels[first_assistant_target], full_ids[len(prefix_ids)])
        self.assertEqual(
            labels[first_assistant_target:first_assistant_target + len(full_ids) - len(prefix_ids)],
            full_ids[len(prefix_ids):],
        )


class FoundationRlRewardTests(unittest.TestCase):
    def test_zero_reward_penalizes_the_sampled_completion(self) -> None:
        token_log_probs = torch.tensor([-0.25], requires_grad=True)

        loss = rl.policy_gradient_loss(token_log_probs, reward=0.0)
        loss.backward()

        self.assertGreater(float(token_log_probs.grad.item()), 0.0)

    def test_rollout_fits_prompt_and_completion_inside_model_context(self) -> None:
        prompt, completion_tokens = rl.bounded_rollout(list(range(40)), 16)
        self.assertEqual(prompt, list(range(32, 40)))
        self.assertEqual(completion_tokens, 8)
        self.assertLessEqual(len(prompt) + completion_tokens, 16)

    def test_rollout_samples_from_the_policy_distribution(self) -> None:
        model = FoundationGPT(model_config_from_depth(1, vocab_size=16, sequence_length=8))
        prompt = torch.tensor([[1, 2]], dtype=torch.long)
        chosen = torch.tensor([[7]], dtype=torch.long)

        with patch.object(rl.torch, "multinomial", return_value=chosen) as sample:
            rollout = rl.sample_rollout(model, prompt, max_new_tokens=2)

        self.assertEqual(rollout.tolist(), [[1, 2, 7, 7]])
        self.assertEqual(sample.call_count, 2)

    def test_parses_the_last_number_and_rewards_exact_matches(self) -> None:
        self.assertEqual(parse_numeric_answer("2 + 2 = 4."), 4.0)
        self.assertEqual(numeric_reward("The answer is 4", "4"), 1.0)
        self.assertEqual(numeric_reward("The answer is 5", "4"), 0.0)
        self.assertEqual(numeric_reward("no digits", "4"), 0.0)


if __name__ == "__main__":
    unittest.main()
