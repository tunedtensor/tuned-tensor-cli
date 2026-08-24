from __future__ import annotations

import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

import rl
from common import write_json
from data import (
    IGNORE_INDEX,
    decoded_byte_count,
    encode_ids,
    encode_sft_example,
    format_chat,
    format_prompt,
    numeric_reward,
    parse_numeric_answer,
    train_tokenizer,
)
from model import FoundationGPT, derived_heads, derived_width, model_config_from_depth


class FoundationOutputTests(unittest.TestCase):
    def test_json_outputs_are_private(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "metrics.json"
            write_json(path, {"ok": True})
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)


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
