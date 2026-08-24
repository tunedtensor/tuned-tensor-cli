from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from data import IGNORE_INDEX, encode_sft_example, numeric_reward, parse_numeric_answer, train_tokenizer
from model import FoundationGPT, derived_heads, derived_width, model_config_from_depth


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


class FoundationSftMaskTests(unittest.TestCase):
    def test_assistant_tokens_are_the_only_supervised_labels(self) -> None:
        tokenizer = train_tokenizer(
            "<|system|> You are helpful. <|user|> hello <|assistant|> world <|end|>\n" * 16,
            vocab_size=96,
        )
        input_ids, labels = encode_sft_example(tokenizer, "You are helpful.", "hello", "world", 32)
        self.assertEqual(len(input_ids), 32)
        self.assertEqual(len(labels), 32)
        supervised = [token for token, label in zip(input_ids, labels) if label != IGNORE_INDEX]
        self.assertTrue(all(label == token or label == IGNORE_INDEX for token, label in zip(input_ids, labels)))
        self.assertGreater(len(supervised), 0)
        self.assertLess(len(supervised), 32)


class FoundationRlRewardTests(unittest.TestCase):
    def test_parses_the_last_number_and_rewards_exact_matches(self) -> None:
        self.assertEqual(parse_numeric_answer("2 + 2 = 4."), 4.0)
        self.assertEqual(numeric_reward("The answer is 4", "4"), 1.0)
        self.assertEqual(numeric_reward("The answer is 5", "4"), 0.0)
        self.assertEqual(numeric_reward("no digits", "4"), 0.0)


if __name__ == "__main__":
    unittest.main()
