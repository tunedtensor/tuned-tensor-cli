from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - depends on the ML venv
    AutoTokenizer = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate import format_prompt  # noqa: E402
from model_contract import NEMOTRON_BASE_MODEL, chat_template_kwargs  # noqa: E402
from sft_data import IGNORE_INDEX, build_assistant_only_example  # noqa: E402


CACHE_ROOT = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
REPO = CACHE_ROOT / "hub" / "models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
SNAPSHOT_REVISION = "ce38b6ab8b252b4b8ee7165b4605e93191cafd73"
SNAPSHOT = REPO / "snapshots" / SNAPSHOT_REVISION


@unittest.skipUnless(
    AutoTokenizer is not None
    and (SNAPSHOT / "config.json").exists(),
    "Nemotron BF16 snapshot and transformers are not present; "
    "downloading the model into the Hugging Face cache is required for "
    "real-chat-template integration tests.",
)
class NemotronChatTemplateIntegrationTests(unittest.TestCase):
    """Exercise training, evaluation, and serving chat-template rendering
    against the real Nemotron tokenizer and chat_template, not fakes."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = AutoTokenizer.from_pretrained(
            str(SNAPSHOT), local_files_only=True, trust_remote_code=False
        )
        cls.messages = [
            {"role": "system", "content": "Classify sentiment as positive or negative."},
            {"role": "user", "content": "I love this new feature"},
            {"role": "assistant", "content": "positive"},
        ]
        cls.kwargs = chat_template_kwargs(NEMOTRON_BASE_MODEL)

    def test_serving_renders_generation_prompt_with_thinking_disabled(self) -> None:
        # Same rendering path serve.py uses for completion requests.
        prompt = format_prompt(self.tokenizer, "Classify sentiment.", "I love this new feature",
                               NEMOTRON_BASE_MODEL)
        self.assertIn("<|im_start|>user", prompt)
        self.assertIn("<|im_start|>assistant", prompt)
        # enable_thinking=False renders an empty thinking block, not a user
        # thinking prompt, and still opens the assistant generation turn.
        self.assertIn("<think></think>", prompt)

    def test_training_sft_example_is_prefix_aligned_and_masked(self) -> None:
        # Same path train.py + sft_data build_assistant_only_example use.
        example = build_assistant_only_example(
            self.tokenizer, self.messages, max_length=256, chat_template_kwargs=self.kwargs
        )
        prompt_ids = self.tokenizer.apply_chat_template(
            self.messages[:-1], tokenize=True, add_generation_prompt=True,
            return_dict=False, **self.kwargs,
        )
        labels = example["labels"]
        self.assertEqual(len(example["input_ids"]), len(labels))
        self.assertTrue(all(token == IGNORE_INDEX for token in labels[: len(prompt_ids)]))
        self.assertTrue(all(token != IGNORE_INDEX for token in labels[len(prompt_ids):]))
        # The assistant completion "positive" labels the masked block.
        completion = self.tokenizer.decode(labels[len(prompt_ids):], skip_special_tokens=True)
        self.assertEqual(completion.strip(), "positive")

    def test_evaluation_chat_render_matches_training_boundary(self) -> None:
        # Training boundary and evaluation prompt must tokenize to a consistent
        # prefix so baseline/candidate scoring compares like-for-like.
        system = "Classify sentiment as positive or negative."
        eval_prompt = format_prompt(self.tokenizer, system,
                                    "I love this new feature", NEMOTRON_BASE_MODEL)
        train_prompt = self.tokenizer.apply_chat_template(
            self.messages[:-1], tokenize=False, add_generation_prompt=True, **self.kwargs
        )
        self.assertEqual(eval_prompt, train_prompt)

    def test_revision_is_pinned_to_the_certified_snapshot(self) -> None:
        # The integration path must only ever target the reviewed revision.
        self.assertEqual(SNAPSHOT_REVISION, "ce38b6ab8b252b4b8ee7165b4605e93191cafd73")
        self.assertTrue(self.tokenizer.vocab_size >= 100_000)


if __name__ == "__main__":
    unittest.main()