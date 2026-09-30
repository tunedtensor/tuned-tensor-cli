"""CPU tests for the decision runtime against a tiny, locally built Laya-format checkpoint.

No download: the checkpoint is a randomly initialised two-layer ModernBERT with the
Laya decision head, a word-level tokenizer, and the files `laya.Agent` requires.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from laya.common import DecisionModel
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoModel, ModernBertConfig, PreTrainedTokenizerFast

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from common import label_index, option_labels  # noqa: E402

WORDS = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "choice", "question", "which", "team", "handles", "this", ":",
    "billing", "technical", "money", "refund", "invoice", "crash", "error", "bug",
]
QUESTION = {
    "type": "choice",
    "instructions": "which team handles this",
    "criteria": {"billing": "money refund invoice", "technical": "crash error bug"},
}


def build_checkpoint(root: Path) -> Path:
    torch.manual_seed(0)
    checkpoint = root / "tiny"
    vocab = {word: index for index, word in enumerate(WORDS)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
    )
    tokenizer.save_pretrained(checkpoint / "tokenizer")
    encoder_config = ModernBertConfig(
        vocab_size=len(WORDS), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=2, max_position_embeddings=256, pad_token_id=0, cls_token_id=2,
        sep_token_id=3, bos_token_id=2, eos_token_id=3, global_attn_every_n_layers=1,
        # A wide init lets two random layers mix context fast enough for a short CPU test.
        initializer_range=0.2,
    )
    encoder_config.save_pretrained(checkpoint / "encoder")
    model = DecisionModel(AutoModel.from_config(encoder_config, attn_implementation="sdpa"), head_layers=1, n_act=2)
    save_file({key: value.contiguous() for key, value in model.state_dict().items()}, str(checkpoint / "model.safetensors"))
    (checkpoint / "rl_agent_config.json").write_text(json.dumps({
        "encoder": "local-tiny",
        "head_layers": 1,
        "max_len": 64,
        "head_max_len": 32,
        "act_costs": {"escalate": 0.5},
        "temperature": [1.5, 1.0, 1.0],
        "temperature_by_options": {"choice:2": 2.0},
    }), encoding="utf-8")
    return checkpoint


def run(script: str, config: dict, directory: Path) -> None:
    path = directory / f"{script}.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    completed = subprocess.run([sys.executable, str(SRC / script), "--config", str(path)], capture_output=True, text=True)
    if completed.returncode != 0:
        raise AssertionError(completed.stdout + completed.stderr)


class LabelTests(unittest.TestCase):
    def test_labels_follow_question_type(self):
        self.assertEqual(option_labels(QUESTION), ["billing", "technical"])
        self.assertEqual(option_labels({"type": "noul"}), ["false", "true"])
        self.assertEqual(option_labels({"type": "score", "criteria": ["low", "mid", "high"]}), ["0", "1", "2"])

    def test_label_index_is_case_insensitive_and_strict(self):
        self.assertEqual(label_index(QUESTION, " Technical "), 1)
        self.assertEqual(label_index({"type": "noul"}, "TRUE"), 1)
        with self.assertRaisesRegex(ValueError, "not one of the decision labels"):
            label_index(QUESTION, "sales")


class TrainEvaluateTests(unittest.TestCase):
    def test_fine_tune_learns_labels_and_writes_a_loadable_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = build_checkpoint(root)
            rows = [
                {"input": "refund money", "output": "billing"},
                {"input": "invoice money", "output": "billing"},
                {"input": "crash error", "output": "technical"},
                {"input": "bug crash", "output": "technical"},
            ]
            (root / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            (root / "inputs.jsonl").write_text(
                "".join(json.dumps({"id": f"r{index}", "input": row["input"]}) + "\n" for index, row in enumerate(rows)),
                encoding="utf-8",
            )
            tuned = root / "tuned"
            run("train.py", {
                "model": str(checkpoint), "question": QUESTION, "train_path": str(root / "train.jsonl"),
                "output_dir": str(tuned), "metrics_path": str(root / "train-metrics.json"),
                "epochs": 60, "batch_size": 4, "learning_rate": 0.003, "device": "cpu", "seed": 3,
            }, root)
            metrics = json.loads((root / "train-metrics.json").read_text())
            self.assertTrue(metrics["ok"])
            self.assertEqual(metrics["steps"], 60)
            self.assertLess(metrics["final_loss"], metrics["loss_history"][0]["loss"])

            config = json.loads((tuned / "rl_agent_config.json").read_text())
            self.assertEqual(config["temperature"], [1.0, 1.0, 1.0])
            self.assertNotIn("temperature_by_options", config)
            self.assertEqual(config["tuned_tensor"]["labels"], ["billing", "technical"])
            self.assertTrue((tuned / "tokenizer" / "tokenizer.json").is_file())
            self.assertTrue((tuned / "encoder" / "config.json").is_file())
            self.assertEqual((tuned / "tokenizer" / "tokenizer.json").stat().st_mode & 0o077, 0)

            run("evaluate.py", {
                "model": str(tuned), "question": QUESTION, "inputs_path": str(root / "inputs.jsonl"),
                "output_path": str(root / "predictions.jsonl"), "metrics_path": str(root / "eval-metrics.json"),
                "device": "cpu",
            }, root)
            predictions = [json.loads(line) for line in (root / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual([row["id"] for row in predictions], ["r0", "r1", "r2", "r3"])
            self.assertEqual([row["prediction"] for row in predictions], [row["output"] for row in rows])
            for row in predictions:
                self.assertEqual(set(row["probabilities"]), {"billing", "technical"})
                self.assertAlmostEqual(sum(row["probabilities"].values()), 1.0, places=4)
                self.assertEqual(row["confidence"], max(row["probabilities"].values()))

    def test_evaluation_refuses_rows_that_carry_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = build_checkpoint(root)
            (root / "inputs.jsonl").write_text(json.dumps({"id": "r0", "input": "refund", "output": "billing"}) + "\n")
            with self.assertRaisesRegex(AssertionError, "only id and input"):
                run("evaluate.py", {
                    "model": str(checkpoint), "question": QUESTION, "inputs_path": str(root / "inputs.jsonl"),
                    "output_path": str(root / "p.jsonl"), "metrics_path": str(root / "m.json"), "device": "cpu",
                }, root)


if __name__ == "__main__":
    unittest.main()
