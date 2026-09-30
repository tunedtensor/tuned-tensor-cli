"""Uncertified prefetch rejects checkpoints the safe loader cannot train."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prefetch import verify_snapshot


class UncertifiedPrefetchTests(unittest.TestCase):
    def test_pickle_only_is_rejected_but_safetensors_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(json.dumps({"model_type": "granite"}))
            (root / "tokenizer_config.json").write_text("{}")
            (root / "tokenizer.json").write_text("{}")
            (root / "pytorch_model.bin").write_bytes(b"weights")
            with self.assertRaisesRegex(ValueError, "safetensors"):
                verify_snapshot(root, "example/model")
            (root / "model.safetensors").write_bytes(b"weights")
            files, _ = verify_snapshot(root, "example/model")
            self.assertIn(root / "model.safetensors", files)
