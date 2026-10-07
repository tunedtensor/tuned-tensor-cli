"""Uncertified prefetch rejects checkpoints the safe loader cannot train."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prefetch import verify_snapshot


class UncertifiedPrefetchTests(unittest.TestCase):
    def test_indexed_shards_are_contained_and_inventoried(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshots" / "revision"
            snapshot.mkdir(parents=True)
            (snapshot / "nested").mkdir()
            (snapshot / "config.json").write_text('{"model_type":"granite"}')
            (snapshot / "tokenizer_config.json").write_text("{}")
            (snapshot / "tokenizer.json").write_text("{}")
            (snapshot / "model.safetensors").write_bytes(b"weights")
            outside = root / "snapshots" / "outside.safetensors"
            outside.write_bytes(b"outside")
            (snapshot / "linked-directory").symlink_to("../../blobs", target_is_directory=True)
            index = snapshot / "model.safetensors.index.json"
            for name in ("../outside.safetensors", str(outside),
                         "nested/../../outside.safetensors", "C:\\outside.safetensors",
                         "..\\outside.safetensors", "nested/../model.safetensors",
                         "missing.safetensors", "nested/empty.safetensors", "", None, []):
                with self.subTest(name=name):
                    (snapshot / "nested" / "empty.safetensors").write_bytes(b"")
                    index.write_text(json.dumps({"weight_map": {"tensor": name}}))
                    with self.assertRaisesRegex(ValueError, "indexed weight shard"):
                        verify_snapshot(snapshot, "example/model")
                    (snapshot / "nested" / "empty.safetensors").unlink()
            blob = root / "blobs" / hashlib.sha256(b"shard").hexdigest()
            blob.parent.mkdir()
            blob.write_bytes(b"shard")
            nested = snapshot / "nested" / "shard.safetensors"
            nested.symlink_to("../../../blobs/" + blob.name)
            index.write_text(json.dumps({"weight_map": {"tensor": "linked-directory/" + blob.name}}))
            with self.assertRaisesRegex(ValueError, "indexed weight shard"):
                verify_snapshot(snapshot, "example/model")
            index.write_text(json.dumps({"weight_map": {"tensor": "nested/shard.safetensors"}}))
            files, verified = verify_snapshot(snapshot, "example/model")
            self.assertIn(nested, files)
            self.assertEqual(verified, 1)
            blob.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_snapshot(snapshot, "example/model")

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
