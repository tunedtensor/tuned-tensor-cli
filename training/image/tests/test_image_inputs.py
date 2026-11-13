import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train import check_images, load_components


class ImageInputTests(unittest.TestCase):
    def test_decode_and_content_hash_are_checked_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "sample.png"
            Image.new("RGB", (32, 32), "red").save(image)
            row = {"image": str(image), "sha256": hashlib.sha256(image.read_bytes()).hexdigest()}
            config = {"training": {"rows": [row]}, "validation": {"rows": []}}
            check_images(config)
            Image.new("RGB", (32, 32), "blue").save(image)
            with self.assertRaisesRegex(ValueError, "changed"):
                check_images(config)
            image.write_bytes(b"not an image")
            row["sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
            with self.assertRaises(OSError):
                check_images(config)

    def test_diffusers_swallowed_load_failure_is_fatal(self):
        class FailedPipeline:
            transformer = None
            def load_components(self, **kwargs):
                pass  # Diffusers logs component errors and may return without raising.
        with self.assertRaisesRegex(RuntimeError, "transformer"):
            load_components(FailedPipeline(), ["transformer"])


if __name__ == "__main__":
    unittest.main()
