import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from diffusers import MiniMaxH3ImageTransformer2DModel
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from h3_lora import attach_adapter, evaluate, load_adapter, loss_for_example, save_adapter, validate_schedule
from train import train

DEVICE = os.environ.get("TT_IMAGE_TEST_DEVICE", "cpu")
SIGMAS = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]


def tiny_model():
    torch.manual_seed(17)
    model = MiniMaxH3ImageTransformer2DModel(
        num_attention_heads=2, attention_head_dim=16, hidden_size=32,
        num_layers=1, num_refiner_layers=1, ffn_dim=48, in_channels=4,
        text_dim=16, freq_dim=16, time_embed_hidden_dim=32, time_embed_dim=16,
        rope_freq_dim=2, num_pdd_heads=6,
    ).to(DEVICE)
    if DEVICE == "cuda":
        model.to(dtype=torch.bfloat16)
        for name in model._keep_in_fp32_modules:
            getattr(model, name).to(dtype=torch.float32)
    return model


def example():
    torch.manual_seed(19)
    return {"latents": torch.randn(1, 4, 4, 4), "prompt_embeds": torch.randn(1, 3, 16)}


class H3LoRATests(unittest.TestCase):
    def test_actual_transformer_train_export_reload(self):
        """Real H3 architecture and PEFT; only dimensions/weights are synthetic."""
        model = tiny_model()
        initial = copy.deepcopy(model.state_dict())
        attach_adapter(model, 2, 2)
        frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.safetensors"
            save_file(example(), str(path))
            baseline = evaluate(model, [path], SIGMAS, 123)
            train(model, [path], SIGMAS, {
                "steps": 3, "gradient_checkpointing": True, "gradient_accumulation_steps": 2,
                "seed": 41, "learning_rate": 0.01,
            })
            candidate = evaluate(model, [path], SIGMAS, 123)
            self.assertNotEqual(baseline, candidate)
            self.assertTrue(any(not torch.equal(before[n], p) for n, p in model.named_parameters() if p.requires_grad))
            for name, param in model.named_parameters():
                if name in frozen:
                    self.assertTrue(torch.equal(frozen[name], param), name)
            adapter = Path(directory) / "adapter"
            save_adapter(model, adapter)
            restored = tiny_model()
            restored.load_state_dict(initial)
            load_adapter(restored, adapter)
            self.assertAlmostEqual(candidate, evaluate(restored, [path], SIGMAS, 123), places=6)
            self.assertLess((adapter / "adapter_model.safetensors").stat().st_size, 100_000)

    def test_each_head_receives_the_matching_time_and_data_ward_target(self):
        from types import SimpleNamespace
        from diffusers.modular_pipelines.minimax_h3_image.before_denoise import patchify_video_latents
        data = example()
        for head in range(6):
            gen = torch.Generator(device=DEVICE).manual_seed(123)
            clean = data["latents"].to(DEVICE)
            noise = torch.randn(clean.shape, device=DEVICE, generator=gen)
            target = patchify_video_latents((clean - noise).unsqueeze(2), (1, 2, 2)).unsqueeze(0)
            class Oracle(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.anchor = torch.nn.Parameter(torch.zeros(1, device=DEVICE))
                    self.config = SimpleNamespace(num_pdd_heads=6, patch_size=(1, 2, 2))
                def forward(inner, **kwargs):
                    self.assertEqual(kwargs["pdd_step_index"], head)
                    self.assertAlmostEqual(kwargs["timestep"].item(), 1 - SIGMAS[head], places=6)
                    self.assertEqual(kwargs["image_indices"].numel(), 4)
                    self.assertEqual(kwargs["text_indices"].numel(), 3)
                    self.assertEqual(kwargs["timestep_indices"].sum().item(), 0)
                    return SimpleNamespace(sample=target)
            loss = loss_for_example(Oracle(), data, SIGMAS, head, torch.Generator(device=DEVICE).manual_seed(123))
            self.assertEqual(loss.item(), 0.0)

    def test_schedule_rejects_wrong_heads_or_nonfinite_values(self):
        validate_schedule(SIGMAS, 6)
        for bad in ([1, 0], [1, float("nan"), 0], [1, 0.5, 0.6, 0]):
            with self.assertRaises(ValueError):
                validate_schedule(bad, 6)

    @unittest.skipUnless(DEVICE == "cuda", "Full runner acceptance uses an actual CUDA device")
    def test_runner_writes_verified_export_and_report_on_cuda(self):
        import hashlib
        import json
        from unittest.mock import patch
        from PIL import Image
        from train import run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "input.png"
            Image.new("RGB", (32, 32), "red").save(image)
            row = {"image": str(image), "sha256": hashlib.sha256(image.read_bytes()).hexdigest(), "caption": "red"}
            data = root / "encoded.safetensors"
            save_file(example(), str(data))
            config = {
                "spec": {"base_model": "MiniMaxAI/MiniMax-H3-Image", "base_model_revision": "a" * 40,
                         "text_encoder_revision": "b" * 40, "hyperparameters": {
                             "steps": 1, "seed": 42, "lora_rank": 2, "lora_alpha": 2,
                             "gradient_accumulation_steps": 1, "gradient_checkpointing": True,
                             "learning_rate": 0.01,
                         }},
                "source": {"sha256": "c" * 64},
                "training": {"rows": [row]}, "validation": {"rows": [row]},
            }
            path = root / "config.json"
            path.write_text(json.dumps(config))
            # Replace only expensive pretrained-component encoding. The CUDA H3
            # trainer, PEFT, evaluation, export/reload and report execute normally.
            with patch("train.encode", return_value=(tiny_model(), SIGMAS, {"training": [data], "validation": [data]})):
                run(path)
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["optimizer_steps"], 1)
            self.assertFalse(report["quality_validated"])
            self.assertAlmostEqual(report["candidate_loss"], report["reloaded_loss"], places=6)
            self.assertEqual(report["adapter_sha256"], hashlib.sha256((root / "adapter/adapter_model.safetensors").read_bytes()).hexdigest())

    def test_incomplete_export_is_rejected(self):
        from safetensors.torch import load_file
        model = attach_adapter(tiny_model(), 2, 2)
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "adapter"
            save_adapter(model, adapter)
            path = adapter / "adapter_model.safetensors"
            weights = load_file(str(path))
            weights.pop(next(iter(weights)))
            save_file(weights, str(path))
            with self.assertRaises(ValueError):
                load_adapter(tiny_model(), adapter)


if __name__ == "__main__":
    unittest.main()
