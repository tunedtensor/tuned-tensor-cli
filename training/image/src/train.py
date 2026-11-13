"""Experimental local H3-Image captioned-image LoRA pipeline."""
import argparse
import gc
import hashlib
from importlib.metadata import version
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from diffusers import ModularPipeline
from diffusers.modular_pipelines.minimax_h3_image.encoders import get_qwen3vl_prompt_embeds
from safetensors.torch import load_file, save_file
from huggingface_hub import hf_hub_download

from h3_lora import attach_adapter, evaluate, loss_for_example, save_adapter, validate_schedule


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_images(config):
    """Verify the reviewed bytes and decode every image before any model download."""
    for split in ("training", "validation"):
        for row in config[split]["rows"]:
            if sha256(row["image"]) != row["sha256"]:
                raise ValueError(f"Image changed after preparation: {row['image']}")
            with Image.open(row["image"]) as image:
                image.verify()


def load_components(pipe, names, **kwargs):
    pipe.load_components(names=names, **kwargs)
    missing = [name for name in names if getattr(pipe, name, None) is None]
    if missing:
        raise RuntimeError(f"Failed to load H3 components: {missing}; see the loading error above")


def collect():
    gc.collect()
    torch.cuda.empty_cache()


def encode(config, output):
    """Load frozen components sequentially so the 32B encoder and 33B DiT never coexist."""
    spec, hp = config["spec"], config["spec"]["hyperparameters"]
    schedule_path = hf_hub_download(spec["base_model"], "pdd_config.json", revision=spec["base_model_revision"])
    schedule = json.loads(Path(schedule_path).read_text())
    sigmas = schedule["sigmas"]
    validate_schedule(sigmas, schedule["num_inference_steps"])
    pipe = ModularPipeline.from_pretrained(spec["base_model"], revision=spec["base_model_revision"])
    load_components(pipe, names=["tokenizer", "processor"], revision=spec["base_model_revision"])
    load_components(pipe, names=["text_encoder"], pretrained_model_name_or_path="MiniMaxAI/MiniMax-H3", subfolder="text_encoder", revision=spec["text_encoder_revision"], dtype=torch.bfloat16, device_map="cuda")
    pipe.text_encoder.requires_grad_(False).eval().to("cuda")
    cache = output / "cache"
    cache.mkdir()
    paths = {"training": [], "validation": []}
    with torch.no_grad():
        for split in paths:
            for index, row in enumerate(config[split]["rows"]):
                ids = pipe.tokenizer(row["caption"], add_special_tokens=False)["input_ids"][:4096]
                if not ids:
                    raise ValueError("Caption contains no tokens")
                embeds = get_qwen3vl_prompt_embeds(
                    pipe.text_encoder, pipe.processor, ids, {}, text_encoder_layer=50,
                    device=torch.device("cuda"), dtype=torch.bfloat16,
                )
                path = cache / f"{split}-{index}.safetensors"
                save_file({"prompt_embeds": embeds.cpu().contiguous()}, str(path))
                paths[split].append(path)
                print(f"Encoded caption {split} {index + 1}", flush=True)
    pipe.update_components(text_encoder=None)
    collect()
    load_components(pipe, names=["vae"], revision=spec["base_model_revision"], dtype=torch.float32, device_map="cuda")
    pipe.vae.requires_grad_(False).eval().to("cuda")
    pixel_mean = torch.tensor(pipe.pixel_mean, device="cuda").view(1, -1, 1, 1)
    pixel_std = torch.tensor(pipe.pixel_std, device="cuda").view(1, -1, 1, 1)
    mean = torch.tensor(pipe.vae.config.latents_mean, device="cuda").view(1, -1, 1, 1)
    std = torch.tensor(pipe.vae.config.latents_std, device="cuda").view(1, -1, 1, 1)
    generator = torch.Generator(device="cuda").manual_seed(hp["seed"])
    with torch.no_grad():
        for split in paths:
            for row, path in zip(config[split]["rows"], paths[split]):
                if sha256(row["image"]) != row["sha256"]:
                    raise ValueError(f"Image changed during encoding: {row['image']}")
                with Image.open(row["image"]) as image:
                    image = ImageOps.fit(ImageOps.exif_transpose(image).convert("RGB"), (hp["width"], hp["height"]), method=Image.Resampling.LANCZOS)
                    pixels = torch.from_numpy(np.array(image)).to("cuda").permute(2, 0, 1)[None].float() / 255.0
                posterior = pipe.vae.encode((pixels - pixel_mean) / pixel_std, return_dict=False)[0]
                latents = posterior.sample(generator=generator).to(torch.float16).float()
                latents = (latents - mean) / std
                tensors = load_file(str(path))
                tensors["latents"] = latents.cpu().contiguous()
                save_file(tensors, str(path))
    pipe.update_components(vae=None)
    collect()
    load_components(pipe, names=["transformer"], revision=spec["base_model_revision"], dtype=torch.bfloat16, device_map="cuda")
    model = pipe.transformer.to("cuda")
    validate_schedule(sigmas, model.config.num_pdd_heads)
    return model, sigmas, paths


def train(model, paths, sigmas, hp):
    """One image per microbatch; each optimizer step averages exactly accumulation microbatches."""
    model.train()
    if hp["gradient_checkpointing"]:
        model.enable_gradient_checkpointing()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=hp["learning_rate"], weight_decay=0.0)
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(hp["seed"])
    rng = random.Random(hp["seed"])
    order = list(paths)
    rng.shuffle(order)
    cursor = 0
    for step in range(hp["steps"]):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(hp["gradient_accumulation_steps"]):
            if cursor == len(order):
                rng.shuffle(order)
                cursor = 0
            example = load_file(str(order[cursor]))
            cursor += 1
            head = rng.randrange(model.config.num_pdd_heads)
            loss = loss_for_example(model, example, sigmas, head, generator)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss at step {step + 1}")
            (loss / hp["gradient_accumulation_steps"]).backward()
            total += loss.item() / hp["gradient_accumulation_steps"]
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        print(json.dumps({"step": step + 1, "total_steps": hp["steps"], "loss": total}), flush=True)
    optimizer.zero_grad(set_to_none=True)


def run(config_path):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    output, spec = config_path.parent, config["spec"]
    hp = spec["hyperparameters"]
    check_images(config)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("H3-Image training requires an NVIDIA CUDA GPU with BF16 support")
    torch.manual_seed(hp["seed"])
    model, sigmas, paths = encode(config, output)
    attach_adapter(model, hp["lora_rank"], hp["lora_alpha"])
    eval_seed = hp["seed"] + 1
    baseline = evaluate(model, paths["validation"], sigmas, eval_seed)
    train(model, paths["training"], sigmas, hp)
    candidate = evaluate(model, paths["validation"], sigmas, eval_seed)
    adapter = output / "adapter"
    save_adapter(model, adapter)
    # Verify the exported tensors, not just the in-memory training result. Reset the
    # adapter before loading so missing or wrong weights cannot hide behind live ones.
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.data.zero_()
    from peft import set_peft_model_state_dict
    result = set_peft_model_state_dict(model, load_file(str(adapter / "adapter_model.safetensors")))
    if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
        raise ValueError("Exported adapter failed to reload")
    reloaded = evaluate(model, paths["validation"], sigmas, eval_seed)
    if not math.isclose(candidate, reloaded, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError("Exported adapter did not reproduce held-out loss")
    provenance = {"base_model": spec["base_model"], "base_model_revision": spec["base_model_revision"],
                  "text_encoder": "MiniMaxAI/MiniMax-H3", "text_encoder_revision": spec["text_encoder_revision"],
                  "sample_sigmas": sigmas, "spec_sha256": config["source"]["sha256"]}
    (adapter / "tt_image_adapter.json").write_text(json.dumps(provenance, indent=2) + "\n")
    report = {"status": "succeeded", "experimental": True, "optimizer_steps": hp["steps"],
              "baseline_loss": baseline, "candidate_loss": candidate, "loss_delta": candidate - baseline,
              "reloaded_loss": reloaded, "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
              "metric": "held_out_data_ward_velocity_mse", "quality_validated": False,
              "validation_images": len(paths["validation"]), "evaluation_seed": eval_seed,
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "diffusers": version("diffusers"), "peft": version("peft"),
              "runtime_lock_sha256": sha256(Path(__file__).resolve().parents[1] / "uv.lock"), **provenance}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    run(parser.parse_args().config)
