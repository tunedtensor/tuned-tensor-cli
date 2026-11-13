"""H3-Image LoRA objective and portable PEFT adapter I/O.

Uses the pinned Diffusers implementation for packing and positional encoding.
H3 predicts data-ward velocity (clean - noise), at t = 1 - sigma.
"""
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model_state_dict, inject_adapter_in_model, set_peft_model_state_dict
from safetensors.torch import load_file, save_file
from diffusers.modular_pipelines.minimax_h3_image.before_denoise import (
    build_image_packed_sequence,
    patchify_video_latents,
)

TARGETS = r"transformer_blocks\.\d+\.attn\.(to_q|to_k|to_v|to_out\.0)"


def attach_adapter(model, rank, alpha):
    model.requires_grad_(False)
    config = LoraConfig(r=rank, lora_alpha=alpha, target_modules=TARGETS, bias="none")
    inject_adapter_in_model(config, model)
    # Keep the frozen checkpoint's mixed precision; only the trainable adapter is FP32.
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    if not any(p.requires_grad for p in model.parameters()):
        raise ValueError("No H3 attention modules matched the adapter targets")
    return model


def save_adapter(model, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    model.peft_config["default"].save_pretrained(directory)
    weights = {key: value.detach().cpu().contiguous() for key, value in get_peft_model_state_dict(model).items()}
    if not weights or not all(torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("Adapter weights are empty or non-finite")
    save_file(weights, directory / "adapter_model.safetensors", metadata={"format": "pt"})


def load_adapter(model, directory):
    """Load TT's standard PEFT adapter into a fresh H3-Image transformer."""
    config = LoraConfig.from_pretrained(directory)
    inject_adapter_in_model(config, model)
    # Injection follows the base dtype; promote before copying saved FP32 values.
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.data = param.data.float()
    weights = load_file(str(Path(directory) / "adapter_model.safetensors"))
    result = set_peft_model_state_dict(model, weights)
    if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
        raise ValueError(f"Adapter does not match this transformer: {result}")
    return model


def validate_schedule(sigmas, heads):
    if len(sigmas) != heads + 1 or sigmas[0] != 1.0 or sigmas[-1] != 0.0:
        raise ValueError("H3 sampling schedule must match its per-step output heads")
    if not all(0 <= b < a <= 1 for a, b in zip(sigmas, sigmas[1:])):
        raise ValueError("H3 sampling sigmas must strictly decrease from 1 to 0")


def loss_for_example(model, example, sigmas, head, generator):
    """One caption/image, one discrete PDD head; no chat labels or CFG branch."""
    if not 0 <= head < model.config.num_pdd_heads:
        raise ValueError("Invalid H3 output head")
    device = next(model.parameters()).device
    clean = example["latents"].to(device=device, dtype=torch.float32)
    embeds = example["prompt_embeds"].to(device=device)
    noise = torch.randn(clean.shape, generator=generator, device=device, dtype=torch.float32)
    # Match the float32 round trip in the upstream sampling schedule.
    timestep = torch.tensor([1.0 - sigmas[head]], device=device, dtype=torch.float32)
    sigma = 1.0 - timestep
    noisy = (1.0 - sigma) * clean + sigma * noise
    patch = tuple(model.config.patch_size)
    pack = lambda value: patchify_video_latents(value.unsqueeze(2), patch).unsqueeze(0)
    positions, tags, image_indices, text_indices, _ = build_image_packed_sequence(
        torch.ones(embeds.shape[1], dtype=torch.long), [], clean.shape[-2], clean.shape[-1], patch, 0,
    )
    prediction = model(
        hidden_states=pack(noisy), encoder_hidden_states=embeds, timestep=timestep,
        timestep_indices=torch.zeros(tags.shape, device=device, dtype=torch.long),
        token_tags=tags.to(device), position_ids=positions.to(device),
        image_indices=image_indices.to(device), text_indices=text_indices.to(device), pdd_step_index=head,
    ).sample
    return torch.nn.functional.mse_loss(prediction.float(), pack(clean - noise))


@torch.no_grad()
def evaluate(model, paths, sigmas, seed):
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    total = 0.0
    for path in paths:
        example = load_file(str(path))
        for head in range(model.config.num_pdd_heads):
            loss = loss_for_example(model, example, sigmas, head, generator)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite held-out loss")
            total += loss.item()
    return total / (len(paths) * model.config.num_pdd_heads)
