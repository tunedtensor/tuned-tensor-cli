from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from safetensors.torch import load_file, save_file

from common import ensure_private_directory, make_private_file, write_json


def derived_width(depth: int) -> int:
    return 32 * int(depth)


def derived_heads(depth: int) -> int:
    width = derived_width(depth)
    for heads in (8, 4, 2, 1):
        if width % heads == 0:
            return heads
    return 1


def model_config_from_depth(depth: int, vocab_size: int, sequence_length: int) -> dict[str, int]:
    return {
        "depth": int(depth),
        "width": derived_width(depth),
        "heads": derived_heads(depth),
        "vocab_size": int(vocab_size),
        "sequence_length": int(sequence_length),
    }


class CausalSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, width = x.shape
        qkv = self.qkv(x).view(batch, steps, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        context = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=True,
        )
        context = context.transpose(1, 2).contiguous().view(batch, steps, width)
        return self.proj(context)


class TransformerBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.attn = CausalSelfAttention(width, heads)
        self.ln2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class FoundationGPT(nn.Module):
    def __init__(self, config: dict[str, int]) -> None:
        super().__init__()
        self.config = config
        width = config["width"]
        self.tok_emb = nn.Embedding(config["vocab_size"], width)
        self.pos_emb = nn.Embedding(config["sequence_length"], width)
        self.blocks = nn.ModuleList(
            [TransformerBlock(width, config["heads"]) for _ in range(config["depth"])],
        )
        self.ln_f = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, config["vocab_size"], bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _batch, steps = input_ids.shape
        if steps > self.config["sequence_length"]:
            raise ValueError("sequence exceeds model sequence_length")
        positions = torch.arange(steps, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))


@torch.no_grad()
def generate(
    model: FoundationGPT,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    stop_token_id: int | None = None,
) -> torch.Tensor:
    model.eval()
    tokens = input_ids
    limit = model.config["sequence_length"]
    finished = torch.zeros(tokens.size(0), dtype=torch.bool, device=tokens.device)
    for _ in range(max_new_tokens):
        logits = model(tokens[:, -limit:])
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if stop_token_id is not None:
            next_token = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_token, stop_token_id),
                next_token,
            )
        tokens = torch.cat([tokens, next_token], dim=1)
        if stop_token_id is not None:
            finished |= next_token.squeeze(1).eq(stop_token_id)
            if bool(finished.all()):
                break
    return tokens


def save_model(model: FoundationGPT, directory: str | Path) -> None:
    destination = ensure_private_directory(directory)
    tensors = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    model_path = destination / "model.safetensors"
    save_file(tensors, str(model_path))
    make_private_file(model_path)
    write_json(destination / "config.json", model.config)


def load_model(directory: str | Path, map_location: str | torch.device = "cpu") -> FoundationGPT:
    destination = Path(directory)
    config: dict[str, Any] = json.loads((destination / "config.json").read_text(encoding="utf-8"))
    model = FoundationGPT({key: int(value) for key, value in config.items()})
    state = load_file(str(destination / "model.safetensors"))
    model.load_state_dict(state)
    return model.to(map_location)


def param_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def bits_per_byte(nll_sum: torch.Tensor, byte_count: int) -> float:
    if byte_count <= 0:
        return 0.0
    return float(nll_sum.item() / math.log(2) / byte_count)
