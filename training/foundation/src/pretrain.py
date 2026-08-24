from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.nn import functional as F

from common import load_config, require_cuda, write_json
from data import corpus_hash, decoded_byte_count, example_pairs, smoke_corpus, window_tokens
from model import FoundationGPT, bits_per_byte, model_config_from_depth, param_count, save_model


def main(argv: list[str] | None = None) -> None:
    config = load_config(argv)
    require_cuda("pretrain")
    device = torch.device("cuda")
    tokenizer = Tokenizer.from_file(str(Path(config["tokenizer_dir"]) / "tokenizer.json"))
    examples = example_pairs(config.get("examples") or [])
    corpus = smoke_corpus(examples, int(config.get("max_chars") or 20_000), extra=str(config.get("system_prompt") or ""))
    token_ids = tokenizer.encode(corpus).ids
    sequence_length = int(config.get("sequence_length") or 64)
    windows = window_tokens(token_ids, sequence_length)
    depth = int(config.get("depth") or 2)
    model_config = model_config_from_depth(depth, tokenizer.get_vocab_size(), sequence_length)
    model = FoundationGPT(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    batch_size = max(1, int(config.get("batch_size") or 2))
    steps = max(1, int(config.get("steps") or 2))
    last_loss = 0.0
    last_bpb = 0.0
    model.train()
    for step in range(steps):
        batch = windows[step % len(windows): step % len(windows) + batch_size]
        if len(batch) < batch_size:
            batch = (batch + windows)[:batch_size]
        tensor = torch.tensor(batch, dtype=torch.long, device=device)
        inputs = tensor[:, :-1]
        labels = tensor[:, 1:]
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())
        byte_count = decoded_byte_count(tokenizer, labels.detach().cpu().tolist())
        last_bpb = bits_per_byte(loss.detach() * labels.numel(), max(byte_count, 1))

    output_dir = Path(config["output_dir"])
    save_model(model, output_dir)
    write_json(output_dir / "metrics.json", {
        "ok": True,
        "steps": steps,
        "loss": last_loss,
        "bpb": last_bpb,
        "parameters": param_count(model),
        "depth": depth,
        "width": model_config["width"],
        "heads": model_config["heads"],
        "corpus_sha256": corpus_hash(corpus),
        "device": str(device),
    })


if __name__ == "__main__":
    main()
