from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.nn import functional as F

from common import load_config, require_cuda, write_json
from data import IGNORE_INDEX, encode_sft_example, example_pairs
from model import load_model, param_count, save_model


def sft_tensors(
    tokenizer: Tokenizer,
    system_prompt: str,
    examples: list[dict[str, str]],
    sequence_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs: list[list[int]] = []
    labels: list[list[int]] = []
    for example in examples:
        input_ids, label_ids = encode_sft_example(
            tokenizer,
            system_prompt,
            example["input"],
            example["output"],
            sequence_length,
        )
        inputs.append(input_ids)
        labels.append(label_ids)
    return (
        torch.tensor(inputs, dtype=torch.long, device=device),
        torch.tensor(labels, dtype=torch.long, device=device),
    )


def main(argv: list[str] | None = None) -> None:
    config = load_config(argv)
    require_cuda("finetune")
    device = torch.device("cuda")
    tokenizer = Tokenizer.from_file(str(Path(config["tokenizer_dir"]) / "tokenizer.json"))
    model = load_model(config["model_dir"], map_location=device)
    examples = example_pairs(config.get("examples") or [])
    if len(examples) < 1:
        raise SystemExit("Foundation finetune needs at least one chat example.")
    sequence_length = int(config.get("sequence_length") or model.config["sequence_length"])
    inputs, labels = sft_tensors(
        tokenizer,
        str(config.get("system_prompt") or ""),
        examples,
        sequence_length,
        device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    steps = max(1, int(config.get("steps") or 2))
    batch_size = max(1, int(config.get("batch_size") or 2))
    last_loss = 0.0
    model.train()
    for step in range(steps):
        start = (step * batch_size) % inputs.size(0)
        end = start + batch_size
        if end <= inputs.size(0):
            batch_inputs = inputs[start:end]
            batch_labels = labels[start:end]
        else:
            indices = [(start + offset) % inputs.size(0) for offset in range(batch_size)]
            batch_inputs = inputs[indices]
            batch_labels = labels[indices]
        logits = model(batch_inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch_labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())

    output_dir = Path(config["output_dir"])
    save_model(model, output_dir)
    write_json(output_dir / "metrics.json", {
        "ok": True,
        "steps": steps,
        "loss": last_loss,
        "examples": inputs.size(0),
        "parameters": param_count(model),
        "ignore_index": IGNORE_INDEX,
        "device": str(device),
    })


if __name__ == "__main__":
    main()
