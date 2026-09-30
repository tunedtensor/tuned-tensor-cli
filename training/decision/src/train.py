"""Fine-tune a Laya-format decision checkpoint on labelled rows for one typed question.

The encoder and decision head are trained with cross-entropy over the option
markers, a strictly proper scoring rule, so the tuned probabilities remain a
calibration target rather than only an argmax. Options are shuffled per example
so the head learns option meaning instead of option position. The output
directory is a complete checkpoint that `laya.load(<dir>)` accepts.
"""
from __future__ import annotations

import json
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from laya.common import QTYPES, build_sequence, collate_items
from safetensors.torch import save_file

from common import (
    ensure_private_directory,
    label_index,
    load_agent,
    load_config,
    option_labels,
    pick_device,
    read_jsonl,
    write_json,
)


def internal_question(question: dict[str, Any]) -> dict[str, Any]:
    """Laya's internal question form; mirrors `Agent._to_internal` for the fields TT emits."""
    criteria = question.get("criteria")
    if question["type"] == "choice" and isinstance(criteria, list):
        criteria = {label: None for label in criteria}
    return {"t": question["type"], "ins": question["instructions"], "crit": criteria}


def encode_example(tok, cfg: dict[str, Any], question: dict[str, Any], text: str, label: int, order: list[int]) -> dict[str, Any]:
    ids, markers = build_sequence(
        tok,
        text,
        question,
        cfg.get("max_len", 512),
        cfg.get("head_max_len", 192),
        option_order=order,
    )
    if len(markers) != len(order):
        raise ValueError("A decision option marker was truncated; shorten the instructions or option descriptions.")
    # Slot s shows option order[s], so the target is the slot holding the labelled option.
    return {"ids": ids, "markers": markers, "qtype": QTYPES[question["t"]], "label": order.index(label)}


def autocast(device: torch.device):
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


def save_checkpoint(model, source: Path, output: Path, cfg: dict[str, Any], provenance: dict[str, Any]) -> None:
    ensure_private_directory(output)
    for name in ("tokenizer", "encoder"):
        if (source / name).is_dir():
            shutil.copytree(source / name, output / name, dirs_exist_ok=True)
    state = {key: value.detach().to("cpu").contiguous() for key, value in model.state_dict().items()}
    # The base checkpoint's fitted temperatures describe its logits, not the tuned ones.
    state["temperature"] = torch.ones_like(state["temperature"])
    save_file(state, str(output / "model.safetensors"))
    tuned = {key: value for key, value in cfg.items() if key != "temperature_by_options"}
    tuned["temperature"] = [1.0, 1.0, 1.0]
    tuned["tuned_tensor"] = provenance
    (output / "rl_agent_config.json").write_text(json.dumps(tuned, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    config = load_config()
    seed = int(config.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    device = pick_device(config.get("device"))
    question = config["question"]
    labels = option_labels(question)
    rows = read_jsonl(config["train_path"])
    if not rows:
        raise ValueError("Decision training needs at least one labelled row.")
    targets = [label_index(question, str(row["output"])) for row in rows]

    agent, checkpoint = load_agent(config["model"], config.get("revision"), device)
    model = agent.model
    if config.get("freeze_encoder", False):
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(config.get("learning_rate", 2e-5)), weight_decay=0.01)
    epochs = int(config.get("epochs", 3))
    batch_size = int(config.get("batch_size", 8))
    shuffle_options = bool(config.get("shuffle_options", True))
    steps_per_epoch = (len(rows) + batch_size - 1) // batch_size
    total_steps = epochs * steps_per_epoch
    warmup = max(1, total_steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / warmup) * max(0.0, (total_steps - step) / max(1, total_steps - warmup + 1)),
    )
    q = internal_question(question)
    pad_id = agent.tok.pad_token_id if agent.tok.pad_token_id is not None else 0

    model.train()
    history: list[dict[str, Any]] = []
    began = time.perf_counter()
    step = 0
    for epoch in range(epochs):
        order_of_rows = list(range(len(rows)))
        random.shuffle(order_of_rows)
        epoch_loss = 0.0
        for start in range(0, len(rows), batch_size):
            items = []
            for index in order_of_rows[start:start + batch_size]:
                order = list(range(len(labels)))
                if shuffle_options:
                    random.shuffle(order)
                items.append([encode_example(agent.tok, agent.cfg, q, str(rows[index]["input"]), targets[index], order)])
            batch = collate_items(items, pad_id)
            with autocast(device):
                logits, _ = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device),
                    batch["marker_mask"].to(device),
                    batch["qtype"].to(device),
                )
            loss = F.cross_entropy(logits.float(), batch["label"].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            epoch_loss += float(loss.item()) * len(items)
            print(json.dumps({"event": "step", "epoch": epoch + 1, "step": step, "total_steps": total_steps, "loss": round(float(loss.item()), 6)}), flush=True)
        history.append({"epoch": epoch + 1, "loss": round(epoch_loss / len(rows), 6)})

    model.eval()
    seconds = time.perf_counter() - began
    provenance = {
        "base_model": config["model"],
        "base_revision": config.get("revision"),
        "question_type": question["type"],
        "labels": labels,
        "train_rows": len(rows),
        "epochs": epochs,
        "learning_rate": float(config.get("learning_rate", 2e-5)),
        "freeze_encoder": bool(config.get("freeze_encoder", False)),
        "seed": seed,
    }
    output = Path(config["output_dir"])
    save_checkpoint(model, checkpoint, output, agent.cfg, provenance)
    write_json(config["metrics_path"], {
        "ok": True,
        "device": str(device),
        "train_rows": len(rows),
        "epochs": epochs,
        "steps": step,
        "train_seconds": round(seconds, 3),
        "loss_history": history,
        "final_loss": history[-1]["loss"],
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
    })


if __name__ == "__main__":
    main()
