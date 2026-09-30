"""Shared helpers for the typed decision-model engine (Laya-format checkpoints).

Node owns the behavior spec, splits, scoring and reports. This package only
loads a decision checkpoint, predicts labels for opaque IDs, and fine-tunes the
checkpoint's encoder and decision head on labelled rows.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

CHECKPOINT_FILES = ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")
QUESTION_ID = "decision"


def load_config(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    return json.loads(Path(args.config).read_text(encoding="utf-8"))


def ensure_private_directory(path: str | Path) -> Path:
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination.chmod(0o700)
    return destination


def write_json(path: str | Path, value: Any) -> None:
    write_text(path, json.dumps(value, indent=2) + "\n")


def write_jsonl(path: str | Path, rows: list[Any]) -> None:
    write_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def write_text(path: str | Path, content: str) -> None:
    destination = Path(path)
    ensure_private_directory(destination.parent)
    descriptor, temporary_name = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            os.fchmod(output.fileno(), 0o600)
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number} must be a JSON object")
        rows.append(row)
    return rows


def pick_device(requested: str | None) -> torch.device:
    if requested and requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Decision device cuda was requested, but PyTorch cannot see a CUDA GPU.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Decision device mps was requested, but PyTorch cannot use Apple MPS.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def option_labels(question: dict[str, Any]) -> list[str]:
    """Answer labels in the model's option order; these are the strings Node scores."""
    kind = question["type"]
    criteria = question.get("criteria")
    if kind == "choice":
        return [str(label) for label in (criteria.keys() if isinstance(criteria, dict) else criteria)]
    if kind == "noul":
        return ["false", "true"]
    if kind == "score":
        return [str(index) for index in range(len(criteria))]
    raise ValueError(f"Unsupported decision type: {kind!r}")


def label_index(question: dict[str, Any], output: str) -> int:
    labels = option_labels(question)
    normalized = output.strip().lower()
    for index, label in enumerate(labels):
        if label.strip().lower() == normalized:
            return index
    raise ValueError(f"Expected output {output!r} is not one of the decision labels {labels}")


def resolve_checkpoint_dir(model: str, revision: str | None) -> Path:
    """Return a local checkpoint directory, downloading only the files a load needs."""
    local = Path(model)
    if local.is_dir():
        return local
    if local.is_absolute() or model.startswith(("./", "../")):
        raise FileNotFoundError(f"Decision checkpoint directory not found: {model}")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, revision=revision, allow_patterns=list(CHECKPOINT_FILES)))


def load_agent(model: str, revision: str | None, device: torch.device):
    from laya import Agent

    checkpoint = resolve_checkpoint_dir(model, revision)
    # Load the resolved directory so Hub and local checkpoints take the same path.
    agent = Agent(str(checkpoint), device=str(device))
    return agent, checkpoint
