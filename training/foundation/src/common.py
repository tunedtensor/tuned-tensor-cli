from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def ensure_private_directory(path: str | Path) -> Path:
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination.chmod(0o700)
    return destination


def make_private_file(path: str | Path) -> Path:
    destination = Path(path)
    destination.chmod(0o600)
    return destination


def load_config(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    return json.loads(Path(args.config).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    ensure_private_directory(destination.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            os.fchmod(output.fileno(), 0o600)
            output.write(json.dumps(value, indent=2) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        make_private_file(destination)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: str | Path, value: Any) -> None:
    destination = Path(path)
    ensure_private_directory(destination.parent)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        payload = (json.dumps(value, separators=(",", ":")) + "\n").encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    make_private_file(destination)


def require_cuda(action: str) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(f"Foundation {action} requires CUDA.")
