from __future__ import annotations

import argparse
import json
from pathlib import Path
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
    with destination.open("w", encoding="utf-8") as output:
        make_private_file(destination)
        output.write(json.dumps(value, indent=2) + "\n")


def require_cuda(action: str) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(f"Foundation {action} requires CUDA.")
