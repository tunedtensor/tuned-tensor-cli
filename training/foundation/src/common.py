from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_config(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    return json.loads(Path(args.config).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination.parent.chmod(0o700)
    with destination.open("w", encoding="utf-8") as output:
        destination.chmod(0o600)
        output.write(json.dumps(value, indent=2) + "\n")


def require_cuda(action: str) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(f"Foundation {action} requires CUDA.")
