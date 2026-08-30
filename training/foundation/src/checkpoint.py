from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from common import ensure_private_directory, write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ResumeState:
    step: int
    samples_consumed: int
    tokens_seen: int
    last_loss: float
    path: Path


class CheckpointManager:
    def __init__(
        self,
        output_dir: str | Path,
        config_fingerprint: str,
        *,
        keep: int = 3,
        backup_dir: str | Path | None = None,
    ) -> None:
        self.root = ensure_private_directory(Path(output_dir) / "checkpoints").resolve()
        self.config_fingerprint = config_fingerprint
        self.keep = max(1, int(keep))
        self.backup_dir = Path(backup_dir).expanduser().resolve() if backup_dir else None
        if self.backup_dir is not None and (
            self.backup_dir == self.root or self.root in self.backup_dir.parents
        ):
            raise ValueError("checkpoint_backup_dir must be outside the primary checkpoints directory.")

    @staticmethod
    def _name(step: int) -> str:
        return f"step-{step:09d}"

    def _candidates(self, root: Path | None = None) -> list[Path]:
        candidate_root = root or self.root
        return sorted(
            (path for path in candidate_root.glob("step-*") if path.is_dir()),
            key=lambda path: path.name,
            reverse=True,
        )

    def _resume_candidates(self) -> list[Path]:
        ranked = [(path, 1) for path in self._candidates()]
        if self.backup_dir is not None and self.backup_dir.is_dir():
            ranked.extend((path, 0) for path in self._candidates(self.backup_dir))
        return [
            path
            for path, _priority in sorted(
                ranked,
                key=lambda candidate: (candidate[0].name, candidate[1]),
                reverse=True,
            )
        ]

    def latest(self) -> Path | None:
        mismatched = False
        corrupt = False
        for path in self._resume_candidates():
            try:
                metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                corrupt = True
                continue
            if not isinstance(metadata, dict) or metadata.get("complete") is not True:
                corrupt = True
                continue
            if metadata.get("config_fingerprint") != self.config_fingerprint:
                mismatched = True
                continue
            model_path = path / "model.safetensors"
            state_path = path / "training-state.pt"
            if not model_path.is_file() or not state_path.is_file():
                corrupt = True
                continue
            files = metadata.get("files")
            if not isinstance(files, dict):
                corrupt = True
                continue
            valid = True
            for name, candidate in [
                ("model.safetensors", model_path),
                ("training-state.pt", state_path),
            ]:
                expected = files.get(name)
                if (
                    not isinstance(expected, dict)
                    or candidate.stat().st_size != expected.get("size")
                    or _sha256(candidate) != expected.get("sha256")
                ):
                    valid = False
                    break
            if valid:
                return path
            corrupt = True
        if mismatched:
            raise ValueError("Existing foundation checkpoints do not match the current training configuration.")
        if corrupt:
            raise ValueError("No intact foundation checkpoint remains for the current training configuration.")
        return None

    def load(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        device: torch.device,
    ) -> ResumeState | None:
        path = self.latest()
        if path is None:
            return None
        model.load_state_dict(load_file(str(path / "model.safetensors"), device=str(device)))
        state = torch.load(path / "training-state.pt", map_location=device, weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["cpu_rng"].cpu())
        if device.type == "cuda" and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
        return ResumeState(
            step=int(state["step"]),
            samples_consumed=int(state["samples_consumed"]),
            tokens_seen=int(state["tokens_seen"]),
            last_loss=float(state.get("last_loss", 0.0)),
            path=path,
        )

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        *,
        step: int,
        samples_consumed: int,
        tokens_seen: int,
        last_loss: float,
        reason: str,
    ) -> Path:
        name = self._name(step)
        final = self.root / name
        if final.exists() and self.latest() == final:
            self._backup(final)
            return final
        temporary = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=self.root))
        temporary.chmod(0o700)
        try:
            tensors = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
            model_path = temporary / "model.safetensors"
            save_file(tensors, str(model_path))
            model_path.chmod(0o600)
            state_path = temporary / "training-state.pt"
            torch.save({
                "step": int(step),
                "samples_consumed": int(samples_consumed),
                "tokens_seen": int(tokens_seen),
                "last_loss": float(last_loss),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            }, state_path)
            state_path.chmod(0o600)
            write_json(temporary / "metadata.json", {
                "complete": True,
                "config_fingerprint": self.config_fingerprint,
                "step": int(step),
                "samples_consumed": int(samples_consumed),
                "tokens_seen": int(tokens_seen),
                "reason": reason,
                "files": {
                    "model.safetensors": {
                        "size": model_path.stat().st_size,
                        "sha256": _sha256(model_path),
                    },
                    "training-state.pt": {
                        "size": state_path.stat().st_size,
                        "sha256": _sha256(state_path),
                    },
                },
            })
            if final.exists():
                shutil.rmtree(final)
            os.replace(temporary, final)
            self._prune()
            self._backup(final)
            return final
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)

    def _prune(self) -> None:
        for stale in self._candidates()[self.keep:]:
            shutil.rmtree(stale)

    def _backup(self, checkpoint: Path) -> None:
        if self.backup_dir is None:
            return
        ensure_private_directory(self.backup_dir)
        destination = self.backup_dir / checkpoint.name
        temporary = self.backup_dir / f".{checkpoint.name}.{os.getpid()}.tmp"
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(checkpoint, temporary)
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
        retained = sorted(
            (path for path in self.backup_dir.glob("step-*") if path.is_dir()),
            key=lambda path: path.name,
            reverse=True,
        )
        for stale in retained[self.keep:]:
            shutil.rmtree(stale)
