from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
from typing import Any

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.nn import functional as F

from checkpoint import CheckpointManager
from common import append_jsonl, ensure_private_directory, load_config, make_private_file, require_cuda, write_json
from data import END, corpus_hash, corpus_manifest, example_pairs, iter_corpus_documents, smoke_corpus
from model import FoundationGPT, model_config_from_depth, param_count, save_model


def stable_fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    ).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def source_signature(config: dict[str, Any]) -> dict[str, Any]:
    corpus_path = config.get("corpus_path")
    if corpus_path:
        return {
            "kind": "files",
            "manifest": corpus_manifest(corpus_path),
            "max_chars": int(config.get("max_chars") or 0),
        }
    examples = example_pairs(config.get("examples") or [])
    corpus = smoke_corpus(
        examples,
        int(config.get("max_chars") or 20_000),
        extra=str(config.get("system_prompt") or ""),
    )
    return {"kind": "smoke", "sha256": corpus_hash(corpus), "chars": len(corpus)}


def prepare_token_cache(
    tokenizer: Tokenizer,
    tokenizer_path: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[np.memmap, dict[str, Any]]:
    cache_path = output_dir / "pretrain-tokens.bin"
    metadata_path = output_dir / "pretrain-tokens.json"
    signature = {
        "source": source_signature(config),
        "tokenizer_sha256": file_sha256(tokenizer_path),
    }
    if cache_path.is_file() and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}
        expected_size = int(metadata.get("tokens", 0)) * np.dtype(np.uint32).itemsize
        if metadata.get("signature") == signature and expected_size == cache_path.stat().st_size:
            return np.memmap(cache_path, dtype=np.uint32, mode="r"), metadata

    partial_path = output_dir / ".pretrain-tokens.partial"
    partial_metadata_path = output_dir / ".pretrain-tokens.partial.json"
    partial_metadata: dict[str, Any] = {}
    if partial_metadata_path.is_file():
        try:
            partial_metadata = json.loads(partial_metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            partial_metadata = {}
    if cache_path.is_file() and partial_metadata.get("signature") == signature:
        expected_size = int(partial_metadata.get("bytes", -1))
        if expected_size == cache_path.stat().st_size and int(partial_metadata.get("tokens", 0)) > 0:
            metadata = {
                "signature": signature,
                "tokens": int(partial_metadata["tokens"]),
                "documents": int(partial_metadata["documents"]),
                "source_documents": int(partial_metadata.get("source_documents", partial_metadata["documents"])),
                "tokens_sha256": str(partial_metadata["tokens_sha256"]),
            }
            write_json(metadata_path, metadata)
            partial_metadata_path.unlink(missing_ok=True)
            return np.memmap(cache_path, dtype=np.uint32, mode="r"), metadata
    if partial_metadata.get("signature") != signature or not partial_path.is_file():
        partial_path.unlink(missing_ok=True)
        partial_metadata_path.unlink(missing_ok=True)
        partial_metadata = {}

    digest = hashlib.sha256()
    token_count = int(partial_metadata.get("tokens", 0))
    document_count = int(partial_metadata.get("documents", 0))
    source_documents = int(partial_metadata.get("source_documents", document_count))
    recorded_bytes = int(partial_metadata.get("bytes", 0))
    if partial_path.is_file():
        with partial_path.open("r+b") as partial:
            partial.truncate(recorded_bytes)
        with partial_path.open("rb") as partial:
            while chunk := partial.read(1_048_576):
                digest.update(chunk)
    corpus_path = config.get("corpus_path")
    if corpus_path:
        documents = iter_corpus_documents(corpus_path, int(config.get("max_chars") or 0))
    else:
        examples = example_pairs(config.get("examples") or [])
        documents = iter([smoke_corpus(
            examples,
            int(config.get("max_chars") or 20_000),
            extra=str(config.get("system_prompt") or ""),
        )])
    separator = tokenizer.token_to_id(END)

    def persist_partial(destination: Any) -> None:
        destination.flush()
        os.fsync(destination.fileno())
        write_json(partial_metadata_path, {
            "signature": signature,
            "tokens": token_count,
            "documents": document_count,
            "source_documents": source_documents,
            "bytes": destination.tell(),
            "tokens_sha256": digest.hexdigest(),
        })

    mode = "ab" if partial_path.is_file() else "wb"
    with partial_path.open(mode) as destination:
        os.fchmod(destination.fileno(), 0o600)
        for source_index, document in enumerate(documents):
            if source_index < source_documents:
                continue
            source_documents += 1
            ids = tokenizer.encode(document).ids
            if separator is not None:
                ids.append(separator)
            if not ids:
                continue
            values = np.asarray(ids, dtype=np.uint32)
            payload = values.tobytes()
            destination.write(payload)
            digest.update(payload)
            token_count += len(ids)
            document_count += 1
            if document_count % 1024 == 0:
                persist_partial(destination)
        persist_partial(destination)
    if source_signature(config) != signature["source"]:
        partial_path.unlink(missing_ok=True)
        partial_metadata_path.unlink(missing_ok=True)
        raise RuntimeError("Foundation corpus changed while the token cache was being built.")
    if token_count < 2:
        partial_path.unlink(missing_ok=True)
        partial_metadata_path.unlink(missing_ok=True)
        raise ValueError("Foundation pretraining corpus produced fewer than two tokens.")
    os.replace(partial_path, cache_path)
    make_private_file(cache_path)
    metadata = {
        "signature": signature,
        "tokens": token_count,
        "documents": document_count,
        "source_documents": source_documents,
        "tokens_sha256": digest.hexdigest(),
    }
    write_json(metadata_path, metadata)
    partial_metadata_path.unlink(missing_ok=True)
    return np.memmap(cache_path, dtype=np.uint32, mode="r"), metadata


def coprime_stride(windows: int, seed: int) -> int:
    if windows <= 1:
        return 1
    candidate = max(1, (2 * abs(seed) + 1) % windows)
    while math.gcd(candidate, windows) != 1:
        candidate = (candidate + 1) % windows or 1
    return candidate


def token_batch(
    tokens: np.memmap,
    *,
    samples_consumed: int,
    batch_size: int,
    sequence_length: int,
    seed: int,
) -> np.ndarray:
    if len(tokens) < sequence_length + 1:
        repeats = math.ceil((sequence_length + 1) / len(tokens))
        expanded = np.tile(np.asarray(tokens), repeats)
        return np.stack([expanded[:sequence_length + 1]] * batch_size)
    windows = max(1, (len(tokens) - 1) // sequence_length)
    stride = coprime_stride(windows, seed)
    offset = abs(seed) % windows
    rows: list[np.ndarray] = []
    for index in range(batch_size):
        window = ((samples_consumed + index) * stride + offset) % windows
        start = window * sequence_length
        rows.append(np.asarray(tokens[start:start + sequence_length + 1]))
    return np.stack(rows)


def learning_rate_multiplier(step: int, *, warmup_steps: int, total_steps: int, min_ratio: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max((step + 1) / warmup_steps, 1 / max(warmup_steps, 1))
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


class StopRequest:
    signal_number: int | None = None

    def install(self) -> None:
        def request_stop(number: int, _frame: Any) -> None:
            self.signal_number = number

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)


def main(argv: list[str] | None = None) -> None:
    config = load_config(argv)
    require_cuda("pretrain")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    seed = int(config.get("seed") or 1337)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    tokenizer_path = Path(config["tokenizer_dir"]) / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    sequence_length = int(config.get("sequence_length") or 64)
    depth = int(config.get("depth") or 2)
    model_config = model_config_from_depth(depth, tokenizer.get_vocab_size(), sequence_length)
    model = FoundationGPT(model_config).to(device)

    output_dir = ensure_private_directory(config["output_dir"])
    work_dir = ensure_private_directory(config.get("work_dir") or output_dir)
    tokens, token_metadata = prepare_token_cache(tokenizer, tokenizer_path, config, work_dir)
    steps = max(1, int(config.get("steps") or 2))
    batch_size = max(1, int(config.get("batch_size") or 2))
    accumulation_steps = max(1, int(config.get("gradient_accumulation_steps") or 1))
    learning_rate = float(config.get("learning_rate") or 3e-4)
    weight_decay = float(config.get("weight_decay") if config.get("weight_decay") is not None else 0.1)
    configured_warmup = config.get("warmup_steps")
    warmup_steps = min(
        max(0, int(configured_warmup)) if configured_warmup is not None else min(2000, max(1, steps // 100)),
        steps,
    )
    min_lr_ratio = float(config.get("min_lr_ratio") if config.get("min_lr_ratio") is not None else 0.1)
    gradient_clip = float(config.get("gradient_clip") or 1.0)
    bf16 = bool(config.get("bf16", True))
    if bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Foundation BF16 training was requested, but this CUDA device does not support BF16.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        fused=True,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda current: learning_rate_multiplier(
            current,
            warmup_steps=warmup_steps,
            total_steps=steps,
            min_ratio=min_lr_ratio,
        ),
    )
    recovery_config = {
        "model": model_config,
        "steps": steps,
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "min_lr_ratio": min_lr_ratio,
        "gradient_clip": gradient_clip,
        "bf16": bf16,
        "seed": seed,
        "token_cache": token_metadata,
    }
    manager = CheckpointManager(
        work_dir,
        stable_fingerprint(recovery_config),
        keep=int(config.get("keep_checkpoints") or 3),
        backup_dir=config.get("checkpoint_backup_dir"),
    )
    start_step = 0
    samples_consumed = 0
    tokens_seen = 0
    last_loss = 0.0
    if bool(config.get("resume", True)):
        resumed = manager.load(model, optimizer, scheduler, device)
        if resumed is not None:
            start_step = resumed.step
            samples_consumed = resumed.samples_consumed
            tokens_seen = resumed.tokens_seen
            last_loss = resumed.last_loss
            print(json.dumps({"event": "resumed", "step": start_step, "checkpoint": str(resumed.path)}), flush=True)

    checkpoint_steps = max(1, int(config.get("checkpoint_interval_steps") or 500))
    checkpoint_seconds = max(1, int(config.get("checkpoint_interval_seconds") or 1800))
    log_steps = max(1, int(config.get("log_interval_steps") or 10))
    metrics_path = work_dir / "training-metrics.jsonl"
    stop = StopRequest()
    stop.install()
    started = time.monotonic()
    last_log_time = started
    last_log_tokens = tokens_seen
    last_checkpoint_time = started
    model.train()

    for step in range(start_step, steps):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _microstep in range(accumulation_steps):
            rows = token_batch(
                tokens,
                samples_consumed=samples_consumed,
                batch_size=batch_size,
                sequence_length=sequence_length,
                seed=seed,
            )
            tensor = torch.from_numpy(rows.astype(np.int64, copy=False)).to(device=device, non_blocking=True)
            inputs = tensor[:, :-1]
            labels = tensor[:, 1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                logits = model(inputs)
                raw_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                loss = raw_loss / accumulation_steps
            if not bool(torch.isfinite(raw_loss)):
                manager.save(
                    model, optimizer, scheduler,
                    step=step,
                    samples_consumed=samples_consumed,
                    tokens_seen=tokens_seen,
                    last_loss=last_loss,
                    reason="non-finite-loss",
                )
                raise FloatingPointError(f"Non-finite pretraining loss at step {step}: {raw_loss.item()}")
            loss.backward()
            accumulated_loss += float(raw_loss.item())
            samples_consumed += batch_size
            tokens_seen += int(labels.numel())
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip).item())
        if not math.isfinite(gradient_norm):
            manager.save(
                model, optimizer, scheduler,
                step=step,
                samples_consumed=samples_consumed - batch_size * accumulation_steps,
                tokens_seen=tokens_seen - batch_size * sequence_length * accumulation_steps,
                last_loss=last_loss,
                reason="non-finite-gradient",
            )
            raise FloatingPointError(f"Non-finite gradient norm at step {step}: {gradient_norm}")
        optimizer.step()
        scheduler.step()
        completed_step = step + 1
        last_loss = accumulated_loss / accumulation_steps
        now = time.monotonic()

        if completed_step % log_steps == 0 or completed_step == steps:
            interval = max(now - last_log_time, 1e-6)
            record = {
                "event": "progress",
                "step": completed_step,
                "total_steps": steps,
                "loss": last_loss,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "gradient_norm": gradient_norm,
                "tokens_seen": tokens_seen,
                "tokens_per_second": (tokens_seen - last_log_tokens) / interval,
                "elapsed_seconds": now - started,
                "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            }
            append_jsonl(metrics_path, record)
            print(json.dumps(record, separators=(",", ":")), flush=True)
            last_log_time = now
            last_log_tokens = tokens_seen

        checkpoint_due = (
            completed_step % checkpoint_steps == 0
            or now - last_checkpoint_time >= checkpoint_seconds
            or completed_step == steps
            or stop.signal_number is not None
        )
        if checkpoint_due:
            reason = "signal" if stop.signal_number is not None else ("complete" if completed_step == steps else "periodic")
            checkpoint = manager.save(
                model, optimizer, scheduler,
                step=completed_step,
                samples_consumed=samples_consumed,
                tokens_seen=tokens_seen,
                last_loss=last_loss,
                reason=reason,
            )
            print(json.dumps({"event": "checkpoint", "step": completed_step, "path": str(checkpoint), "reason": reason}), flush=True)
            last_checkpoint_time = now
        if stop.signal_number is not None:
            raise SystemExit(128 + stop.signal_number)

    save_model(model, output_dir)
    write_json(output_dir / "metrics.json", {
        "ok": True,
        "steps": steps,
        "loss": last_loss,
        "parameters": param_count(model),
        "depth": depth,
        "width": model_config["width"],
        "heads": model_config["heads"],
        "tokens_seen": tokens_seen,
        "training_tokens": int(token_metadata["tokens"]),
        "token_cache_sha256": token_metadata["tokens_sha256"],
        "bf16": bf16,
        "device": str(device),
    })


if __name__ == "__main__":
    main()
