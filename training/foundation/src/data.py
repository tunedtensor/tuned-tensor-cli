from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Iterator

from tokenizers import Tokenizer
from tokenizers import models as tokenizer_models
from tokenizers import pre_tokenizers, trainers

SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
END = "<|end|>"
UNK = "<unk>"
PAD = "<pad>"
SPECIAL_TOKENS = [UNK, PAD, SYSTEM, USER, ASSISTANT, END]
IGNORE_INDEX = -100


def train_tokenizer(corpus: str | Iterable[str], vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(tokenizer_models.BPE(unk_token=UNK))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=max(int(vocab_size), len(SPECIAL_TOKENS) + 32),
        special_tokens=SPECIAL_TOKENS,
        min_frequency=1,
    )
    tokenizer.train_from_iterator([corpus] if isinstance(corpus, str) else corpus, trainer=trainer)
    return tokenizer


def corpus_files(path: str | Path) -> list[Path]:
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise ValueError(f"Foundation corpus must not be a symbolic link: {unresolved}")
    source = unresolved.resolve()
    if source.is_file():
        files = [source]
    elif source.is_dir():
        files = sorted(
            candidate for candidate in source.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in {".txt", ".jsonl"}
        )
    else:
        raise ValueError(f"Foundation corpus path does not exist: {source}")
    if not files:
        raise ValueError(f"Foundation corpus has no .txt or .jsonl files: {source}")
    for candidate in files:
        if candidate.is_symlink():
            raise ValueError(f"Foundation corpus must not contain symbolic links: {candidate}")
    return files


def corpus_manifest(path: str | Path) -> list[dict[str, int | str]]:
    manifest: list[dict[str, int | str]] = []
    for candidate in corpus_files(path):
        info = candidate.stat()
        manifest.append({
            "path": str(candidate),
            "size": int(info.st_size),
            "mtime_ns": int(info.st_mtime_ns),
        })
    return manifest


def iter_corpus_documents(path: str | Path, max_chars: int | None = None) -> Iterator[str]:
    remaining = max_chars if max_chars and max_chars > 0 else None
    for candidate in corpus_files(path):
        if remaining is not None and remaining <= 0:
            return
        if candidate.suffix.lower() == ".jsonl":
            with candidate.open("r", encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    if remaining is not None and remaining <= 0:
                        return
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"Invalid JSONL at {candidate}:{line_number}: {error.msg}") from error
                    if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                        raise ValueError(f'Foundation JSONL rows require a string "text" field: {candidate}:{line_number}')
                    text = row["text"]
                    if remaining is not None:
                        text = text[:remaining]
                        remaining -= len(text)
                    if text:
                        yield text
        else:
            with candidate.open("r", encoding="utf-8") as source:
                while True:
                    requested = min(1_048_576, remaining) if remaining is not None else 1_048_576
                    if requested <= 0:
                        return
                    text = source.read(requested)
                    if not text:
                        break
                    if remaining is not None:
                        remaining -= len(text)
                    yield text


def corpus_documents_hash(documents: Iterable[str]) -> tuple[str, int]:
    digest = hashlib.sha256()
    chars = 0
    for document in documents:
        encoded = document.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        chars += len(document)
    return digest.hexdigest(), chars


def smoke_corpus(examples: list[dict[str, str]], max_chars: int, extra: str = "") -> str:
    chunks = [extra, *[example.get("input", "") for example in examples], *[example.get("output", "") for example in examples]]
    seed = "\n".join(part for part in chunks if part).strip() or "the quick brown fox jumps over the lazy dog"
    seed = f"{seed}\n"
    repeats = max(1, (max_chars // max(len(seed), 1)) + 1)
    return (seed * repeats)[:max_chars]


def corpus_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def format_prompt(system_prompt: str, user: str) -> str:
    system = system_prompt.strip() or "You are a helpful assistant."
    return f"{SYSTEM}{system}{END}{USER}{user}{END}{ASSISTANT}"


def format_chat(system_prompt: str, user: str, assistant: str) -> str:
    return f"{format_prompt(system_prompt, user)}{assistant}{END}"


def encode_ids(tokenizer: Tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text).ids)


def decode_ids(tokenizer: Tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False)


def decoded_byte_count(tokenizer: Tokenizer, rows: list[list[int]]) -> int:
    """Count decoded UTF-8 bytes across every target-token row."""
    return sum(len(tokenizer.decode(row).encode("utf-8")) for row in rows)


def pad_or_trim(ids: list[int], length: int, pad_id: int) -> list[int]:
    if len(ids) >= length:
        return ids[:length]
    return ids + [pad_id] * (length - len(ids))


def encode_sft_example(
    tokenizer: Tokenizer,
    system_prompt: str,
    user: str,
    assistant: str,
    sequence_length: int,
) -> tuple[list[int], list[int]]:
    pad_id = tokenizer.token_to_id(PAD) or 0
    prefix_ids = encode_ids(tokenizer, format_prompt(system_prompt, user))
    full_ids = encode_ids(tokenizer, format_chat(system_prompt, user, assistant))
    model_tokens = sequence_length + 1
    if len(full_ids) > model_tokens:
        overflow = len(full_ids) - model_tokens
        prefix_ids = prefix_ids[overflow:] if overflow < len(prefix_ids) else []
        full_ids = full_ids[-model_tokens:]
    input_ids = pad_or_trim(full_ids[:-1], sequence_length, pad_id)
    target_ids = full_ids[1:]
    first_assistant_target = max(0, min(len(prefix_ids) - 1, len(target_ids)))
    labels = [IGNORE_INDEX] * first_assistant_target + target_ids[first_assistant_target:]
    labels = pad_or_trim(labels, sequence_length, IGNORE_INDEX)
    return input_ids, labels


def window_tokens(ids: list[int], sequence_length: int) -> list[list[int]]:
    if len(ids) < sequence_length + 1:
        pad = [ids[-1] if ids else 0] * (sequence_length + 1 - len(ids))
        ids = ids + pad
    windows: list[list[int]] = []
    stride = max(1, sequence_length // 2)
    for start in range(0, len(ids) - sequence_length, stride):
        windows.append(ids[start:start + sequence_length + 1])
        if len(windows) >= 256:
            break
    if not windows:
        windows.append(ids[:sequence_length + 1])
    return windows


def parse_numeric_answer(text: str) -> float | None:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not matches:
        return None
    return float(matches[-1])


def numeric_reward(prediction: str, expected: str) -> float:
    predicted = parse_numeric_answer(prediction)
    target = parse_numeric_answer(expected)
    if predicted is None or target is None:
        return 0.0
    return 1.0 if predicted == target else 0.0


def example_pairs(raw: list[Any]) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        user = str(item.get("input") or item.get("user") or "")
        assistant = str(item.get("output") or item.get("assistant") or "")
        if user and assistant:
            pairs.append({"input": user, "output": assistant})
    return pairs
