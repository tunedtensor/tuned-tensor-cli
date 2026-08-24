from __future__ import annotations

import hashlib
import re
from typing import Any

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


def train_tokenizer(corpus: str, vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(tokenizer_models.BPE(unk_token=UNK))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=max(int(vocab_size), len(SPECIAL_TOKENS) + 32),
        special_tokens=SPECIAL_TOKENS,
        min_frequency=1,
    )
    tokenizer.train_from_iterator([corpus], trainer=trainer)
    return tokenizer


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
