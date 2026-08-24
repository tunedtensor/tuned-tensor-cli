from __future__ import annotations

import time
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.nn import functional as F

from common import load_config, write_json
from data import END, decoded_byte_count, encode_ids, example_pairs, format_prompt, smoke_corpus, window_tokens
from model import bits_per_byte, generate, load_model


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_bpb(model, tokenizer: Tokenizer, corpus: str, sequence_length: int, device: torch.device) -> dict[str, float | int]:
    windows = window_tokens(tokenizer.encode(corpus).ids, sequence_length)
    tensor = torch.tensor(windows[:8], dtype=torch.long, device=device)
    inputs = tensor[:, :-1]
    labels = tensor[:, 1:]
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="sum")
    byte_count = max(decoded_byte_count(tokenizer, labels.detach().cpu().tolist()), 1)
    return {
        "nll": float(nll.item()),
        "bpb": bits_per_byte(nll, byte_count),
        "tokens": int(inputs.numel()),
        "bytes": int(byte_count),
    }


def evaluate_chat(
    model,
    tokenizer: Tokenizer,
    system_prompt: str,
    examples: list[dict[str, str]],
    device: torch.device,
) -> dict[str, float | int | list[dict[str, str | bool]]]:
    rows: list[dict[str, str | bool]] = []
    correct = 0
    model.eval()
    for example in examples:
        prompt_ids = encode_ids(tokenizer, format_prompt(system_prompt, example["input"])) or [0]
        tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated = generate(
            model,
            tokens,
            max_new_tokens=32,
            stop_token_id=tokenizer.token_to_id(END),
        )[0].tolist()[len(prompt_ids):]
        text = tokenizer.decode(generated).split(END)[0].strip()
        matched = text == example["output"].strip()
        correct += int(matched)
        rows.append({"input": example["input"], "expected": example["output"], "actual": text, "match": matched})
    return {
        "examples": len(examples),
        "correct": correct,
        "accuracy": (correct / len(examples)) if examples else 0.0,
        "predictions": rows,
    }


def evaluate_inference(model, tokenizer: Tokenizer, prompt: str, device: torch.device) -> dict[str, float | int | str]:
    prompt_ids = encode_ids(tokenizer, prompt) or [0]
    tokens = torch.tensor([prompt_ids[: min(16, len(prompt_ids))]], dtype=torch.long, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    started = time.perf_counter()
    generated = generate(model, tokens, max_new_tokens=32)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - started, 1e-6)
    new_tokens = max(int(generated.size(1) - tokens.size(1)), 0)
    vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if device.type == "cuda" else 0.0
    return {
        "tokens": new_tokens,
        "seconds": elapsed,
        "tokens_per_second": new_tokens / elapsed,
        "vram_mb": vram_mb,
        "device": str(device),
    }


def main(argv: list[str] | None = None) -> None:
    config = load_config(argv)
    evaluator = str(config.get("evaluator") or "bpb")
    device = pick_device()
    tokenizer = Tokenizer.from_file(str(Path(config["tokenizer_dir"]) / "tokenizer.json"))
    model = load_model(config["model_dir"], map_location=device)
    examples = example_pairs(config.get("examples") or [])
    sequence_length = int(config.get("sequence_length") or model.config["sequence_length"])
    output_dir = Path(config["output_dir"])
    metrics: dict = {"ok": True, "evaluator": evaluator, "device": str(device)}
    if evaluator == "bpb":
        corpus = smoke_corpus(examples, int(config.get("max_chars") or 20_000), extra=str(config.get("system_prompt") or ""))
        metrics.update(evaluate_bpb(model, tokenizer, corpus, sequence_length, device))
    elif evaluator == "chat":
        metrics.update(evaluate_chat(model, tokenizer, str(config.get("system_prompt") or ""), examples, device))
    elif evaluator == "inference":
        prompt = examples[0]["input"] if examples else "hello"
        metrics.update(evaluate_inference(model, tokenizer, prompt, device))
    else:
        raise SystemExit(f"Unknown foundation evaluator: {evaluator}")
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "report.json", metrics)


if __name__ == "__main__":
    main()
