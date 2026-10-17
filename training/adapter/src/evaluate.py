from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import time
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any

from model_contract import (
    CERTIFIED_BASE_MODEL,
    assert_certified_base_model,
    assert_certified_base_model_revision,
    assert_certified_model_config,
    chat_template_kwargs,
    cuda_model_load_device_map,
    uncertified_load_kwargs,
)

MAX_ARCHIVE_MEMBERS = 20_000
MAX_ARCHIVE_EXPANDED_BYTES = 20 * 1024 * 1024 * 1024


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def configure_hugging_face_cache(cache_home: str | None) -> None:
    """Treat model_cache as HF_HOME and keep legacy overrides consistent."""
    if not cache_home:
        return
    home = Path(cache_home).expanduser().absolute()
    hub = home / "hub"
    os.environ.update({
        "HF_HOME": str(home),
        "HF_HUB_CACHE": str(hub),
        "HUGGINGFACE_HUB_CACHE": str(hub),
    })
    for deprecated in (
        "TRANSFORMERS_CACHE",
        "PYTORCH_TRANSFORMERS_CACHE",
        "PYTORCH_PRETRAINED_BERT_CACHE",
    ):
        os.environ.pop(deprecated, None)


def import_runtime_dependencies() -> None:
    """Import libraries only after Hugging Face cache settings are final."""
    global torch, PeftModel, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    import torch as torch_module
    from peft import PeftModel as peft_model
    from transformers import (
        AutoModelForCausalLM as auto_model_for_causal_lm,
        AutoModelForImageTextToText as auto_model_for_image_text_to_text,
        AutoTokenizer as auto_tokenizer,
    )

    torch = torch_module
    PeftModel = peft_model
    AutoModelForCausalLM = auto_model_for_causal_lm
    AutoModelForImageTextToText = auto_model_for_image_text_to_text
    AutoTokenizer = auto_tokenizer


def strip_file_uri(value: str | None) -> str | None:
    if not value:
        return value
    if value.startswith("file://"):
        return value[7:]
    return value


def _extract_adapter_archive(path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise ValueError(f"Model archive exceeds {MAX_ARCHIVE_MEMBERS} members")
        expanded_bytes = sum(max(0, member.size) for member in members if member.isfile())
        if expanded_bytes > MAX_ARCHIVE_EXPANDED_BYTES:
            raise ValueError("Model archive exceeds the 20 GiB expanded-size limit")
        for member in members:
            member_path = (destination / member.name).resolve()
            try:
                member_path.relative_to(destination_root)
            except ValueError as exc:
                raise ValueError(f"Unsafe archive member: {member.name}") from exc
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"Unsafe archive member type: {member.name}")
        try:
            archive.extractall(destination, filter="data")
        except TypeError:
            archive.extractall(destination)


def resolve_adapter_path(value: str | None, tmp: Path) -> str | None:
    path_value = strip_file_uri(value)
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_file() and path.name.endswith(".tar.gz"):
        extracted = tmp / "adapter"
        _extract_adapter_archive(path, extracted)
        candidates = sorted(extracted.rglob("adapter_config.json"))
        if candidates:
            return str(candidates[0].parent)
        raise ValueError(f"Adapter archive contains no adapter_config.json: {path}")
    if not path.is_dir():
        raise ValueError(f"Adapter path is not a directory or .tar.gz archive: {path}")
    return str(path)


def format_prompt(
    tokenizer: Any,
    system: str,
    prompt: str,
    base_model: str = CERTIFIED_BASE_MODEL,
) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs(base_model),
        )
    except Exception as exc:
        raise ValueError(f"Model chat-template rendering failed: {exc}") from exc


def resolve_device(device: str) -> str:
    if device not in {"cuda", "cpu"}:
        raise ValueError(f"device must be cuda or cpu; got {device!r}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation was requested, but torch.cuda.is_available() is false")
    return device


def _assert_certified_model_source(base_model: str, model_source: str) -> None:
    # A hub source must be the requested model; certified or not, it is
    # checked by assert_certified_base_model.
    if not Path(model_source).exists() and model_source != base_model:
        raise ValueError(
            f"The bundled evaluator does not certify {model_source!r}"
        )


def load_text_model(
    payload: dict[str, Any],
    adapter_path: str | None,
    model_loader: str = "causal_lm",
):
    base_model = str(payload["base_model"])
    model_source = str(payload.get("model_source", base_model))
    assert_certified_base_model(base_model, "Evaluation base model")
    _assert_certified_model_source(base_model, model_source)
    revision = payload.get("base_model_revision")
    assert_certified_base_model_revision(
        base_model,
        revision,
        "Evaluation base model revision",
        local_source=Path(model_source).is_dir(),
    )
    device = resolve_device(str(payload.get("device", "cuda")))
    token = os.getenv("HF_TOKEN")
    source_kwargs: dict[str, Any] = {
        "trust_remote_code": False,
        "token": token,
    }
    if not Path(model_source).exists():
        source_kwargs["local_files_only"] = True
    if revision and not Path(model_source).exists():
        source_kwargs["revision"] = revision

    tokenizer = AutoTokenizer.from_pretrained(model_source, **source_kwargs)
    if tokenizer.eos_token_id is None:
        raise ValueError(f"{base_model} tokenizer has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only generation must end every padded prompt on a real token.
    tokenizer.padding_side = "left"

    dtype = None
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model_kwargs: dict[str, Any] = {**source_kwargs, **uncertified_load_kwargs(base_model)}
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    if device == "cuda":
        model_kwargs["device_map"] = cuda_model_load_device_map(torch)

    model_class = (
        AutoModelForImageTextToText
        if model_loader == "image_text_to_text"
        else AutoModelForCausalLM
    )
    model = model_class.from_pretrained(model_source, **model_kwargs)
    assert_certified_model_config(model.config, expected_model_id=base_model)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    if device != "cuda" or model_kwargs["device_map"] is None:
        model = model.to(device)
    model.eval()
    return model, tokenizer, device


def sampling_kwargs(generation: dict[str, Any]) -> dict[str, Any]:
    temperature = float(generation.get("temperature", 0))
    if temperature <= 0:
        return {"do_sample": False}
    return {
        "do_sample": True,
        "temperature": temperature,
        "top_p": float(generation.get("top_p", 1)),
    }


def generate_text_batch(
    model: Any,
    tokenizer: Any,
    system: str,
    examples: list[dict[str, Any]],
    generation: dict[str, Any],
    base_model: str = CERTIFIED_BASE_MODEL,
) -> list[dict[str, Any]]:
    if any(example.get("input_assets") for example in examples):
        raise ValueError("The bundled SFT evaluator is text-only and does not accept input_assets")
    formatted = [format_prompt(tokenizer, system, str(example["input"]), base_model)
                 for example in examples]
    inputs = tokenizer(formatted, return_tensors="pt", padding=True, add_special_tokens=False)
    target_device = next(model.parameters()).device
    inputs = {key: value.to(target_device) for key, value in inputs.items()}
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(generation.get("max_new_tokens", 256)),
            pad_token_id=tokenizer.pad_token_id,
            **sampling_kwargs(generation),
        )
    input_length = int(inputs["input_ids"].shape[-1])
    # Slice at the padded width, not each prompt's unpadded length.
    actuals = [tokenizer.decode(row[input_length:], skip_special_tokens=True).strip()
               for row in generated]
    # Decoding synchronizes CUDA before measuring wall latency.
    latency_ms = max(0, round((time.perf_counter() - started) * 1000))
    return [{
        "id": str(example["id"]),
        "actual": actual,
        # Each request waits for the whole batch; this is batch wall latency.
        "latency_ms": latency_ms,
    } for example, actual in zip(examples, actuals, strict=True)]


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary: str | None = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                prefix=f".{path.name}.", delete=False) as handle:
            temporary = handle.name
            json.dump(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def checkpoint_identity(payload: dict[str, Any], adapter_path: str | None) -> str:
    digest = hashlib.sha256(str(payload.get("checkpoint_key", "")).encode())
    # A retrained/replaced adapter at the same path must never reuse predictions.
    if adapter_path:
        root = Path(adapter_path)
        for path in sorted(root.rglob("*")):
            if path.is_file():
                digest.update(str(path.relative_to(root)).encode())
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
    return digest.hexdigest()


def read_checkpoint(path: Path, identity: str, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        value = load_json(path)
        results = value.get("results")
        if value.get("identity") != identity or not isinstance(results, list) or len(results) > len(examples):
            return []
        for example, result in zip(examples, results):
            if (not isinstance(result, dict) or result.get("id") != str(example["id"])
                    or not isinstance(result.get("actual"), str)
                    or type(result.get("latency_ms")) is not int or result["latency_ms"] < 0):
                return []
        return results
    except (OSError, ValueError):
        # A missing, interrupted or incompatible checkpoint is a fresh evaluation.
        return []


def emit_progress(phase: str, completed: int, total: int, batch_size: int,
                  elapsed: float = 0, eta: float | None = None) -> None:
    print("@@tt-evaluation " + json.dumps({
        "phase": phase, "completed": completed, "total": total,
        "batch_size": batch_size, "elapsed_seconds": round(elapsed, 2),
        "eta_seconds": round(eta, 2) if eta is not None else None,
    }), flush=True)


def evaluate_batches(payload: dict[str, Any], adapter_path: str | None,
                     checkpoint: Path, model_loader: str) -> list[dict[str, Any]]:
    examples = payload.get("examples", [])
    batch_size = payload.get("batch_size", 4)
    if type(batch_size) is not int or not 1 <= batch_size <= 128:
        raise ValueError("batch_size must be an integer between 1 and 128")
    generation = payload.get("generation", {})
    if not isinstance(generation, dict):
        raise ValueError("generation must be a JSON object")
    identity = checkpoint_identity(payload, adapter_path)
    results = read_checkpoint(checkpoint, identity, examples) if payload.get("checkpoint_key") else []
    resumed = len(results)
    emit_progress("resuming" if resumed else "loading", resumed, len(examples), batch_size)
    if resumed == len(examples):
        return results
    model, tokenizer, device = load_text_model(payload, adapter_path, model_loader)
    started = time.perf_counter()
    emit_progress("generating", resumed, len(examples), batch_size)
    while len(results) < len(examples):
        batch = examples[len(results):len(results) + batch_size]
        predictions = None
        try:
            predictions = generate_text_batch(model, tokenizer, str(payload.get("system", "")),
                                              batch, generation, str(payload["base_model"]))
        except torch.cuda.OutOfMemoryError:
            if device != "cuda" or len(batch) == 1:
                raise
            # Retry only allocation failures; release the failed call's tensors first.
            batch_size = max(1, len(batch) // 2)
        if predictions is None:
            # The exception's traceback is now released, including failed tensors.
            torch.cuda.empty_cache()
            emit_progress("reducing_batch", len(results), len(examples), batch_size)
            continue
        results.extend(predictions)
        write_json_atomic(checkpoint, {"identity": identity, "results": results})
        elapsed = time.perf_counter() - started
        eta = elapsed / (len(results) - resumed) * (len(examples) - len(results))
        emit_progress("generating", len(results), len(examples), batch_size, elapsed, eta)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()

    payload = load_json(Path(args.input))
    if payload.get("protocol_version") != 2:
        raise ValueError("Unsupported inference protocol; expected protocol_version 2")
    model_loader = str(payload.get("model_loader") or "causal_lm")
    if model_loader not in ("causal_lm", "image_text_to_text"):
        raise ValueError(
            f"The bundled evaluator does not support model_loader={model_loader!r}; "
            "expected causal_lm or image_text_to_text"
        )
    configure_hugging_face_cache(payload.get("model_cache"))
    emit_progress("initializing", 0, len(payload.get("examples", [])), payload.get("batch_size", 4))
    import_runtime_dependencies()

    with TemporaryDirectory(prefix="tt-local-evaluate-") as tmp_dir:
        adapter_path = resolve_adapter_path(payload.get("adapter_path"), Path(tmp_dir))
        checkpoint = Path(args.checkpoint or f"{args.output}.checkpoint.json")
        results = evaluate_batches(payload, adapter_path, checkpoint, model_loader)

    output = {
        "provider": "transformers",
        "model_id": payload.get("model_id"),
        "base_model": payload.get("base_model"),
        "adapter_path": payload.get("adapter_path"),
        "generation_config": payload.get("generation", {}),
        "results": results,
    }
    output_path = Path(args.output)
    write_json_atomic(output_path, output)


if __name__ == "__main__":
    main()
