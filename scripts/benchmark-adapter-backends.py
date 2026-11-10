"""Opt-in CUDA comparison; run each backend in a fresh process with pinned dependencies.

Uses TT's assistant-only dataset, collator, Trainer settings and saved-adapter
format for both backends. No downloads, dependency changes or model activation.
Unsloth is an experimental benchmark dependency, not a production backend.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["transformers", "unsloth"], required=True)
    parser.add_argument("--training-file", type=Path, required=True)
    parser.add_argument("--hyperparameters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.seed <= 4294967295:
        parser.error("seed must be a uint32")
    args.output.mkdir(parents=True, exist_ok=False)
    # The explicit recipe, not a caller's SageMaker overrides, defines the run.
    for key in list(os.environ):
        if key.startswith("SM_HP_"):
            del os.environ[key]
    os.environ["TT_HYPERPARAMETERS_PATH"] = str(args.hyperparameters.resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # Match the bundled runner's default (minimalMachineLearningEnvironment).
    os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
    started = time.perf_counter()
    if args.backend == "unsloth":
        # Must patch before importing Transformers/PEFT.
        from unsloth import FastLanguageModel
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training/adapter/src"))
    import train
    import torch
    train.HP["seed"] = str(args.seed)
    train.MODEL_DIR = args.output / "model"
    train.MODEL_DIR.mkdir()
    train.assert_certified_request()
    train.assert_certified_base_model_revision(
        train.hp("base_model"), train.hp("base_model_revision"), "Benchmark base model", local_source=False,
    )
    if train.hp("base_model") != "Qwen/Qwen3.5-2B" or train.hp("gradient_checkpointing", "false") != "false":
        parser.error("this matched benchmark currently supports Qwen/Qwen3.5-2B without gradient checkpointing")
    hardware = train.require_cuda()
    rows = [json.loads(line) for line in args.training_file.read_text().splitlines() if line.strip()]
    tokenized = {}
    original_dataset = train.AssistantOnlyDataset

    def capture_dataset(*positional, **kwargs):
        dataset = original_dataset(*positional, **kwargs)
        digest = hashlib.sha256()
        for row in dataset.examples:
            digest.update(json.dumps({k: v.tolist() for k, v in row.items()}, sort_keys=True).encode())
        tokenized.update(sha256=digest.hexdigest(), rows=len(dataset),
                         input_tokens=sum(len(row["input_ids"]) for row in dataset.examples),
                         supervised_tokens=sum(int((row["labels"] != -100).sum()) for row in dataset.examples))
        return dataset
    train.AssistantOnlyDataset = capture_dataset
    if args.backend == "unsloth":
        def load_model(source):
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=source, revision=train.hp("base_model_revision"),
                max_seq_length=train.hp_int("max_seq_length", 2048),
                dtype=torch.bfloat16, load_in_4bit=False, load_in_16bit=True,
                full_finetuning=False, trust_remote_code=False,
                text_only=True, use_exact_model_name=True,
                use_gradient_checkpointing=False, random_state=args.seed,
            )
            train.assert_certified_model_config(model.config, expected_model_id=train.hp("base_model"))
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            return model, tokenizer, torch.bfloat16

        def apply_lora(model):
            return FastLanguageModel.get_peft_model(
                model, r=train.hp_int("lora_rank", 16),
                lora_alpha=train.hp_int("lora_alpha", 32),
                lora_dropout=train.hp_float("lora_dropout", .05),
                target_modules=train.parse_lora_target_modules(train.hp("lora_target_modules", "all-linear")),
                bias="none", use_gradient_checkpointing=False, random_state=args.seed,
            )
        train.create_model_and_tokenizer = load_model
        train.apply_lora = apply_lora
    original_lora = train.apply_lora
    initialization = {}

    def capture_lora(model):
        model = original_lora(model)
        digest = hashlib.sha256()
        for name, parameter in sorted(model.named_parameters()):
            if parameter.requires_grad:
                digest.update(name.encode())
                digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        initialization["trainable_weights_sha256"] = digest.hexdigest()
        return model
    train.apply_lora = capture_lora
    metrics, dtype = train.run_training(rows, train.hp("base_model"))
    versions = {}
    for package in ["torch", "transformers", "peft", "accelerate", "unsloth", "unsloth_zoo", "triton"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    result = dict(backend=args.backend, python_version=sys.version, hardware=hardware, versions=versions,
                  hyperparameters=train.HP, dtype=str(dtype), tokenized=tokenized, initialization=initialization,
                  torch_disable_native_jit=os.environ["TORCH_DISABLE_NATIVE_JIT"],
                  data_sha256=hashlib.sha256(args.training_file.read_bytes()).hexdigest(),
                  elapsed_seconds=time.perf_counter()-started, metrics=metrics)
    (args.output / "result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
