from __future__ import annotations

from typing import Any


CERTIFIED_BASE_MODEL = "Qwen/Qwen3.5-2B"
NEMOTRON_BASE_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
# Immutable Hugging Face revision reviewed for Nemotron local fine-tuning.
# If there is an explicit path to the untarred/snapshot base model on disk we
# cannot verify the revision from configuration alone; callers are encouraged
# to pin it via TT_BASE_MODEL_REVISION / base_model_revision.
NEMOTRON_BASE_MODEL_REVISION = "ce38b6ab8b252b4b8ee7165b4605e93191cafd73"
CERTIFIED_BASE_MODELS = (CERTIFIED_BASE_MODEL, NEMOTRON_BASE_MODEL)
CERTIFIED_QWEN_TEXT_CONFIG = {
    "model_type": "qwen3_5_text",
    "hidden_size": 2048,
    "num_hidden_layers": 24,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "intermediate_size": 6144,
    "vocab_size": 248320,
}
CERTIFIED_NEMOTRON_CONFIG = {
    "hidden_size": 2688,
    "num_hidden_layers": 52,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "intermediate_size": 1856,
    "vocab_size": 131072,
    "n_routed_experts": 128,
    "num_experts_per_tok": 6,
    "num_nextn_predict_layers": 1,
    "max_position_embeddings": 262144,
}


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def assert_certified_model_config(value: Any, label: str = "base-model config") -> None:
    model_type = _field(value, "model_type")
    architectures = _field(value, "architectures")
    if model_type == "qwen3_5":
        text_config = _field(value, "text_config")
        if (
            not isinstance(architectures, (list, tuple))
            or "Qwen3_5ForConditionalGeneration" not in architectures
            or text_config is None
        ):
            raise ValueError(f"{label} is not the certified {CERTIFIED_BASE_MODEL} architecture")
    elif model_type == "qwen3_5_text":
        # AutoModelForCausalLM exposes the selected text sub-config after it
        # dispatches the verified repository-level Qwen3.5 config.
        text_config = value
    elif model_type == "nemotron_h":
        if (
            not isinstance(architectures, (list, tuple))
            or "NemotronHForCausalLM" not in architectures
        ):
            raise ValueError(f"{label} is not the certified {NEMOTRON_BASE_MODEL} architecture")
        for name, expected in CERTIFIED_NEMOTRON_CONFIG.items():
            actual = _field(value, name)
            if actual != expected:
                raise ValueError(
                    f"{label} is not the certified {NEMOTRON_BASE_MODEL} architecture: "
                    f"{name} must be {expected!r}, got {actual!r}"
                )
        return
    else:
        raise ValueError(f"{label} is not a certified TT Local base-model architecture")

    for name, expected in CERTIFIED_QWEN_TEXT_CONFIG.items():
        actual = _field(text_config, name)
        if actual != expected:
            raise ValueError(
                f"{label} is not the certified {CERTIFIED_BASE_MODEL} architecture: "
                f"text_config.{name} must be {expected!r}, got {actual!r}"
            )


def assert_certified_base_model(model_id: str, label: str = "base model") -> None:
    if model_id not in CERTIFIED_BASE_MODELS:
        supported = ", ".join(CERTIFIED_BASE_MODELS)
        raise ValueError(f"{label} must be one of {supported}; got {model_id!r}")


def assert_certified_base_model_revision(base_model: str, revision: str | None, label: str = "base model revision") -> None:
    """Require the certified immutable revision for the Nemotron training model.

    Qwen remains unpinned for backward compatibility; Nemotron is bound to the
    reviewed Hugging Face revision unless a local snapshot is used directly.
    """
    if base_model != NEMOTRON_BASE_MODEL:
        return
    if revision is None:
        raise ValueError(f"{label} is required for {NEMOTRON_BASE_MODEL}")
    if revision != NEMOTRON_BASE_MODEL_REVISION:
        raise ValueError(
            f"{label} must be {NEMOTRON_BASE_MODEL_REVISION} for {NEMOTRON_BASE_MODEL}; got {revision!r}"
        )


def parse_lora_target_modules(value: str | None) -> str | list[str]:
    raw = (value or "all-linear").strip()
    if raw == "all-linear":
        return raw
    modules = [item.strip() for item in raw.split(",")]
    if not modules or any(not item for item in modules):
        raise ValueError("LoRA target modules must be 'all-linear' or a comma-separated non-empty list")
    return modules


def chat_template_kwargs(model_id: str) -> dict[str, Any]:
    if model_id == NEMOTRON_BASE_MODEL:
        return {"enable_thinking": False}
    return {}
