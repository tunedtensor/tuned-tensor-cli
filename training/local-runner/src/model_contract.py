from __future__ import annotations

from typing import Any


CERTIFIED_BASE_MODEL = "Qwen/Qwen3.5-2B"
NEMOTRON_BASE_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
MUSE_GLIMMER_BASE_MODEL = "meta-models/Muse-Glimmer-30B"
# Immutable Hugging Face revision reviewed for Nemotron local fine-tuning.
# Node validates local snapshots against this revision and its certified file
# digests before invoking Python.
NEMOTRON_BASE_MODEL_REVISION = "ce38b6ab8b252b4b8ee7165b4605e93191cafd73"
# Immutable Hugging Face revision reviewed and certified for Muse Glimmer.
MUSE_GLIMMER_BASE_MODEL_REVISION = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
CERTIFIED_BASE_MODELS = (CERTIFIED_BASE_MODEL, NEMOTRON_BASE_MODEL, MUSE_GLIMMER_BASE_MODEL)
CERTIFIED_QWEN_TEXT_CONFIG = {
    "model_type": "qwen3_5_text",
    "hidden_size": 2048,
    "num_hidden_layers": 24,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "intermediate_size": 6144,
    "vocab_size": 248320,
}
CERTIFIED_MUSE_GLIMMER_TEXT_CONFIG = {
    "model_type": "muse_glimmer_text",
    "hidden_size": 6656,
    "num_hidden_layers": 52,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "intermediate_size": 19968,
    "vocab_size": 202048,
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


def assert_certified_model_config(
    value: Any,
    label: str = "base-model config",
    expected_model_id: str | None = None,
) -> None:
    model_type = _field(value, "model_type")
    architectures = _field(value, "architectures")

    if model_type in ("qwen3_5", "qwen3_5_text"):
        actual_model_id = CERTIFIED_BASE_MODEL
        if model_type == "qwen3_5":
            text_config = _field(value, "text_config")
            if (
                not isinstance(architectures, (list, tuple))
                or "Qwen3_5ForConditionalGeneration" not in architectures
                or text_config is None
            ):
                raise ValueError(f"{label} is not the certified {CERTIFIED_BASE_MODEL} architecture")
        else:
            # AutoModelForCausalLM exposes the selected text sub-config after it
            # dispatches the verified repository-level Qwen3.5 config.
            text_config = value
        certified = CERTIFIED_QWEN_TEXT_CONFIG
    elif model_type in ("muse_glimmer", "muse_glimmer_text"):
        actual_model_id = MUSE_GLIMMER_BASE_MODEL
        if model_type == "muse_glimmer":
            text_config = _field(value, "text_config")
            if (
                not isinstance(architectures, (list, tuple))
                or "MuseGlimmerForConditionalGeneration" not in architectures
                or text_config is None
            ):
                raise ValueError(f"{label} is not the certified {MUSE_GLIMMER_BASE_MODEL} architecture")
        else:
            text_config = value
        certified = CERTIFIED_MUSE_GLIMMER_TEXT_CONFIG
    elif model_type == "nemotron_h":
        actual_model_id = NEMOTRON_BASE_MODEL
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
        if expected_model_id and expected_model_id != actual_model_id:
            raise ValueError(
                f"{label} does not match requested base model {expected_model_id}; "
                f"it matches {actual_model_id}"
            )
        return
    else:
        raise ValueError(f"{label} is not a certified TT Local base-model architecture")

    for name, expected in certified.items():
        actual = _field(text_config, name)
        if actual != expected:
            raise ValueError(
                f"{label} is not the certified {actual_model_id} architecture: "
                f"text_config.{name} must be {expected!r}, got {actual!r}"
            )
    if expected_model_id and expected_model_id != actual_model_id:
        raise ValueError(
            f"{label} does not match requested base model {expected_model_id}; "
            f"it matches {actual_model_id}"
        )


def assert_certified_base_model(model_id: str, label: str = "base model") -> None:
    if model_id not in CERTIFIED_BASE_MODELS:
        supported = ", ".join(CERTIFIED_BASE_MODELS)
        raise ValueError(f"{label} must be one of {supported}; got {model_id!r}")


def assert_certified_base_model_revision(base_model: str, revision: str | None, label: str = "base model revision") -> None:
    """Require the certified immutable revision for pinned training models.

    Qwen remains unpinned for backward compatibility. Nemotron and Muse
    Glimmer loads are bound here; local snapshot contents are additionally
    verified by the Node runtime.
    """
    expected = {
        NEMOTRON_BASE_MODEL: NEMOTRON_BASE_MODEL_REVISION,
        MUSE_GLIMMER_BASE_MODEL: MUSE_GLIMMER_BASE_MODEL_REVISION,
    }.get(base_model)
    if expected is None:
        return
    if revision is None:
        raise ValueError(f"{label} is required for {base_model}")
    if revision != expected:
        raise ValueError(
            f"{label} must be {expected} for {base_model}; got {revision!r}"
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
