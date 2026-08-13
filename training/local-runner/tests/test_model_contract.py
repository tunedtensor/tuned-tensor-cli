from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_contract import (
    MUSE_GLIMMER_BASE_MODEL,
    NEMOTRON_BASE_MODEL,
    NEMOTRON_BASE_MODEL_REVISION,
    assert_certified_base_model_revision,
    assert_certified_model_config,
    chat_template_kwargs,
    parse_lora_target_modules,
)


class ModelContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 2048,
                "num_hidden_layers": 24,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "intermediate_size": 6144,
                "vocab_size": 248320,
            },
        }

    def test_accepts_the_certified_2b_architecture(self) -> None:
        assert_certified_model_config(self.config)
        assert_certified_model_config(self.config["text_config"])

    def test_config_validation_is_bound_to_the_requested_model_family(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match requested base model"):
            assert_certified_model_config(self.config, expected_model_id=NEMOTRON_BASE_MODEL)

        nemotron = {
            "architectures": ["NemotronHForCausalLM"],
            "model_type": "nemotron_h",
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
        with self.assertRaisesRegex(ValueError, "does not match requested base model"):
            assert_certified_model_config(nemotron, expected_model_id="Qwen/Qwen3.5-2B")

    def test_rejects_a_larger_same_family_snapshot(self) -> None:
        larger = {
            **self.config,
            "text_config": {
                **self.config["text_config"],
                "hidden_size": 2560,
            },
        }
        with self.assertRaisesRegex(ValueError, "Qwen/Qwen3.5-2B"):
            assert_certified_model_config(larger)

    def test_accepts_only_the_released_nemotron_lightning_architecture(self) -> None:
        config = {
            "architectures": ["NemotronHForCausalLM"],
            "model_type": "nemotron_h",
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
        assert_certified_model_config(config)
        with self.assertRaisesRegex(ValueError, "Nemotron-3.5-Lightning-30B-A3B-BF16"):
            assert_certified_model_config({**config, "n_routed_experts": 64})

    def test_accepts_only_the_released_muse_glimmer_text_tower(self) -> None:
        config = {
            "architectures": ["MuseGlimmerForConditionalGeneration"],
            "model_type": "muse_glimmer",
            "text_config": {
                "model_type": "muse_glimmer_text",
                "hidden_size": 6656,
                "num_hidden_layers": 52,
                "num_attention_heads": 32,
                "num_key_value_heads": 2,
                "intermediate_size": 19968,
                "vocab_size": 202048,
            },
        }
        assert_certified_model_config(config)
        assert_certified_model_config(config["text_config"])
        with self.assertRaisesRegex(ValueError, MUSE_GLIMMER_BASE_MODEL):
            assert_certified_model_config({
                **config,
                "text_config": {**config["text_config"], "hidden_size": 6784},
            })
        with self.assertRaisesRegex(ValueError, "does not match requested base model"):
            assert_certified_model_config(config, expected_model_id="Qwen/Qwen3.5-2B")

    def test_parses_model_specific_lora_targets(self) -> None:
        self.assertEqual(parse_lora_target_modules("all-linear"), "all-linear")
        self.assertEqual(
            parse_lora_target_modules("q_proj,k_proj,v_proj,o_proj,in_proj,out_proj"),
            ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj"],
        )
        with self.assertRaisesRegex(ValueError, "LoRA target modules"):
            parse_lora_target_modules("q_proj,,o_proj")

    def test_nemotron_uses_non_thinking_chat_rendering_for_task_specialization(self) -> None:
        self.assertEqual(chat_template_kwargs(NEMOTRON_BASE_MODEL), {"enable_thinking": False})
        self.assertEqual(chat_template_kwargs("Qwen/Qwen3.5-2B"), {})

    def test_nemotron_pins_and_asserts_the_certified_revision(self) -> None:
        assert_certified_base_model_revision(NEMOTRON_BASE_MODEL, NEMOTRON_BASE_MODEL_REVISION)
        with self.assertRaisesRegex(ValueError, "base model revision is required"):
            assert_certified_base_model_revision(NEMOTRON_BASE_MODEL, None)
        with self.assertRaisesRegex(ValueError, NEMOTRON_BASE_MODEL_REVISION):
            assert_certified_base_model_revision(NEMOTRON_BASE_MODEL, "deadbeef" * 5)

    def test_revision_assertion_does_not_pin_the_qwen_model(self) -> None:
        assert_certified_base_model_revision("Qwen/Qwen3.5-2B", None)
        assert_certified_base_model_revision("Qwen/Qwen3.5-2B", "any-revision")

    def test_revision_assertion_does_not_pin_the_muse_glimmer_model(self) -> None:
        assert_certified_base_model_revision(MUSE_GLIMMER_BASE_MODEL, None)
        assert_certified_base_model_revision(MUSE_GLIMMER_BASE_MODEL, "any-revision")


if __name__ == "__main__":
    unittest.main()
