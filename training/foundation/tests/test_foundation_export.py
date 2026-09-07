"""CPU parity against the GPT-2 format consumed by the pinned serving runtime."""
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TRAINING = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRAINING / 'foundation/src'))
sys.path.append(str(TRAINING / 'adapter/src'))
from model import FoundationGPT, model_config_from_depth, save_model
from data import train_tokenizer, format_prompt
from foundation_export import export_foundation


class FoundationExportTests(unittest.TestCase):
    def test_rejects_incompatible_tokenizer_and_untied_weights(self):
        from safetensors.torch import load_file, save_file
        from tokenizers import Tokenizer, models

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenizer = train_tokenizer('hello world', 64)
            tokenizer_path = root / 'tokenizer.json'
            tokenizer.save(str(tokenizer_path))
            original = FoundationGPT(model_config_from_depth(2, tokenizer.get_vocab_size(), 64))
            save_model(original, root / 'checkpoint')
            def export():
                return export_foundation(str(root / 'checkpoint'), str(tokenizer_path), root / 'export')

            Tokenizer(models.WordLevel({'hello': 0}, unk_token='hello')).save(str(tokenizer_path))
            with self.assertRaisesRegex(ValueError, 'missing required special tokens'):
                export()
            tokenizer.add_tokens(['extra-token'])
            tokenizer.save(str(tokenizer_path))
            with self.assertRaisesRegex(ValueError, 'vocabulary does not match'):
                export()
            tokenizer = train_tokenizer('hello world', 64)
            tokenizer.save(str(tokenizer_path))
            weights_path = root / 'checkpoint/model.safetensors'
            weights = load_file(str(weights_path))
            weights['lm_head.weight'] += 1
            save_file(weights, str(weights_path))
            with self.assertRaisesRegex(ValueError, 'tied embedding weights'):
                export()

    def test_export_preserves_logits_cached_generation_and_chat_tokens(self):
        torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenizer = train_tokenizer('hello world answer helpful assistant', 64)
            tokenizer.save(str(root / 'tokenizer.json'))
            original = FoundationGPT(model_config_from_depth(2, tokenizer.get_vocab_size(), 64)).eval()
            save_model(original, root / 'checkpoint')
            export_foundation(str(root / 'checkpoint'), str(root / 'tokenizer.json'), root / 'export')
            exported = AutoModelForCausalLM.from_pretrained(root / 'export').eval()
            ids = torch.tensor([[2, 8, 10, 4], [3, 9, 11, 5]])
            with torch.no_grad():
                expected = original(ids)
                actual = exported(ids, use_cache=True)
                torch.testing.assert_close(expected, actual.logits, atol=1e-5, rtol=1e-5)
                next_ids = actual.logits[:, -1].argmax(-1, keepdim=True)
                cached = exported(next_ids, past_key_values=actual.past_key_values)
                torch.testing.assert_close(original(torch.cat([ids, next_ids], 1))[:, -1],
                                           cached.logits[:, -1], atol=1e-5, rtol=1e-5)
            restored = AutoTokenizer.from_pretrained(root / 'export')
            prompt = restored.apply_chat_template([{'role': 'user', 'content': 'hello'}],
                                                  tokenize=False, add_generation_prompt=True)
            self.assertEqual(prompt, format_prompt('', 'hello'))
            self.assertEqual(restored.encode(prompt), tokenizer.encode(prompt).ids)
            self.assertEqual(restored.eos_token_id, tokenizer.token_to_id('<|end|>'))
            for system in ['', ' \n\t', '  Be helpful. \n']:
                with self.subTest(system=system):
                    prompt = restored.apply_chat_template([
                        {'role': 'system', 'content': system},
                        {'role': 'user', 'content': ' hello '},
                    ], tokenize=False, add_generation_prompt=True)
                    self.assertEqual(prompt, format_prompt(system, ' hello '))
                    self.assertEqual(restored.encode(prompt), tokenizer.encode(prompt).ids)


if __name__ == '__main__':
    unittest.main()
