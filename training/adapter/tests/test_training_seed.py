"""Exercise training setup without requiring CUDA or importing the ML stack."""
from __future__ import annotations

import ast
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock


class TrainingSeedTests(unittest.TestCase):
    def exercise(self, seed):
        # Execute the production function with boundary doubles. In particular,
        # Trainer reseeding must not conceal unseeded model/adapter initialization.
        tree = ast.parse((Path(__file__).resolve().parents[1] / "src/train.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_training")
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
        events = []
        weights = []
        arguments = {}
        model = Mock()
        model.config = SimpleNamespace(use_cache=True)
        model.peft_config = {"default": SimpleNamespace()}
        model.parameters.return_value = []
        tokenizer = SimpleNamespace(pad_token_id=0, save_pretrained=Mock())

        def set_seed(value):
            events.append("seed")
            random.seed(value)

        def load_model(_):
            events.append("load")
            weights.append(random.random())
            return model, tokenizer, "bf16"

        def lora(model):
            events.append("lora")
            weights.append(random.random())
            return model

        def training_args(**kwargs):
            arguments.update(kwargs)
            return kwargs

        def trainer(**kwargs):
            random.seed(kwargs["args"]["seed"])
            return SimpleNamespace(train=lambda: SimpleNamespace(metrics={"train_loss": 1}))

        scope = dict(
            hp_int=lambda name, default: seed if name == "seed" else default,
            hp=lambda name, default=None: default,
            hp_float=lambda name, default: default,
            set_seed=set_seed,
            torch=SimpleNamespace(bfloat16="bf16", float16="fp16", cuda=SimpleNamespace(
                reset_peak_memory_stats=Mock(), max_memory_allocated=lambda: 100, max_memory_reserved=lambda: 200)),
            create_model_and_tokenizer=load_model, apply_lora=lora,
            AssistantOnlyDataset=Mock(), AssistantOnlyCollator=Mock(),
            chat_template_kwargs=lambda _: {}, CERTIFIED_BASE_MODEL="fixture/model",
            TemporaryDirectory=TemporaryDirectory, TrainingArguments=training_args,
            Trainer=trainer, MODEL_DIR="unused",
        )
        exec(compile(ast.fix_missing_locations(module), "train.py", "exec"), scope)
        metrics, _ = scope["run_training"]([], "fixture/model")
        self.assertEqual(events, ["seed", "load", "lora"])
        self.assertEqual(arguments["seed"], seed)
        self.assertEqual(arguments["data_seed"], seed)
        self.assertEqual(metrics["seed"], seed)
        self.assertEqual(metrics["peak_cuda_allocated_bytes"], 100)
        return weights

    def test_seed_controls_initialization_and_sampling_including_zero(self):
        original = random.getstate()
        try:
            self.assertEqual(self.exercise(0), self.exercise(0))
            self.assertNotEqual(self.exercise(0), self.exercise(42))
        finally:
            random.setstate(original)

    def test_direct_python_entry_rejects_invalid_seed(self):
        for seed in [-1, 4294967296]:
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "Training seed"):
                self.exercise(seed)
