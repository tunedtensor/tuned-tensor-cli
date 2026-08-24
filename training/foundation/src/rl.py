from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.nn import functional as F

from common import load_config, require_cuda, write_json
from data import END, encode_ids, example_pairs, format_prompt, numeric_reward
from model import generate, load_model, param_count, save_model


def main(argv: list[str] | None = None) -> None:
    config = load_config(argv)
    require_cuda("rl")
    device = torch.device("cuda")
    tokenizer = Tokenizer.from_file(str(Path(config["tokenizer_dir"]) / "tokenizer.json"))
    model = load_model(config["model_dir"], map_location=device)
    examples = example_pairs(config.get("examples") or [])
    if not examples:
        raise SystemExit("Foundation RL needs chat examples with verifiable answers.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    steps = max(1, int(config.get("steps") or 1))
    last_reward = 0.0
    last_loss = 0.0
    for step in range(steps):
        example = examples[step % len(examples)]
        prompt_ids = encode_ids(tokenizer, format_prompt(str(config.get("system_prompt") or ""), example["input"])) or [0]
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        sampled = generate(model, prompt, max_new_tokens=24)
        completion_ids = sampled[0, len(prompt_ids):].tolist()
        text = tokenizer.decode(completion_ids).split(END)[0]
        reward = numeric_reward(text, example["output"])
        logits = model(sampled[:, :-1])
        completion = sampled[:, len(prompt_ids):]
        if completion.size(1) == 0:
            last_reward = reward
            continue
        step_logits = logits[:, len(prompt_ids) - 1: len(prompt_ids) - 1 + completion.size(1)]
        log_probs = F.log_softmax(step_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, completion.unsqueeze(-1)).squeeze(-1)
        loss = -(reward * token_log_probs.mean())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_reward = reward
        last_loss = float(loss.item())

    output_dir = Path(config["output_dir"])
    save_model(model, output_dir)
    write_json(output_dir / "metrics.json", {
        "ok": True,
        "steps": steps,
        "reward": last_reward,
        "loss": last_loss,
        "parameters": param_count(model),
        "device": str(device),
    })


if __name__ == "__main__":
    main()
