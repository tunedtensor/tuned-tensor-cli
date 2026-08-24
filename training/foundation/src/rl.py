from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.nn import functional as F

from common import load_config, require_cuda, write_json
from data import END, encode_ids, example_pairs, format_prompt, numeric_reward
from model import FoundationGPT, load_model, param_count, save_model


def bounded_rollout(
    prompt_ids: list[int],
    context_length: int,
    requested_completion_tokens: int = 24,
) -> tuple[list[int], int]:
    """Fit the newest prompt tokens and a useful completion inside one context."""
    if context_length < 2:
        raise ValueError("RL requires a model context of at least two tokens.")
    completion_tokens = min(requested_completion_tokens, context_length // 2)
    prompt_tokens = context_length - completion_tokens
    return (prompt_ids or [0])[-prompt_tokens:], completion_tokens


def policy_gradient_loss(token_log_probs: torch.Tensor, reward: float) -> torch.Tensor:
    """Use a fixed binary-reward baseline so failed samples are discouraged."""
    advantage = reward - 0.5
    return -(advantage * token_log_probs.mean())


@torch.no_grad()
def sample_rollout(
    model: FoundationGPT,
    prompt: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    stop_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one on-policy completion, bounded by the model context."""
    if temperature <= 0:
        raise ValueError("RL sampling temperature must be positive.")
    model.eval()
    tokens = prompt
    limit = int(model.config["sequence_length"])
    for _ in range(max_new_tokens):
        logits = model(tokens[:, -limit:])[:, -1, :] / temperature
        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1, generator=generator)
        tokens = torch.cat([tokens, next_token], dim=1)
        if stop_token_id is not None and bool(torch.all(next_token == stop_token_id)):
            break
    return tokens


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
    generator = torch.Generator(device=device)
    generator.manual_seed(int(config.get("seed") or 0))
    steps = max(1, int(config.get("steps") or 1))
    last_reward = 0.0
    last_loss = 0.0
    for step in range(steps):
        example = examples[step % len(examples)]
        prompt_ids = encode_ids(tokenizer, format_prompt(str(config.get("system_prompt") or ""), example["input"]))
        prompt_ids, completion_tokens = bounded_rollout(
            prompt_ids,
            int(model.config["sequence_length"]),
        )
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        sampled = sample_rollout(
            model,
            prompt,
            max_new_tokens=completion_tokens,
            stop_token_id=tokenizer.token_to_id(END),
            generator=generator,
        )
        completion_ids = sampled[0, len(prompt_ids):].tolist()
        text = tokenizer.decode(completion_ids).split(END)[0]
        reward = numeric_reward(text, example["output"])
        model.train()
        logits = model(sampled[:, :-1])
        completion = sampled[:, len(prompt_ids):]
        if completion.size(1) == 0:
            last_reward = reward
            continue
        step_logits = logits[:, len(prompt_ids) - 1: len(prompt_ids) - 1 + completion.size(1)]
        log_probs = F.log_softmax(step_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, completion.unsqueeze(-1)).squeeze(-1)
        loss = policy_gradient_loss(token_log_probs, reward)
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
