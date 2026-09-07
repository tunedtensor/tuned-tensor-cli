"""Export TT FoundationGPT weights to vLLM's existing GPT-2 implementation."""
from __future__ import annotations

import json
from pathlib import Path

CHAT_TEMPLATE = """{%- if not messages or messages[0]['role'] != 'system' -%}
{{- '<|system|>You are a helpful assistant.<|end|>' -}}
{%- endif -%}
{%- for message in messages -%}
{%- set content = message['content'] -%}
{%- if message['role'] == 'system' -%}
{%- set content = (content | trim) or 'You are a helpful assistant.' -%}
{%- endif -%}
{%- if message['role'] not in ['system', 'user', 'assistant'] -%}
{{- raise_exception('Foundation chat supports system, user and assistant messages only.') -}}
{%- endif -%}
{{- '<|' + message['role'] + '|>' + content + '<|end|>' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|assistant|>' -}}{%- endif -%}"""


def export_foundation(checkpoint: str, tokenizer_path: str, destination: Path) -> str:
    import torch
    from safetensors.torch import load_file, save_file
    from tokenizers import Tokenizer
    from transformers import GPT2Config, PreTrainedTokenizerFast

    source = Path(checkpoint)
    config = json.loads((source / 'config.json').read_text())
    for key in ('depth', 'width', 'heads', 'vocab_size', 'sequence_length'):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f'Invalid foundation config: {key}')
    if config['width'] % config['heads']:
        raise ValueError('Foundation width must be divisible by heads.')
    raw_tokenizer = Tokenizer.from_file(tokenizer_path)
    special_tokens = ['<unk>', '<pad>', '<|system|>', '<|user|>', '<|assistant|>', '<|end|>']
    if any(raw_tokenizer.token_to_id(token) is None for token in special_tokens):
        raise ValueError('Foundation tokenizer is missing required special tokens.')
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer,
        unk_token='<unk>', pad_token='<pad>', eos_token='<|end|>',
        additional_special_tokens=['<|system|>', '<|user|>', '<|assistant|>'],
        model_max_length=config['sequence_length'], chat_template=CHAT_TEMPLATE)
    if len(tokenizer) != config['vocab_size']:
        raise ValueError('Foundation tokenizer vocabulary does not match the checkpoint.')
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(destination)
    GPT2Config(n_layer=config['depth'], n_embd=config['width'], n_head=config['heads'],
        n_positions=config['sequence_length'], vocab_size=config['vocab_size'],
        activation_function='gelu', layer_norm_epsilon=1e-5,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        bos_token_id=None, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, architectures=['GPT2LMHeadModel'],
        tie_word_embeddings=True).save_pretrained(destination)
    weights = load_file(str(source / 'model.safetensors'))
    exported = {}
    def move(old, new, transpose=False):
        tensor = weights.pop(old)
        exported[new] = tensor.t().contiguous() if transpose else tensor
    move('tok_emb.weight', 'transformer.wte.weight')
    move('pos_emb.weight', 'transformer.wpe.weight')
    head = weights.pop('lm_head.weight')
    if not torch.equal(head, exported['transformer.wte.weight']):
        raise ValueError('Foundation checkpoint must have tied embedding weights.')
    for suffix in ('weight', 'bias'):
        move('ln_f.' + suffix, 'transformer.ln_f.' + suffix)
    for i in range(config['depth']):
        old, new = f'blocks.{i}.', f'transformer.h.{i}.'
        for src, dst in [('ln1', 'ln_1'), ('ln2', 'ln_2'),
                         ('mlp.0', 'mlp.c_fc'), ('mlp.2', 'mlp.c_proj')]:
            for suffix in ('weight', 'bias'):
                move(old + src + '.' + suffix, new + dst + '.' + suffix,
                     transpose=src.startswith('mlp') and suffix == 'weight')
        for src, dst in [('attn.qkv', 'attn.c_attn'), ('attn.proj', 'attn.c_proj')]:
            move(old + src + '.weight', new + dst + '.weight', transpose=True)
            weight = exported[new + dst + '.weight']
            exported[new + dst + '.bias'] = torch.zeros(weight.shape[1], dtype=weight.dtype)
    if weights:
        raise ValueError(f'Unexpected foundation weights: {sorted(weights)}')
    save_file(exported, str(destination / 'model.safetensors'), metadata={'format': 'pt'})
    return str(destination)
