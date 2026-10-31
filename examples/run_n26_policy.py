"""Reconstruct a reviewed global policy for a caller-supplied prompt."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer

from interpretability.prefix_decode import generate_prefix_gain
from interpretability.prescribed_columns import prescribed_columns, tensor_sha
from reproduce.model_edits import load_n02_base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', type=Path, default=Path('models/n26/recipe.json'))
    parser.add_argument('--base-bundle', type=Path, default=Path('models/n02/bf16_text'))
    parser.add_argument('--condition', default='selected64_gain0')
    parser.add_argument('--prompt-file', type=Path, required=True)
    parser.add_argument('--format', choices=['python', 'json'], default='python')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    recipe = json.loads(args.recipe.read_text())
    assert transformers.__version__ == '5.8.0'
    for name, expected in recipe['source_sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    condition = next(r for r in recipe['conditions'] if r['name'] == args.condition)
    torch.manual_seed(recipe['seed'])
    model, _ = load_n02_base(args.base_bundle)
    assert sum(p.numel() for p in model.parameters()) == 4628569344
    assert all(p.dtype == torch.bfloat16 and not p.requires_grad and p.grad is None for p in model.parameters())
    for layer, expected in recipe['source_down_matrix_sha256'].items():
        assert tensor_sha(model.model.language_model.layers[int(layer)].mlp.down_proj.weight) == expected
    tokenizer = AutoTokenizer.from_pretrained(recipe['parent'], revision=recipe['revision'], trust_remote_code=False)
    text = tokenizer.apply_chat_template([{'role': 'user', 'content': args.prompt_file.read_text()}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
    encoded = tokenizer(text, add_special_tokens=False, return_tensors='pt').to(next(model.parameters()).device)
    scope = (nullcontext(None) if condition['group'] is None else prescribed_columns(
             model, recipe['groups'][condition['group']], condition['factor']))
    with scope as proof:
        if proof:
            for row in proof['layers']:
                assert row['edited_matrix_sha256'] == recipe['expected_down_matrix_sha256'][condition['name']][str(row['layer'])]
        result = generate_prefix_gain(model, encoded, tokenizer, coordinates=[], gain=1,
            max_new_tokens=recipe['generation']['json_cap' if args.format == 'json' else 'coding_cap'],
            eos_ids=set(recipe['generation']['eos_ids']), expected_cache_slots=15, stop_at_fence=True)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
