"""Reconstruct one reviewed runtime policy using a caller-supplied prompt."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from interpretability.prefix_decode import generate_prefix_gain
from reproduce.model_edits import load_n02_base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, default=Path("models/n25/recipe.json"))
    parser.add_argument("--base-bundle", type=Path, default=Path("models/n02/bf16_text"))
    parser.add_argument("--condition", default="selected64_gain2")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--format", choices=["python", "json"], default="python")
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    root = Path(__file__).resolve().parents[1]
    for name, expected in recipe["source_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    policy = next(row for row in recipe["conditions"] if row["name"] == args.condition)
    torch.manual_seed(recipe["seed"])
    model, _ = load_n02_base(args.base_bundle)
    assert sum(p.numel() for p in model.parameters()) == recipe["base"]["unique_parameters"]
    assert all(p.dtype == torch.bfloat16 and not p.requires_grad and p.grad is None for p in model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(recipe["base"]["repository"], revision=recipe["base"]["revision"],
                                              trust_remote_code=False)
    text = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt_file.read_text()}],
              tokenize=False, add_generation_prompt=True, enable_thinking=False)
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt").to(next(model.parameters()).device)
    coordinates = [] if policy["group"] is None else recipe["groups"][policy["group"]]
    result = generate_prefix_gain(model, encoded, tokenizer, coordinates=coordinates, gain=policy["gain"],
        max_new_tokens=recipe["generation"]["json_cap" if args.format == "json" else "coding_cap"],
        eos_ids=set(recipe["generation"]["eos_ids"]), expected_cache_slots=15, stop_at_fence=True,
        event_cap=recipe["controller"]["event_cap"])
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
