"""Verify a capped native trajectory, then continue its own history freely."""

import torch
from transformers.cache_utils import DynamicCache
from interpretability.cache_proofs import check_cache
from research.experiments.n19.trajectory import cache_witness
from research.experiments.n35.core import stable, tensor_sha


def decision(output, step, layers, dtype):
    logits = output.logits[0, -1].detach()
    assert torch.isfinite(logits).all() and str(logits.dtype) == dtype
    cache = cache_witness(output.past_key_values)
    check_cache(cache, layers, dtype)
    assert all(layer['keys'] is not None for layer in cache['layers'])
    return {'step': step, 'argmax': int(logits.argmax()), 'full_logits_sha256': tensor_sha(logits.float()), 'cache_sha256': cache['sha256'], 'raw_logits_dtype': dtype}, cache


def extend(model, tokenizer, encoded, parent, total_cap=2048, eos=frozenset({1, 106, 50})):
    """Parent IDs are verified native history, never a donor or hidden answer."""
    ids = parent['generated_ids']
    assert parent['termination'] == 'cap' and len(ids) == parent['cap'] < total_cap
    assert len(ids) == len(parent['decision_trace']) and not any(t in eos for t in ids)
    config = model.config.text_config
    n = int(encoded['input_ids'].numel())
    assert n + total_cap <= config.max_position_embeddings
    slots = config.num_hidden_layers - config.num_kv_shared_layers
    dtype = str(next(model.parameters()).dtype)
    cache = DynamicCache(config=config)
    trace = []
    first_cache = None
    with torch.no_grad():
        output = model(**encoded, past_key_values=cache, use_cache=True, logits_to_keep=1)
        for i, expected in enumerate(ids):
            item, current_cache = decision(output, i, slots, dtype)
            assert item == parent['decision_trace'][i], 'Native parent decision changed before new suffix'
            assert item['argmax'] == expected
            if i == 0:
                first_cache = current_cache
                assert first_cache == parent['first_cache']
            trace.append(item)
            if i + 1 < len(ids):
                output = model(input_ids=torch.tensor([[expected]], device=encoded['input_ids'].device), past_key_values=output.past_key_values, use_cache=True, logits_to_keep=1)
        # The final saved choice had been emitted but not consumed at the old cap.
        previous = ids[-1]
        suffix = []
        while len(ids) + len(suffix) < total_cap:
            output = model(input_ids=torch.tensor([[previous]], device=encoded['input_ids'].device), past_key_values=output.past_key_values, use_cache=True, logits_to_keep=1)
            item, _ = decision(output, len(ids) + len(suffix), slots, dtype)
            previous = item['argmax']
            suffix.append(previous)
            trace.append(item)
            if previous in eos:
                break
    full = list(ids) + suffix
    assert suffix and len(full) <= total_cap
    return {'generated_ids': full, 'text': tokenizer.decode(full, skip_special_tokens=True), 'generated_tokens': len(full), 'new_generated_tokens': len(suffix), 'teacher_replayed_reference_tokens': len(ids), 'prefix_ids_sha256': stable(ids), 'prefix_exact': True, 'cap': total_cap, 'at_cap': len(full) == total_cap, 'termination': 'eos' if full[-1] in eos else 'cap', 'input_tokens': n, 'decision_trace': trace, 'first_cache': first_cache, 'expected_counter': {'calls': len(full), 'input_token_exposures': n + len(full) - 1}}
