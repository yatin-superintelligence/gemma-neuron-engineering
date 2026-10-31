"""Pinned Gemma current-unit helpers; function bodies match the frozen runner."""
import hashlib
import torch
from interpretability.cache_proofs import check_cache
from research.experiments.n19.trajectory import cache_witness
from research.experiments.n20.operations import policy

def sha_tensor(value):
    return hashlib.sha256(value.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()

def validate_live_cache(cache, dtype='torch.bfloat16'):
    if len(cache.layers) != 15:
        raise ValueError('Exactly fifteen populated native cache entries required')
    for layer in cache.layers:
        keys, values = getattr(layer, 'keys', None), getattr(layer, 'values', None)
        if not torch.is_tensor(keys) or not torch.is_tensor(values):
            raise ValueError('Both native cache tensors required')
        if keys.shape != values.shape or keys.ndim != 4 or keys.shape[0] != 1:
            raise ValueError('A single native key/value trajectory is required')
        if any(size <= 0 for size in keys.shape):
            raise ValueError('Empty native cache tensor')
        if str(keys.dtype) != dtype or str(values.dtype) != dtype:
            raise ValueError('Native cache precision mismatch')
        if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
            raise ValueError('Nonfinite native cache tensor')

def witness(value, dtype='torch.bfloat16'):
    validate_live_cache(value.past_key_values, dtype)
    assert torch.isfinite(value.logits).all()
    cache = cache_witness(value.past_key_values)
    check_cache(cache, 15, dtype=dtype)
    return {'full_logits_sha256': sha_tensor(value.logits[0, -1]),
            'top1_token_id': int(value.logits[0, -1].argmax()), 'cache': cache}

def unit_policy(model, operation, group, groups, own, opposite, same, full_coordinates):
    coordinates = groups[group] if group is not None else None
    if group is not None:
        if len(coordinates) != 1:
            raise ValueError('Exactly one native scalar is declared')
        c = coordinates[0]
        widths = [layer.mlp.down_proj.in_features for layer in model.model.language_model.layers]
        if c['layer'] != len(widths) - 1 or not 0 <= c['channel'] < widths[-1]:
            raise ValueError('A valid final-layer native scalar is required')
        if c['flat'] != sum(widths[:-1]) + c['channel']:
            raise ValueError('Flat coordinate must match its native source column')
    return policy(model, operation, coordinates, own, opposite, same, full_coordinates)
