"""Clone an actual native past for one current-token causal measurement.

The clone is ephemeral. It is not a reusable past for a changed conversation.
Only the pinned DynamicCache with initialized dense KV layers is accepted.
"""

import copy
import hashlib
import json
from contextlib import contextmanager

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer, DynamicSlidingWindowLayer


def tensor_hash(value):
    return hashlib.sha256(value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def _value(value):
    if torch.is_tensor(value):
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'device': str(value.device), 'sha256': tensor_hash(value)}
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    raise ValueError('Unsupported mutable cache metadata')


def fingerprint(cache, entries, length, dtype):
    """Validate logical and physical layout and retain exact compact witnesses."""
    if type(cache) is not DynamicCache or cache.offloading or cache.layer_class_to_replicate is not None:
        raise ValueError('Initialized non-offloaded DynamicCache required')
    if type(entries) is not int or type(length) is not int or min(entries, length) < 1 or len(cache.layers) != entries:
        raise ValueError('Exact initialized cache cardinality and length required')
    if set(cache.__dict__) != {'layers', 'layer_class_to_replicate', 'offloading'}:
        raise ValueError('Unexpected cache implementation state')
    rows = []
    for index, layer in enumerate(cache.layers):
        if type(layer) not in (DynamicLayer, DynamicSlidingWindowLayer) or not layer.is_initialized:
            raise ValueError('Actual initialized native layer required')
        if int(layer.get_seq_length()) != length:
            raise ValueError('Logical cache length differs from declared native past')
        k, v = layer.keys, layer.values
        if not torch.is_tensor(k) or not torch.is_tensor(v) or k.shape != v.shape or k.ndim != 4 or k.shape[0] != 1:
            raise ValueError('Dense batch-one aligned key/value tensors required')
        physical = min(length, layer.sliding_window - 1) if type(layer) is DynamicSlidingWindowLayer else length
        if k.shape[-2] != physical or min(k.shape) < 1 or k.dtype != dtype or v.dtype != dtype or k.device != v.device:
            raise ValueError('Truncated, mismatched or wrong-dtype native cache')
        if not torch.isfinite(k).all() or not torch.isfinite(v).all():
            raise ValueError('Nonfinite native cache')
        rows.append({'layer': index, 'type': type(layer).__name__, 'attributes': {key: _value(value) for key, value in sorted(layer.__dict__.items())}})
    raw = json.dumps(rows, sort_keys=True, separators=(',', ':'))
    return {'entries': entries, 'logical_length': length, 'layers': rows, 'sha256': hashlib.sha256(raw.encode()).hexdigest()}


def _storages(cache):
    return {value.untyped_storage().data_ptr() for layer in cache.layers for value in layer.__dict__.values() if torch.is_tensor(value) and value.numel()}


@contextmanager
def native_cache_branch(cache, entries, length, dtype):
    """Yield a fresh nonaliasing native past and require one-token consumption.

    The source must stay byte-exact, including sliding-window counters. A failed
    probe still checks source immutability. Full raw pasts are never serialized.
    """
    before = fingerprint(cache, entries, length, dtype)
    branch = copy.deepcopy(cache)
    if fingerprint(branch, entries, length, dtype) != before or _storages(cache) & _storages(branch):
        raise ValueError('Cache copy is inexact or aliases native storage')
    success = False
    try:
        yield branch
        success = True
    finally:
        if fingerprint(cache, entries, length, dtype) != before:
            raise AssertionError('Native source past changed during measurement')
        if success:
            fingerprint(branch, entries, length + 1, dtype)
