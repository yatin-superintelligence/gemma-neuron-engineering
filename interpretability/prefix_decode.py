"""Greedy prefix-triggered native gains with isolated same-prefix shadows."""
import copy
import hashlib
import json
import math

import torch
from transformers.cache_utils import DynamicCache

from interpretability.current_group import current_group
from interpretability.decode_step import frozen_versions
from interpretability.python_prefix import PrefixGate
from interpretability.scoped_channel import bit_equal, digest


def cache_fingerprint(cache, expected_slots):
    """Reject absent, truncated or uninitialized native dynamic caches."""
    if not isinstance(cache, DynamicCache) or len(cache.layers) != expected_slots:
        raise ValueError("A complete native DynamicCache is required")
    rows = []
    for index, layer in enumerate(cache.layers):
        row = {"layer": index}
        for name in ("keys", "values"):
            value = getattr(layer, name, None)
            if (not torch.is_tensor(value) or value.ndim != 4
                    or value.shape[0] != 1 or any(size <= 0 for size in value.shape)
                    or not torch.isfinite(value).all()):
                raise ValueError("Every cache slot must hold finite batch-one K/V")
            row[name] = {"shape": list(value.shape), "dtype": str(value.dtype),
                         "sha256": digest(value)}
        rows.append(row)
    return {"layers": rows, "sha256": hashlib.sha256(json.dumps(
        rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def clone_cache(cache, expected_slots):
    """Copy ephemeral cache values exactly, with no shared tensor storage."""
    before = cache_fingerprint(cache, expected_slots)
    clone = copy.deepcopy(cache)
    if before != cache_fingerprint(clone, expected_slots):
        raise AssertionError("Cache clone changed values")
    source_pointers = {getattr(layer, name).untyped_storage().data_ptr()
                       for layer in cache.layers for name in ("keys", "values")}
    for layer in clone.layers:
        for name in ("keys", "values"):
            if getattr(layer, name).untyped_storage().data_ptr() in source_pointers:
                raise AssertionError("Cache clone aliases native storage")
    return clone


def closed_python_fence(text):
    lines = text.splitlines()
    return (bool(lines) and lines[0].strip() in {"```python", "```"}
            and any(line.strip() == "```" for line in lines[1:]))


def _hooks(model):
    return {name: (tuple(module._forward_pre_hooks), tuple(module._forward_hooks))
            for name, module in model.named_modules()}


def generate_prefix_gain(model, encoded, tokenizer, *, coordinates, gain,
                         max_new_tokens, eos_ids, expected_cache_slots,
                         stop_at_fence=True, event_cap=4):
    """Generate freely; inspect only already generated text before each step.

    The tokenizer must decode actual generated token IDs. A native shadow
    forward at each candidate uses a nonaliasing cache and never advances
    the live response history. There is no reference answer or teacher path.
    An empty coordinate sequence supplies the unmodified control.
    """
    if (type(max_new_tokens) is not int or max_new_tokens <= 0
            or not math.isfinite(gain) or gain not in (1, 2)):
        raise ValueError("Fixed gain one/two and a positive token cap required")
    ids = encoded["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
        raise ValueError("Exactly one nonempty prompt is required")
    if set(encoded) - {"input_ids", "attention_mask"}:
        raise ValueError("Only text tokens and an unpadded attention mask are supported")
    if "attention_mask" in encoded and not torch.all(encoded["attention_mask"] == 1):
        raise ValueError("This bounded decoder requires an unpadded prompt")
    versions, hooks = frozen_versions(model), _hooks(model)
    flags = {name: p.requires_grad for name, p in model.named_parameters()}
    gate = PrefixGate(event_cap)
    generated, steps, events, counted = [], [], [], []
    def meter(module, arguments, kwargs):
        value = kwargs.get("input_ids", arguments[0] if arguments else None)
        counted.append(value.numel())
    handle = model.register_forward_pre_hook(meter, with_kwargs=True)
    reason = None
    try:
        with torch.inference_mode():
            value = model(**encoded, use_cache=True, logits_to_keep=1)
            cache_fingerprint(value.past_key_values, expected_cache_slots)
            while len(generated) < max_new_tokens:
                if not torch.isfinite(value.logits).all():
                    raise ValueError("Nonfinite live logits")
                steps.append(digest(value.logits))
                next_id = int(value.logits[0, -1].argmax())
                generated.append(next_id)
                text = tokenizer.decode(generated, skip_special_tokens=True)
                if next_id in eos_ids:
                    reason = "eos"
                    break
                if stop_at_fence and closed_python_fence(text):
                    reason = "closing_fence"
                    break
                if len(generated) == max_new_tokens:
                    reason = "cap"
                    break
                event = gate.consider(text, has_cache=True)
                kwargs = {"input_ids": torch.tensor([[next_id]], device=ids.device),
                          "past_key_values": value.past_key_values, "use_cache": True,
                          "logits_to_keep": 1}
                if event is None:
                    value = model(**kwargs)
                    continue
                live_before = cache_fingerprint(value.past_key_values, expected_cache_slots)
                shadow_kwargs = {**kwargs, "past_key_values": clone_cache(
                    value.past_key_values, expected_cache_slots)}
                shadow = model(**shadow_kwargs)
                assert live_before == cache_fingerprint(value.past_key_values, expected_cache_slots)
                if coordinates:
                    with current_group(model, coordinates, "gain", gain=gain) as edit:
                        value = model(**kwargs)
                    # Retain exact scope hashes, not raw activation samples.
                    edit = {**edit, "scope": [{key: item for key, item in row.items()
                             if key not in {"target_before_values", "target_after_values"}}
                             for row in edit["scope"]]}
                else:
                    value = model(**kwargs)
                    edit = None
                actual_cache = cache_fingerprint(value.past_key_values, expected_cache_slots)
                shadow_cache = cache_fingerprint(shadow.past_key_values, expected_cache_slots)
                assert actual_cache == shadow_cache
                exact = bit_equal(value.logits, shadow.logits)
                if gain == 1 or not coordinates:
                    assert exact
                events.append({**event, "generated_prefix_tokens": len(generated),
                               "native_logits_sha256": digest(shadow.logits),
                               "edited_logits_sha256": digest(value.logits),
                               "logits_exact": exact, "whole_current_cache_exact": True,
                               "current_cache_sha256": actual_cache["sha256"], "edit": edit})
                del shadow
    finally:
        handle.remove()
        assert versions == frozen_versions(model)
        assert flags == {name: p.requires_grad for name, p in model.named_parameters()}
        assert hooks == _hooks(model)
    assert len(counted) == len(generated) + len(events)
    assert sum(counted) == ids.numel() + len(generated) - 1 + len(events)
    return {"generated_ids": generated, "text": text, "stop_reason": reason,
            "step_logit_sha256": steps, "events": events,
            "actual_forward_calls": len(counted), "input_token_exposures": sum(counted),
            "original_parameter_versions_unchanged": True,
            "gradient_flags_unchanged": True, "all_parameter_gradients_none": True,
            "hooks_restored": True, "backward_calls": 0, "optimizer_steps": 0}
