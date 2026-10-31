"""One declared current-position edit, including the last retention prefill token."""
from contextlib import contextmanager
import hashlib

import torch

from interpretability.cache_proofs import check_cache
from interpretability.decode_step import frozen_versions
from interpretability.scoped_channel import bit_equal, digest
from research.experiments.n19.trajectory import cache_witness, current, prepare


def float_sha(value):
    return hashlib.sha256(value.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()


def validate_cache(cache, dtype):
    if len(cache.layers) != 15:
        raise ValueError('Fifteen populated native cache entries required')
    for layer in cache.layers:
        keys, values = getattr(layer, 'keys', None), getattr(layer, 'values', None)
        if not torch.is_tensor(keys) or not torch.is_tensor(values):
            raise ValueError('Native keys and values required')
        if keys.shape != values.shape or keys.ndim != 4 or keys.shape[0] != 1:
            raise ValueError('One matching key/value trajectory required')
        if any(size <= 0 for size in keys.shape):
            raise ValueError('Nonempty cache tensors required')
        if str(keys.dtype) != dtype or str(values.dtype) != dtype:
            raise ValueError('Native cache dtype required')
        if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
            raise ValueError('Finite cache values required')


def witness(value, dtype):
    validate_cache(value.past_key_values, dtype)
    if not torch.isfinite(value.logits).all():
        raise ValueError('Finite current logits required')
    cache = cache_witness(value.past_key_values)
    check_cache(cache, 15, dtype=dtype)
    return {'full_logits_sha256': float_sha(value.logits[0, -1]),
            'top1_token_id': int(value.logits[0, -1].argmax()), 'cache': cache}


def prepare_context(model, encoded, prefix):
    cache = prepare(model, encoded, prefix) if prefix else None
    if prefix:
        validate_cache(cache, str(next(model.parameters()).dtype))
    return cache


def run_current(model, encoded, prefix, cache):
    if prefix:
        if cache is None:
            raise ValueError('A fresh prepared cache is required')
        validate_cache(cache, str(next(model.parameters()).dtype))
        return current(model, encoded, prefix, cache)
    if cache is not None:
        raise ValueError('Initial prefill must start without a prepared cache')
    return model(**encoded, use_cache=True, logits_to_keep=1)


@contextmanager
def group_policy(model, coordinates, operation, donor, sequence_length):
    """Assign only the final position of one explicitly sized current forward."""
    if operation not in ['swap', 'zero', 'rescue'] or type(sequence_length) is not int or sequence_length < 1:
        raise ValueError('Declared operation and current input length required')
    layers = model.model.language_model.layers
    widths = [layer.mlp.down_proj.in_features for layer in layers]
    producers = model.config.text_config.num_hidden_layers - model.config.text_config.num_kv_shared_layers
    offsets = [sum(widths[:i]) for i in range(len(widths))]
    grouped, seen = {}, set()
    for coordinate in coordinates:
        layer, channel, flat = coordinate['layer'], coordinate['channel'], coordinate['flat']
        if type(layer) is not int or type(channel) is not int or type(flat) is not int:
            raise ValueError('Integer native coordinates required')
        if not producers <= layer < len(layers) or not 0 <= channel < widths[layer]:
            raise ValueError('A native late-producer channel is required')
        if flat != offsets[layer] + channel or (layer, channel) in seen:
            raise ValueError('Unique source-corresponding flat coordinates required')
        seen.add((layer, channel))
        grouped.setdefault(layer, []).append(channel)
    if not seen:
        raise ValueError('Nonempty group required')
    versions = frozen_versions(model)
    receipt = {'operation': operation, 'current_sequence_length': sequence_length,
               'position': sequence_length - 1, 'coordinates': len(seen),
               'hits': {layer: 0 for layer in grouped}, 'scope': [], 'optimizer_steps': 0}
    handles, success = [], False
    try:
        for layer, channels in sorted(grouped.items()):
            if operation != 'zero':
                if donor is None or layer not in donor['MLP']:
                    raise ValueError('Actual unmodified donor state required')
                state = donor['MLP'][layer]
                if state.shape != (1, widths[layer]) or not torch.isfinite(state).all():
                    raise ValueError('Native finite donor shape required')

            def change(module, arguments, layer=layer, channels=channels):
                before = arguments[0]
                if before.ndim != 3 or before.shape[:2] != (1, sequence_length):
                    raise ValueError('Current forward shape differs from its declared path')
                if receipt['hits'][layer]:
                    raise ValueError('Only one current forward allowed per edit')
                receipt['hits'][layer] += 1
                prior = before[:, -1, channels].detach().clone()
                if operation == 'zero':
                    expected = torch.zeros_like(prior)
                else:
                    state = donor['MLP'][layer]
                    if state.dtype != before.dtype:
                        raise ValueError('Actual donor dtype must match native current dtype')
                    expected = state[:, channels].to(before.device)
                if not torch.isfinite(prior).all() or not torch.isfinite(expected).all():
                    raise ValueError('Finite native targets required')
                after = before.clone()
                if operation == 'rescue':
                    after[:, -1, channels] = 0
                after[:, -1, channels] = expected
                outside = after.clone()
                outside[:, -1, channels] = prior
                assert bit_equal(outside, before)
                assert bit_equal(after[:, -1, channels], expected)
                receipt['scope'].append({'layer': layer, 'channels': channels,
                    'before_target_sha256': digest(prior),
                    'after_target_sha256': digest(after[:, -1, channels]),
                    'expected_target_sha256': digest(expected),
                    'earlier_positions_and_other_channels_bit_exact': True})
                return (after, *arguments[1:])

            handles.append(layers[layer].mlp.down_proj.register_forward_pre_hook(change))
        yield receipt
        success = True
    finally:
        for handle in handles:
            handle.remove()
        assert versions == {name: value._version for name, value in model.named_parameters()}
        assert all(not value.requires_grad and value.grad is None for value in model.parameters())
        if success:
            assert all(value == 1 for value in receipt['hits'].values())
        receipt['native_parameter_versions_and_gradflags_unchanged'] = True


@contextmanager
def whole_policy(model, donor, sequence_length):
    if type(sequence_length) is not int or sequence_length < 1:
        raise ValueError('Explicit current sequence length required')
    layers = model.model.language_model.layers
    if set(donor['MLP']) != set(range(len(layers))):
        raise ValueError('Every actual native layer donor required')
    versions = frozen_versions(model)
    receipt = {'operation': 'whole_actual_donor', 'position': sequence_length - 1,
               'current_sequence_length': sequence_length,
               'hits': {index: 0 for index in range(len(layers))},
               'scope': [], 'optimizer_steps': 0}
    handles, success = [], False
    try:
        for index, layer in enumerate(layers):
            state = donor['MLP'][index]
            if state.shape != (1, layer.mlp.down_proj.in_features) or not torch.isfinite(state).all():
                raise ValueError('Finite actual full native donor shape required')

            def change(module, arguments, index=index):
                before = arguments[0]
                if before.ndim != 3 or before.shape[:2] != (1, sequence_length):
                    raise ValueError('Declared whole current forward shape required')
                if receipt['hits'][index]:
                    raise ValueError('One whole current forward per scope required')
                receipt['hits'][index] += 1
                state = donor['MLP'][index]
                if state.dtype != before.dtype:
                    raise ValueError('Actual full donor dtype must match native dtype')
                expected = state.to(before.device)
                after = before.clone()
                after[:, -1] = expected
                outside = after.clone()
                outside[:, -1] = before[:, -1]
                assert bit_equal(outside, before)
                assert bit_equal(after[:, -1], expected)
                receipt['scope'].append({'layer': index,
                    'before_target_sha256': digest(before[:, -1]),
                    'after_target_sha256': digest(after[:, -1]),
                    'expected_target_sha256': digest(expected),
                    'earlier_positions_bit_exact': True})
                return (after, *arguments[1:])

            handles.append(layer.mlp.down_proj.register_forward_pre_hook(change))
        yield receipt
        success = True
    finally:
        for handle in handles:
            handle.remove()
        assert versions == {name: value._version for name, value in model.named_parameters()}
        assert all(not value.requires_grad and value.grad is None for value in model.parameters())
        if success:
            assert all(value == 1 for value in receipt['hits'].values())
        receipt['native_parameter_versions_and_gradflags_unchanged'] = True
