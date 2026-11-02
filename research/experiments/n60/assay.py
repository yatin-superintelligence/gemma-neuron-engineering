"""Fixed BOS/current-query value-edge surgery; no parameter or cache writes."""
from contextlib import contextmanager
import hashlib
import inspect
import random

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.gemma4 import modeling_gemma4 as native_module

from research.experiments.n35.core import snapshot, tensor_sha
from research.experiments.n59.assay import native_architecture, forward_record, source_check

CONDITIONS = ('remove_BOS_value_link', 'remove_one_nonBOS_value_link')
NATIVE_EAGER = native_module.eager_attention_forward
EAGER_SHA = '7f8506e7461c8fb3a702c2c83535c1cd8e4381ce029ccc8a29bc4e0a37fb293e'
OPERATIONS = ('native_sourceguards', 'native_restoration_forwards',
    'new_first_argmax_observations', 'source_attention_observations',
    'edited_attention_callbacks', 'native_value_BMM_replay_attempts',
    'edited_value_BMM_attempts', 'scratch_V_clone_attempts',
    'scratch_V_zero_assignments', 'current_head_output_assignments',
    'zero_assigned_elements', 'zero_assigned_bytes',
    'current_output_assigned_elements', 'current_output_assigned_bytes',
    'global_function_reference_assignments')


def record(t):
    if not isinstance(t, torch.Tensor) or t.requires_grad or not torch.isfinite(t).all():
        raise ValueError('Actual finite detached tensor required')
    return {'sha256': tensor_sha(t), 'shape': list(t.shape), 'dtype': str(t.dtype),
            'numel': t.numel(), 'bytes': t.numel()*t.element_size()}


def check_native(model):
    text = native_architecture(model)
    if (native_module.eager_attention_forward is not NATIVE_EAGER
            or hashlib.sha256(inspect.getsource(NATIVE_EAGER).rstrip('\n').encode()).hexdigest() != EAGER_SHA
            or 'eager' in native_module.ALL_ATTENTION_FUNCTIONS
            or any(b.self_attn.config._attn_implementation != 'eager' for b in text.layers)):
        raise ValueError('Exact unshadowed native eager fallback required')
    return text


def function_witness(model):
    check_native(model)
    return {'native_eager_source_sha256': EAGER_SHA,
            'actual_global_is_original_reference': True, 'eager_registry_shadow_absent': True}


def control_position(length, row_index):
    if type(length) is not int or length < 3 or type(row_index) is not int or row_index < 0:
        raise ValueError('Actual nontrivial context and nonnegative case index required')
    return random.Random(6001+row_index).choice(range(1, length-1))


def noncurrent_cache_prefix(cache):
    rows = []
    for i, layer in enumerate(cache.layers):
        if not layer.is_initialized or layer.keys.shape[-2] < 2:
            raise ValueError('Actual initialized native noncurrent KV prefix required')
        rows.append({'layer': i, 'keys': record(layer.keys[..., :-1, :]),
                     'values': record(layer.values[..., :-1, :])})
    if len(rows) != 15:
        raise ValueError('All15 original physical KV entries required')
    return rows


@contextmanager
def value_operator(model, length, control, count, condition=None):
    text = check_native(model)
    if condition is not None and condition not in CONDITIONS:
        raise ValueError('Registered fixed value policy required')
    if not 0 < control < length-1:
        raise ValueError('Predetermined nonBOS/noncurrent control required')
    observations = []
    selected = None if condition is None else (0 if condition == CONDITIONS[0] else control)
    modules = [b.self_attn for b in text.layers]

    def wrapper(module, query, key, value, attention_mask, dropout=0.0,
                scaling=None, softcap=None, **kwargs):
        index = len(observations)
        if (index >= 35 or module is not modules[index] or dropout != 0.0 or module.training
                or query.shape[0] != 1 or query.shape[-2] != length
                or value.shape[-2] != length or key.shape[-2] != length):
            raise ValueError('All35 ordered native full-prefill attention bindings required')
        count('source_attention_observations' if selected is None else 'edited_attention_callbacks')
        output, weights = NATIVE_EAGER(module, query, key, value, attention_mask,
            dropout=dropout, scaling=scaling, softcap=softcap, **kwargs)
        repeated = native_module.repeat_kv(value, module.num_key_value_groups)
        if (output.shape != (1, length, query.shape[1], query.shape[-1])
                or weights.shape != (1, query.shape[1], length, length)
                or not torch.isfinite(weights).all() or not torch.isfinite(output).all()):
            raise ValueError('Actual native finite weights/head output shape required')
        observation = {'layer': index, 'layer_type': module.layer_type,
            'control_position': control, 'selected_position': selected,
            'weights': record(weights), 'repeated_value': record(repeated),
            'native_full_head_output': record(output),
            'BOS_mass_perhead': weights[0, :, -1, 0].float().cpu().tolist(),
            'control_mass_perhead': weights[0, :, -1, control].float().cpu().tolist(),
            'BOS_value': record(repeated[:, :, 0, :]),
            'control_value': record(repeated[:, :, control, :]),
            'BOS_value_RMS': float(repeated[:, :, 0, :].float().square().mean().sqrt()),
            'control_value_RMS': float(repeated[:, :, control, :].float().square().mean().sqrt())}
        if selected is not None:
            count('native_value_BMM_replay_attempts')
            replay = torch.matmul(weights, repeated).transpose(1, 2).contiguous()
            if record(replay) != observation['native_full_head_output']:
                raise ValueError('Full-shape native value BMM replay differs; whole panel censored')
            old_value_sha = tensor_sha(value)
            count('scratch_V_clone_attempts')
            scratch = repeated.clone()
            if scratch.untyped_storage().data_ptr() == repeated.untyped_storage().data_ptr():
                raise ValueError('Scratch V must not alias actual V/KV')
            count('scratch_V_zero_assignments')
            scratch[:, :, selected, :] = 0
            zero = scratch[:, :, selected, :]
            count('zero_assigned_elements', zero.numel())
            count('zero_assigned_bytes', zero.numel()*zero.element_size())
            if torch.count_nonzero(zero):
                raise ValueError('Literal selected scratch V row zero required')
            if (tensor_sha(scratch[:, :, :selected, :]) != tensor_sha(repeated[:, :, :selected, :])
                    or tensor_sha(scratch[:, :, selected+1:, :]) != tensor_sha(repeated[:, :, selected+1:, :])):
                raise ValueError('Unselected scratch value rows changed')
            count('edited_value_BMM_attempts')
            edited = torch.matmul(weights, scratch).transpose(1, 2).contiguous()
            assigned = output.clone()
            count('current_head_output_assignments')
            assigned[:, -1, :, :] = edited[:, -1, :, :]
            count('current_output_assigned_elements', assigned[:, -1].numel())
            count('current_output_assigned_bytes', assigned[:, -1].numel()*assigned.element_size())
            if (tensor_sha(assigned[:, :-1]) != tensor_sha(output[:, :-1])
                    or tensor_sha(value) != old_value_sha):
                raise ValueError('Noncurrent head outputs or actual native V changed')
            observation.update({'native_replay_exact': True, 'actual_value_unchanged': True,
                'unselected_scratch_rows_exact': True, 'noncurrent_head_outputs_exact': True,
                'scratch_selected_zero': record(zero),
                'assigned_full_head_output': record(assigned),
                'assigned_current_head_output': record(assigned[:, -1])})
            output = assigned
        observations.append(observation)
        return output, weights

    count('global_function_reference_assignments')
    native_module.eager_attention_forward = wrapper
    try:
        yield observations
        if len(observations) != 35:
            raise ValueError('Exactly35 actual native attention calls required')
    finally:
        count('global_function_reference_assignments')
        native_module.eager_attention_forward = NATIVE_EAGER
        check_native(model)


def order(row_index):
    k = (row_index//2) % 2
    return list(CONDITIONS[k:]+CONDITIONS[:k])


def one_case(model, encoded, answers, parent, row_index, count):
    check_native(model)
    initial = snapshot(model)
    ids = encoded['input_ids']
    if ids.ndim != 2 or ids.shape[0] != 1 or int(ids[0, 0]) != 2:
        raise ValueError('Actual first token must be native BOS2')
    length = ids.shape[1]
    control = control_position(length, row_index)
    count('native_sourceguards')
    with value_operator(model, length, control, count) as source_observations:
        source, cache = forward_record(model, ids, DynamicCache(config=model.config.text_config), answers)
    source_check(source, parent)
    source_prefix = noncurrent_cache_prefix(cache)
    cells, restorations = {}, {}
    for condition in order(row_index):
        count('new_first_argmax_observations')
        with value_operator(model, length, control, count, condition) as observations:
            current, edited_cache = forward_record(model, ids,
                DynamicCache(config=model.config.text_config), answers)
        edited_prefix = noncurrent_cache_prefix(edited_cache)
        if edited_prefix != source_prefix:
            raise ValueError('Actual noncurrent native KV prefix changed')
        cells[condition] = {'prefill': current, 'attention_observations': observations,
            'noncurrent_KV_prefix': edited_prefix, 'noncurrent_prefix_exact_native': True,
            'whole_KV_equal_native': current['cache'] == source['cache'],
            'full_FP32_equal_native': current['decision']['full_logits_sha256'] == source['decision']['full_logits_sha256']}
        count('native_restoration_forwards')
        restored, restored_cache = forward_record(model, ids,
            DynamicCache(config=model.config.text_config), answers)
        source_check(restored, parent)
        actual_state = snapshot(model)
        if (restored != source or noncurrent_cache_prefix(restored_cache) != source_prefix
                or actual_state != initial):
            raise ValueError('Actual native first/state restoration failed')
        restorations[condition] = {'native': restored, 'actual_state': actual_state,
                                   'function_reference': function_witness(model)}
    return {'parent_binding': parent['binding'], 'carried_native': parent['original'],
        'native_source': source, 'native_attention_observations': source_observations,
        'native_noncurrent_KV_prefix': source_prefix, 'control_position': control,
        'cells': cells, 'native_restorations': restorations, 'condition_order': order(row_index),
        'native_function_reference': function_witness(model),
        'response_continuation_tokens': 0, 'own_choice_tokens_consumed': 0,
        'base_parameter_assignments': 0, 'base_buffer_assignments': 0,
        'backward_calls': 0, 'optimizer_steps': 0}
