"""Literal full-context native branch deletion, with actual source restoration."""
from contextlib import contextmanager
import hashlib
import inspect

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer, Gemma4TextMLP

from interpretability.signed_native_mask import MODULE_SHA, MLP_FORWARD
from research.experiments.n35.core import tensor_sha, snapshot
from research.experiments.n37.continuation import decision
from research.experiments.n48.assay import choice_metrics

CONDITIONS = ('remove_PLE_additions', 'remove_attention_additions', 'remove_FFN_additions')
NORM_PATHS = {'remove_PLE_additions': 'post_per_layer_input_norm',
              'remove_attention_additions': 'post_attention_layernorm',
              'remove_FFN_additions': 'post_feedforward_layernorm'}
DECODER_FORWARD = Gemma4TextDecoderLayer.forward


def sites(model):
    config = model.config.text_config
    blocks = model.model.language_model.layers
    if (len(blocks) != 35 or config.num_hidden_layers != 35 or config.num_kv_shared_layers != 20
            or not config.hidden_size_per_layer_input):
        raise ValueError('Native35-layer PLE/shared-KV architecture required')
    path = inspect.getsourcefile(Gemma4TextDecoderLayer)
    with open(path, 'rb') as stream:
        if hashlib.sha256(stream.read()).hexdigest() != MODULE_SHA:
            raise ValueError('Pinned native Transformer source required')
    selected = {condition: [] for condition in CONDITIONS}
    for index, block in enumerate(blocks):
        if (type(block) is not Gemma4TextDecoderLayer or type(block).forward is not DECODER_FORWARD
                or 'forward' in block.__dict__ or block.layer_idx != index or block.enable_moe_block
                or type(block.mlp) is not Gemma4TextMLP or type(block.mlp).forward is not MLP_FORWARD
                or 'forward' in block.mlp.__dict__ or not block.hidden_size_per_layer_input):
            raise ValueError('Exact unmodified native decoder/MLP membership required')
        for condition, name in NORM_PATHS.items():
            norm = getattr(block, name)
            if (norm._forward_hooks or norm._forward_pre_hooks
                    or norm.weight.shape != (config.hidden_size,)
                    or norm.weight.requires_grad or norm.weight.grad is not None):
                raise ValueError('Clean native additive norm sites required')
            selected[condition].append(norm)
    if len({id(norm) for group in selected.values() for norm in group}) != 105:
        raise ValueError('All105 actual native norm sites must be distinct')
    return selected


def tensor_record(tensor):
    if (not isinstance(tensor, torch.Tensor) or tensor.requires_grad or tensor.ndim != 3
            or tensor.shape[0] != 1 or min(tensor.shape) < 1 or not torch.isfinite(tensor).all()):
        raise ValueError('Actual finite detached full-context norm tensor required')
    return {'sha256': tensor_sha(tensor), 'shape': list(tensor.shape),
            'dtype': str(tensor.dtype), 'numel': tensor.numel(),
            'element_size': tensor.element_size(),
            'rms': float(tensor.float().square().mean().sqrt()),
            'nonzero': int(torch.count_nonzero(tensor))}


@contextmanager
def observe_paths(model, count):
    selected = sites(model)
    observations = {condition: [] for condition in CONDITIONS}
    handles = []
    def observe(condition, index):
        def callback(module, args, output):
            count('native_path_observations')
            records = observations[condition]
            if index != len(records) or len(args) != 1:
                raise ValueError('Native path observations must visit every layer in order')
            records.append({'layer_index': index, 'norm_path': NORM_PATHS[condition],
                            'input_sha256': tensor_sha(args[0]), 'output': tensor_record(output)})
        return callback
    try:
        for condition in CONDITIONS:
            for index, norm in enumerate(selected[condition]):
                handles.append(norm.register_forward_hook(observe(condition, index)))
        yield observations
        if any(len(records) != 35 for records in observations.values()):
            raise ValueError('All105 actual native path observations required')
    finally:
        for handle in reversed(handles):
            handle.remove()


@contextmanager
def zero_path(model, condition, count):
    if condition not in CONDITIONS:
        raise ValueError('Fixed branch-deletion condition required')
    selected = sites(model)[condition]
    records, handles = [], []
    def replace(index):
        def callback(module, args, output):
            count('norm_zero_callbacks')
            if index != len(records) or len(args) != 1:
                raise ValueError('Exactly ordered35 full-context norm callbacks required')
            original = tensor_record(output)
            if original['shape'][2] != model.config.text_config.hidden_size:
                raise ValueError('Actual native residual width required')
            count('literal_zero_tensor_attempts')
            zero = torch.zeros_like(output)
            assigned = tensor_record(zero)
            if (assigned['shape'] != original['shape'] or assigned['dtype'] != original['dtype']
                    or assigned['nonzero'] != 0 or assigned['rms'] != 0.0
                    or zero.untyped_storage().data_ptr() == output.untyped_storage().data_ptr()):
                raise ValueError('New literal zero tensor with exact native shape/dtype required')
            records.append({'layer_index': index, 'norm_path': NORM_PATHS[condition],
                            'input_sha256': tensor_sha(args[0]),
                            'native_computed_output': original, 'assigned_output': assigned})
            count('activation_tensor_assignments')
            count('zero_assigned_elements', zero.numel())
            count('zero_assigned_bytes', zero.numel()*zero.element_size())
            return zero
        return callback
    try:
        for index, norm in enumerate(selected):
            handles.append(norm.register_forward_hook(replace(index)))
        yield records
        if len(records) != 35:
            raise ValueError('All35 actual selected branch deletions required')
    finally:
        for handle in reversed(handles):
            handle.remove()


def native(model, encoded, answers, parent, count, *, capture_paths=False):
    sites(model)
    captured = []
    def capture_h(module, args):
        if captured or len(args) != 1 or args[0].requires_grad:
            raise ValueError('Exactly one actual native full h observation required')
        captured.append(tensor_sha(args[0]))
    handle = model.model.language_model.layers[-1].mlp.down_proj.register_forward_pre_hook(capture_h)
    try:
        with torch.no_grad():
            if capture_paths:
                with observe_paths(model, count) as paths:
                    output = model(**encoded, past_key_values=DynamicCache(config=model.config.text_config),
                                   use_cache=True, logits_to_keep=1)
            else:
                paths = None
                output = model(**encoded, past_key_values=DynamicCache(config=model.config.text_config),
                               use_cache=True, logits_to_keep=1)
            current, cache = decision(output, 0, 15, str(next(model.parameters()).dtype))
            metrics = choice_metrics(output.logits[0, -1], answers)
        if len(captured) != 1:
            raise ValueError('Missing actual native h observation')
        if (current != parent['original']['decision'] or metrics != parent['original']['metrics']
                or cache != parent['native_first_full_cache']
                or captured[0] != parent['native_full_h_sha256']):
            raise ValueError('Actual original fullFP32/choice/cache/h source changed')
        result = {'decision': current, 'metrics': metrics, 'cache': cache, 'full_h_sha256': captured[0]}
        return result, paths
    finally:
        handle.remove()


def order(row_index):
    k = (row_index//2) % 3
    return list(CONDITIONS[k:]+CONDITIONS[:k])


def one_case(model, encoded, answers, parent, row_index, count):
    initial = snapshot(model)
    count('native_sourceguards')
    source, paths = native(model, encoded, answers, parent, count, capture_paths=True)
    cells, restorations = {}, {}
    for condition in order(row_index):
        with torch.no_grad(), zero_path(model, condition, count) as observations:
            output = model(**encoded, past_key_values=DynamicCache(config=model.config.text_config),
                           use_cache=True, logits_to_keep=1)
            current, cache = decision(output, 0, 15, str(next(model.parameters()).dtype))
            metrics = choice_metrics(output.logits[0, -1], answers)
        cells[condition] = {'decision': current, 'metrics': metrics, 'cache': cache,
                            'zero_path_observations': observations,
                            'full_FP32_equal_native': current['full_logits_sha256'] == source['decision']['full_logits_sha256'],
                            'whole_KV_equal_native': cache == source['cache']}
        count('native_restoration_forwards')
        restored, unused = native(model, encoded, answers, parent, count)
        if restored != source or unused is not None:
            raise ValueError('Actual native first fullFP32/choice/cache/h restoration failed')
        actual_state = snapshot(model)
        if actual_state != initial:
            raise ValueError('Actual native state/hook/version/RNG/config/buffer/grad restoration failed')
        restorations[condition] = {'native': restored, 'actual_state': actual_state}
    return {'parent_binding': parent['binding'], 'carried_native': parent['original'],
            'native_source': source, 'native_path_observations': paths,
            'cells': cells, 'native_restorations': restorations, 'condition_order': order(row_index),
            'scope': 'ALL35layers/ALLprefillpositions/literalzero returned additive outputs',
            'new_emitted_tokens': 0, 'base_parameter_assignments': 0, 'base_buffer_assignments': 0,
            'backward_calls': 0, 'optimizer_steps': 0}
