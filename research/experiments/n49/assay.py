"""Only the missing RMS crosses; old complete diagonal decisions are carried."""
from contextlib import contextmanager
import hashlib
import inspect
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer, Gemma4TextMLP
from interpretability.native_rms_factor_v2 import check_norm, compose
from interpretability.signed_native_mask import transform, MODULE_SHA, MLP_FORWARD
from research.experiments.n35.core import tensor_sha
from research.experiments.n37.continuation import decision
from research.experiments.n48.assay import choice_metrics, EDITS


def site(model):
    block = model.model.language_model.layers[-1]
    if (type(block) is not Gemma4TextDecoderLayer or type(block.mlp) is not Gemma4TextMLP
            or block.layer_idx != len(model.model.language_model.layers)-1 or block.enable_moe_block
            or type(block.mlp).forward is not MLP_FORWARD or 'forward' in block.mlp.__dict__):
        raise ValueError('Unmodified final dense native branch required')
    if hashlib.sha256(open(inspect.getsourcefile(Gemma4TextDecoderLayer), 'rb').read()).hexdigest() != MODULE_SHA:
        raise ValueError('Pinned transformer source required')
    norm = block.post_feedforward_layernorm
    check_norm(norm)
    if any(m._forward_hooks or m._forward_pre_hooks for m in [block.mlp.down_proj, norm]):
        raise ValueError('Clean capture and replacement sites required')
    return block, norm


def reciprocal(norm, raw):
    if raw.requires_grad or raw.ndim != 3 or raw.shape[:2] != (1, 1):
        raise ValueError('Detached single current native branch required')
    return torch.pow(raw.float().pow(2).mean(-1, keepdim=True)+norm.eps, -0.5)


def state_witness(raw, q, output):
    return {'raw_sha256': tensor_sha(raw), 'reciprocal_sha256': tensor_sha(q),
            'reciprocal': float(q.item()), 'output_sha256': tensor_sha(output)}


@contextmanager
def observe(model):
    block, norm = site(model); saved = {}; handles = []
    def h_observed(module, args):
        if saved or args[0].requires_grad:
            raise ValueError('Exactly one detached capture required')
        saved['h'] = args[0].detach().clone()
        _, saved['activation'] = transform(args[0], 'original')
    def norm_observed(module, args, output):
        if 'h' not in saved or 'raw' in saved: raise ValueError('Aligned single capture required')
        raw = args[0][:, -1:, :].detach().clone(); q = reciprocal(module, raw)
        if tensor_sha(compose(module, raw, q)) != tensor_sha(output[:, -1:, :]):
            raise ValueError('Native direct-gamma operator order failed')
        saved.update(raw=raw, q=q, shape=list(args[0].shape), norm_output=output[:, -1:, :].detach().clone())
    try:
        handles.append(block.mlp.down_proj.register_forward_pre_hook(h_observed))
        handles.append(norm.register_forward_hook(norm_observed))
        yield saved
        if set(saved) != {'h', 'activation', 'raw', 'q', 'shape', 'norm_output'}:
            raise ValueError('Incomplete actual source capture')
    finally:
        for h in reversed(handles): h.remove()


def source_forward(model, encoded, answer_ids):
    slots = model.config.text_config.num_hidden_layers-model.config.text_config.num_kv_shared_layers
    with torch.no_grad(), observe(model) as captured:
        output = model(**encoded, past_key_values=DynamicCache(config=model.config.text_config), use_cache=True, logits_to_keep=1)
        d, cache = decision(output, 0, slots, str(next(model.parameters()).dtype))
        metrics = choice_metrics(output.logits[0, -1], answer_ids)
    record = {'decision': d, 'activation': captured['activation'], 'metrics': metrics}
    return record, cache, captured


@contextmanager
def cross(model, captured, raw, q, count):
    block, norm = site(model)
    replacement = compose(norm, raw, q)
    witness = {'calls': 0, 'activation_assignments': 0, 'native_h_exact': False, 'other_positions_byte_exact': False,
               'replacement_sha256': tensor_sha(replacement), 'raw_sha256': tensor_sha(raw), 'reciprocal_sha256': tensor_sha(q)}
    def guard_h(module, args):
        if tensor_sha(args[0]) != tensor_sha(captured['h']): raise ValueError('Full native h changed')
        witness['native_h_exact'] = True
    def replace(module, args, output):
        if (witness['calls'] or not witness['native_h_exact'] or list(args[0].shape) != captured['shape']
                or tensor_sha(args[0][:, -1:, :]) != tensor_sha(captured['raw'])
                or tensor_sha(output[:, -1:, :]) != tensor_sha(captured['norm_output'])):
            raise ValueError('Cross recipient differs from actual native source')
        value = output.clone(); count('activation_assignments')
        value[:, -1:, :].copy_(replacement)
        if tensor_sha(value[:, :-1]) != tensor_sha(output[:, :-1]):
            raise ValueError('Noncurrent RMS output changed')
        witness.update(calls=1, activation_assignments=1, other_positions_byte_exact=True)
        return value
    handles = [block.mlp.down_proj.register_forward_pre_hook(guard_h), norm.register_forward_hook(replace)]
    try:
        yield witness
        if witness['calls'] != 1: raise ValueError('Missing actual cross call')
    finally:
        for h in reversed(handles): h.remove()


def one_case(model, encoded, answer_ids, parent, row_index, count=lambda name: None):
    count('native_sourceguards'); original, cache, captured = source_forward(model, encoded, answer_ids)
    if original != parent['original'] or cache != parent['native_first_full_cache']:
        raise ValueError('N48 source decision/activation/full cache mismatch')
    block, norm = site(model)
    slots = model.config.text_config.num_hidden_layers-model.config.text_config.num_kv_shared_layers
    groups = {}
    for index, mode in enumerate(EDITS):
        count('full_shaped_h_constructions'); edited_h, activation = transform(captured['h'], mode)
        if activation != parent['edits'][mode]['activation']:
            raise ValueError('Exact completed N48 activation and mask lineage required')
        count('extra_down_projection_evaluations')
        with torch.no_grad(): raw = block.mlp.down_proj(edited_h)[:, -1:, :].detach().clone()
        q = reciprocal(norm, raw); edited_output = compose(norm, raw, q)
        cells, restorations = {}, {}
        order = ['01', '10'] if (row_index+index) % 2 == 0 else ['10', '01']
        for key in order:
            v, stat = (captured['raw'], q) if key == '01' else (raw, captured['q'])
            with torch.no_grad(), cross(model, captured, v, stat, count) as receipt:
                output = model(**encoded, past_key_values=DynamicCache(config=model.config.text_config), use_cache=True, logits_to_keep=1)
                d, actual_cache = decision(output, 0, slots, str(next(model.parameters()).dtype))
                metrics = choice_metrics(output.logits[0, -1], answer_ids)
            if actual_cache != cache: raise ValueError('Cross changed upstream KV cache')
            cells[key] = {'decision': d, 'metrics': metrics, 'replacement': receipt}
            count('native_restoration_forwards'); restored, restored_cache, restored_state = source_forward(model, encoded, answer_ids)
            if (restored != original or restored_cache != cache
                    or tensor_sha(restored_state['h']) != tensor_sha(captured['h'])
                    or state_witness(restored_state['raw'], restored_state['q'], restored_state['norm_output']) != state_witness(captured['raw'], captured['q'], captured['norm_output'])):
                raise ValueError('Actual complete native restoration failed')
            restorations[key] = restored
        groups[mode] = {'activation': activation, 'edited_source': state_witness(raw, q, edited_output),
                        'cells': cells, 'order': order, 'restorations': restorations,
                        'carried_00': parent['original'], 'carried_11': parent['edits'][mode]}
    return {'parent_binding': parent['binding'], 'original_source_guard': original,
            'native_first_full_cache': cache, 'native_source': state_witness(captured['raw'], captured['q'], captured['norm_output']),
            'full_h_sha256': tensor_sha(captured['h']), 'groups': groups,
            'answer_token_ids': answer_ids, 'input_tokens': int(encoded['input_ids'].numel()),
            'new_emitted_tokens': 0, 'parameter_assignments': 0, 'backward_calls': 0, 'optimizer_steps': 0}
