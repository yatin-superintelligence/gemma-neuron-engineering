"""Registered36 native graph, split native PLE copies and compact physical KV."""
from contextlib import contextmanager
import copy
import hashlib
import inspect
import types

import torch
from torch import nn
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4TextModel, Gemma4TextDecoderLayer, Gemma4TextAttention, Gemma4TextMLP)

from interpretability.signed_native_mask import MODULE_SHA
from interpretability.cache_proofs import check_cache
from research.experiments.n19.trajectory import cache_witness
from research.experiments.n35.core import tensor_sha, stable, snapshot
from research.experiments.n48.assay import choice_metrics

CONDITIONS = ('new_native_KV_producer', 'shared_native_KV_control',
              'silent_new_native_KV_producer')
FORWARDS = {c: c.forward for c in
            (Gemma4TextModel, Gemma4TextDecoderLayer, Gemma4TextAttention, Gemma4TextMLP)}
OPERATIONS = (
    'native_sourceguards', 'native_cached_argmax_observations',
    'native_restoration_forwards', 'edited_first_argmax_observations',
    'edited_cached_argmax_observations', 'own_choice_tokens_consumed',
    'extra_full_block_evaluations', 'extra_attention_evaluations',
    'extra_FFN_evaluations', 'extra_PLE_evaluations', 'new_KV_update_attempts',
    'block_copy_constructions', 'copied_parameter_objects', 'copied_parameter_elements',
    'copied_parameter_bytes', 'copied_buffer_objects', 'copied_buffer_elements',
    'copied_buffer_bytes', 'manual_copied_parameter_zero_assignments',
    'manual_copied_buffer_one_assignments', 'copied_attribute_assignments',
    'temporary_module_reference_assignments', 'temporary_config_reference_assignments',
    'cache_layer_appends', 'cache_update_observer_reference_assignments',
    'prepared_module_reference_bindings', 'top_config_object_copies',
    'new_text_config_value_assignments',
    'old_PLE_embedding_calls', 'new_PLE_embedding_calls',
    'old_PLE_projection_calls', 'new_PLE_projection_calls')


def tensor_record(t):
    if not isinstance(t, torch.Tensor) or t.requires_grad or not torch.isfinite(t).all():
        raise ValueError('Actual finite detached tensor required')
    return {'sha256': tensor_sha(t), 'shape': list(t.shape), 'dtype': str(t.dtype)}


def module_state(module):
    return {kind: {n: {**tensor_record(p), 'version': p._version,
                      'requires_grad': p.requires_grad, 'grad_absent': p.grad is None}
                   for n, p in iterator()}
            for kind, iterator in [('parameters', module.named_parameters),
                                   ('buffers', module.named_buffers)]}


def native_architecture(model):
    text = model.model.language_model
    cfg = model.config.text_config
    if (type(text) is not Gemma4TextModel or len(text.layers) != 35
            or cfg.num_hidden_layers != 35 or cfg.num_kv_shared_layers != 20
            or not cfg.hidden_size_per_layer_input):
        raise ValueError('Exact original35 native graph required')
    with open(inspect.getsourcefile(Gemma4TextModel), 'rb') as stream:
        if hashlib.sha256(stream.read()).hexdigest() != MODULE_SHA:
            raise ValueError('Pinned native module required')
    for c, fn in FORWARDS.items():
        if c.forward is not fn:
            raise ValueError('Native forward function changed')
    for i, block in enumerate(text.layers):
        if (type(block) is not Gemma4TextDecoderLayer or block.layer_idx != i
                or type(block.self_attn) is not Gemma4TextAttention
                or type(block.mlp) is not Gemma4TextMLP or block.enable_moe_block
                or any('forward' in m.__dict__ for m in (block, block.mlp, block.self_attn))):
            raise ValueError('Actual native decoder membership required')
    source = text.layers[14]
    if (source.self_attn.is_kv_shared_layer or not source.self_attn.store_full_length_kv
            or source.self_attn.layer_type != 'full_attention'
            or source.mlp.intermediate_size != cfg.intermediate_size):
        raise ValueError('Architecture-fixed final native producer14 required')
    return text


class AppendedEmbedding(nn.Module):
    """Preserve the actual original embedding call and append one native slice."""
    def __init__(self, original, slot_weight, count):
        super().__init__()
        self.original = original
        self.slot_weight = nn.Parameter(slot_weight.detach().clone(), requires_grad=False)
        self.register_buffer('slot_scale', original.embed_scale.detach().clone(), persistent=False)
        self.count = count

    def forward(self, ids):
        self.count('old_PLE_embedding_calls')
        old = self.original(ids)
        self.count('new_PLE_embedding_calls')
        new = F.embedding(ids, self.slot_weight, self.original.padding_idx,
                          self.original.max_norm, self.original.norm_type,
                          self.original.scale_grad_by_freq, self.original.sparse)
        new = new * self.slot_scale.to(self.slot_weight.dtype)
        return torch.cat((old, new), dim=-1)


class AppendedProjection(nn.Module):
    def __init__(self, original, slot_weight, count):
        super().__init__()
        self.original = original
        self.slot_weight = nn.Parameter(slot_weight.detach().clone(), requires_grad=False)
        self.count = count

    def forward(self, hidden):
        self.count('old_PLE_projection_calls')
        old = self.original(hidden)
        self.count('new_PLE_projection_calls')
        return torch.cat((old, F.linear(hidden, self.slot_weight)), dim=-1)


def compact_recipe(config):
    return {'format': 'digital_surgery_native_extension_v1', 'base_layers': 35,
            'source_layer': 14, 'logical_added_layer': 35, 'physical_added_cache': 15,
            'cache_slots': 16, 'extra_cache_class': 'DynamicLayer',
            'PLE_slot_source': 14, 'PLE_width': config.hidden_size_per_layer_input,
            'preserve_old_embedding_projection_shapes': True,
            'producer_override': {'new_native_KV_producer': False,
                'shared_native_KV_control': True, 'silent_new_native_KV_producer': False},
            'silent': {'post_norm_gamma': 0, 'layer_scalar_buffer': 1},
            'stock_AutoModel_or_default_generate_compatible': False}


class ExtensionPool:
    """Prepared native copies live outside base registration until installed."""
    def __init__(self, model, count, recipe=None):
        text = native_architecture(model)
        self.model, self.text, self.count = model, text, count
        self.base_layers = text.layers
        self.base_embedding, self.base_projection = text.embed_tokens_per_layer, text.per_layer_model_projection
        self.base_configs = (model.config, model.model.config, text.config)
        self.base_ids = {n: id(p) for n, p in model.named_parameters()}
        self.base_module_ids = {n: id(m) for n, m in model.named_modules()}
        self.recipe = compact_recipe(text.config)
        if recipe is not None and recipe != self.recipe:
            raise ValueError('Exact registered compact recipe required')
        self.clones, self.handles = {}, []
        self.source = text.layers[14]
        self.native_source_state = module_state(self.source)
        for condition in CONDITIONS:
            count('block_copy_constructions')
            clone = copy.deepcopy(self.source).eval().requires_grad_(False)
            if module_state(clone) != self.native_source_state:
                # deepcopy retains values and tensor versions for parameters; buffers can reset
                # their version. Require byte/value structure, not assumed version equivalence.
                for kind in ('parameters', 'buffers'):
                    for name, actual in module_state(clone)[kind].items():
                        if any(actual[k] != self.native_source_state[kind][name][k]
                               for k in ('sha256', 'shape', 'dtype', 'requires_grad', 'grad_absent')):
                            raise ValueError('Native source copy changed values')
            clone.layer_idx = 35
            clone.self_attn.layer_idx = 15
            clone.self_attn.is_kv_shared_layer = condition == 'shared_native_KV_control'
            clone.self_attn.store_full_length_kv = condition != 'shared_native_KV_control'
            count('copied_attribute_assignments', 4)
            if condition == 'silent_new_native_KV_producer':
                with torch.no_grad():
                    for name in ('post_attention_layernorm', 'post_feedforward_layernorm',
                                 'post_per_layer_input_norm'):
                        count('manual_copied_parameter_zero_assignments')
                        getattr(clone, name).weight.zero_()
                    count('manual_copied_buffer_one_assignments')
                    clone.layer_scalar.fill_(1)
            self.clones[condition] = clone
            for module, name in ((clone, 'extra_full_block_evaluations'),
                    (clone.self_attn, 'extra_attention_evaluations'),
                    (clone.mlp, 'extra_FFN_evaluations'),
                    (clone.per_layer_projection, 'extra_PLE_evaluations')):
                self.handles.append(module.register_forward_pre_hook(
                    lambda module, args, name=name: count(name)))
        width = text.config.hidden_size_per_layer_input
        self.embedding = AppendedEmbedding(self.base_embedding,
            self.base_embedding.weight[:, 14*width:15*width], count).eval()
        self.projection = AppendedProjection(self.base_projection,
            self.base_projection.weight[14*width:15*width, :], count).eval()
        if (tensor_sha(self.embedding.slot_weight) != tensor_sha(self.base_embedding.weight[:, 14*width:15*width])
                or tensor_sha(self.projection.slot_weight) != tensor_sha(self.base_projection.weight[14*width:15*width])
                or tensor_sha(self.embedding.slot_scale) != tensor_sha(self.base_embedding.embed_scale)):
            raise ValueError('Exact native PLE slice and scale copies required')
        count('prepared_module_reference_bindings', 2)
        self.outer_config = copy.deepcopy(model.config)
        count('top_config_object_copies')
        cfg = self.outer_config.text_config
        cfg.num_hidden_layers = 36
        cfg.num_kv_shared_layers = 21
        cfg.layer_types = list(cfg.layer_types)+['full_attention']
        cfg.digital_surgery_native_extension = copy.deepcopy(self.recipe)
        count('new_text_config_value_assignments', 4)
        self.extended_config = cfg
        self.extended_lists = {c: nn.ModuleList([*self.base_layers, self.clones[c]]).eval()
                               for c in CONDITIONS}
        for graph in self.extended_lists.values():
            count('prepared_module_reference_bindings', len(graph))
        copied = list(self.copied_parameters())
        buffers = list(self.copied_buffers())
        base_storages = {p.untyped_storage().data_ptr() for p in model.parameters()}
        ptrs = [p.untyped_storage().data_ptr() for _, p in copied]
        if len(set(ptrs)) != len(ptrs) or set(ptrs) & base_storages:
            raise ValueError('Native copies must be nonaliasing')
        buffer_ptrs = [p.untyped_storage().data_ptr() for _, p in buffers]
        if (len(set(buffer_ptrs)) != len(buffer_ptrs)
                or set(buffer_ptrs) & {p.untyped_storage().data_ptr() for p in model.buffers()}):
            raise ValueError('Native copied buffers must be nonaliasing')
        for _, p in copied:
            for key, amount in (('copied_parameter_objects', 1),
                    ('copied_parameter_elements', p.numel()),
                    ('copied_parameter_bytes', p.numel()*p.element_size())):
                count(key, amount)
        for _, p in buffers:
            count('copied_buffer_objects')
            count('copied_buffer_elements', p.numel())
            count('copied_buffer_bytes', p.numel()*p.element_size())
        self.initial = self.witness()
        self.check()

    def copied_parameters(self):
        for condition, clone in self.clones.items():
            for name, p in clone.named_parameters():
                yield condition+'.'+name, p
        yield 'PLE_embedding.slot_weight', self.embedding.slot_weight
        yield 'PLE_projection.slot_weight', self.projection.slot_weight

    def copied_buffers(self):
        for condition, clone in self.clones.items():
            for name, p in clone.named_buffers():
                yield condition+'.'+name, p
        yield 'PLE_embedding.slot_scale', self.embedding.slot_scale

    def witness(self):
        return {'native_source': module_state(self.source),
                'copies': {c: module_state(v) for c, v in self.clones.items()},
                'clone_architecture': {c: {'logical_index': v.layer_idx,
                    'physical_cache_index': v.self_attn.layer_idx,
                    'is_KV_shared': v.self_attn.is_kv_shared_layer,
                    'store_full_length_KV': v.self_attn.store_full_length_kv,
                    'native_source_config_sha256': stable(v.config.to_dict())}
                    for c, v in self.clones.items()},
                'PLE_embedding_copy': {**tensor_record(self.embedding.slot_weight),
                                       'version': self.embedding.slot_weight._version},
                'PLE_projection_copy': {**tensor_record(self.projection.slot_weight),
                                        'version': self.projection.slot_weight._version},
                'PLE_scale_copy': tensor_record(self.embedding.slot_scale),
                'actual_copied_parameter_objects': sum(1 for _ in self.copied_parameters()),
                'actual_copied_parameter_elements': sum(p.numel() for _, p in self.copied_parameters()),
                'actual_copied_parameter_bytes': sum(p.numel()*p.element_size() for _, p in self.copied_parameters()),
                'actual_copied_buffer_objects': sum(1 for _ in self.copied_buffers()),
                'actual_copied_buffer_bytes': sum(p.numel()*p.element_size() for _, p in self.copied_buffers()),
                'source_and_copies_nonaliasing': True, 'recipe': self.recipe}

    def check(self):
        if ({n: id(p) for n, p in self.model.named_parameters()} != self.base_ids
                or {n: id(m) for n, m in self.model.named_modules()} != self.base_module_ids
                or any(a is not b for a, b in zip(
                    (self.model.config, self.model.model.config, self.text.config), self.base_configs))
                or self.witness() != self.initial):
            raise ValueError('Native/copy values, versions or references changed')
        return True

    def cache(self):
        cache = DynamicCache(config=self.base_configs[2])
        if len(cache.layers) != 15:
            raise ValueError('Original15 compact physical KV layers required')
        extra = DynamicLayer()
        native_update = extra.update
        def update(layer, *args, **kwargs):
            self.count('new_KV_update_attempts')
            return native_update(*args, **kwargs)
        extra.update = types.MethodType(update, extra)
        self.count('cache_update_observer_reference_assignments')
        cache.layers.append(extra)
        self.count('cache_layer_appends')
        return cache

    @contextmanager
    def installed(self, condition):
        if condition not in CONDITIONS:
            raise ValueError('Fixed native extension condition required')
        self.check()
        try:
            for name, value in (('layers', self.extended_lists[condition]),
                    ('embed_tokens_per_layer', self.embedding),
                    ('per_layer_model_projection', self.projection)):
                self.count('temporary_module_reference_assignments')
                setattr(self.text, name, value)
            for module, config in ((self.model, self.outer_config),
                    (self.model.model, self.outer_config), (self.text, self.extended_config)):
                self.count('temporary_config_reference_assignments')
                module.config = config
            if (len(self.text.layers) != 36 or self.text.layers[35] is not self.clones[condition]
                    or any(self.text.layers[i] is not self.base_layers[i] for i in range(35))):
                raise ValueError('Actual registered36 graph required')
            yield
        finally:
            for name, value in (('layers', self.base_layers),
                    ('embed_tokens_per_layer', self.base_embedding),
                    ('per_layer_model_projection', self.base_projection)):
                self.count('temporary_module_reference_assignments')
                setattr(self.text, name, value)
            for module, config in zip((self.model, self.model.model, self.text), self.base_configs):
                self.count('temporary_config_reference_assignments')
                module.config = config
        self.check()

    def remove(self):
        for handle in self.handles:
            handle.remove()


def projected_cache(witness):
    layers = witness['layers'][:15]
    return {'layers': layers, 'sha256': stable(layers)}


def cache_retained_prefix(cache):
    """Fingerprint the exact old positions retained after the next one-token update."""
    rows = []
    for i, layer in enumerate(cache.layers):
        if not layer.is_initialized:
            rows.append({'layer': i, 'empty': True})
            continue
        old_length = layer.keys.shape[-2]
        retained = min(old_length, layer.sliding_window-2) if layer.is_sliding else old_length
        rows.append({'layer': i, 'empty': False, 'old_seen': int(layer.get_seq_length()),
            'old_stored': old_length, 'retained_positions': retained,
            'keys': tensor_record(layer.keys[..., -retained:, :]) if retained else None,
            'values': tensor_record(layer.values[..., -retained:, :]) if retained else None})
    return rows


def validate_growth(cache, before):
    rows = []
    for layer, old in zip(cache.layers, before):
        if old['empty']:
            if layer.is_initialized:
                raise ValueError('Shared control unexpectedly wrote extra slot')
            rows.append({'layer': old['layer'], 'empty_unchanged': True})
            continue
        n = old['retained_positions']
        expected = min(old['old_stored']+1, layer.sliding_window-1) if layer.is_sliding else old['old_stored']+1
        if layer.keys.shape[-2] != expected or int(layer.get_seq_length()) != old['old_seen']+1:
            raise ValueError('Actual one-token native cache growth required')
        for name in ('keys', 'values'):
            tensor = getattr(layer, name)
            if not torch.isfinite(tensor).all() or (n and tensor_record(tensor[..., :n, :]) != old[name]):
                raise ValueError('Retained actual cache prefix changed')
        rows.append({'layer': old['layer'], 'old_positions_retained': n,
                     'retained_prefix_exact': True, 'new_stored': expected,
                     'new_seen': int(layer.get_seq_length())})
    return rows


def forward_record(model, ids, cache, answers, step=0, extended=False, condition=None):
    text = model.model.language_model
    observed = {}
    def base_output(module, args, output):
        if 'base_hidden' in observed:
            raise ValueError('Exactly one original35 prefix required')
        observed['base_hidden'] = tensor_record(output)
    def h_input(module, args):
        if 'full_h_sha256' in observed or len(args) != 1:
            raise ValueError('Exactly one last-original MLP h observation required')
        observed['full_h_sha256'] = tensor_sha(args[0])
    handles = [text.layers[34].register_forward_hook(base_output),
               text.layers[34].mlp.down_proj.register_forward_pre_hook(h_input)]
    try:
        with torch.no_grad():
            output = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        logits = output.logits[0, -1]
        if not torch.isfinite(logits).all() or logits.dtype != next(model.parameters()).dtype:
            raise ValueError('Actual finite native-dtype logits required')
        whole = cache_witness(output.past_key_values)
        check_cache(whole, 16 if extended else 15, str(logits.dtype))
        if any(x['keys'] is None for x in whole['layers'][:15]):
            raise ValueError('All original15 cache entries required')
        if extended and ((whole['layers'][15]['keys'] is None) != (condition == 'shared_native_KV_control')):
            raise ValueError('Actual producer/shared extra cache slot mismatch')
        if len(observed) != 2:
            raise ValueError('Actual base prefix and native h observations required')
        record = {'decision': {'step': step, 'argmax': int(logits.argmax()),
            'full_logits_sha256': tensor_sha(logits.float()), 'cache_sha256': whole['sha256'],
            'raw_logits_dtype': str(logits.dtype)}, 'metrics': choice_metrics(logits, answers),
            'cache': whole, 'original15_cache': projected_cache(whole), **observed,
            'actual_input_ids_sha256': tensor_sha(ids), 'actual_input_tokens': ids.numel(),
            'actual_cache_classes': [type(v).__name__ for v in cache.layers]}
        return record, output.past_key_values
    finally:
        for handle in reversed(handles):
            handle.remove()


def source_check(record, parent):
    if (record['decision'] != parent['original']['decision']
            or record['metrics'] != parent['original']['metrics']
            or record['cache'] != parent['native_first_full_cache']
            or record['full_h_sha256'] != parent['native_full_h_sha256']):
        raise ValueError('Frozen original fullFP32/argmax/wholecache/h source changed')


def order(row_index):
    k = (row_index//2) % 3
    return list(CONDITIONS[k:]+CONDITIONS[:k])


def one_case(model, encoded, answers, parent, row_index, pool, count,
             eos=frozenset({1, 106, 50}), require_native_probe=True):
    initial = snapshot(model)
    ids = encoded['input_ids']
    count('native_sourceguards')
    source, source_cache = forward_record(model, ids,
        DynamicCache(config=model.config.text_config), answers)
    source_check(source, parent)
    token = source['decision']['argmax']
    native_probe = None
    if token not in eos:
        old = cache_retained_prefix(source_cache)
        count('native_cached_argmax_observations')
        count('own_choice_tokens_consumed')
        native_probe, source_cache = forward_record(model, ids.new_tensor([[token]]),
                                                    source_cache, answers, step=1)
        native_probe['growth'] = validate_growth(source_cache, old)
        native_probe['consumed_own_first_argmax'] = token
    elif require_native_probe:
        raise ValueError('Original real-cohort A/B first choice must be nonEOS')
    cells, restorations = {}, {}
    for condition in order(row_index):
        with pool.installed(condition):
            count('edited_first_argmax_observations')
            first, edited_cache = forward_record(model, ids, pool.cache(), answers,
                extended=True, condition=condition)
            if (first['base_hidden'] != source['base_hidden']
                    or first['original15_cache'] != source['cache']
                    or first['full_h_sha256'] != source['full_h_sha256']):
                raise ValueError('Appended graph changed original35 prefill prefix')
            chosen = first['decision']['argmax']
            probe, growth = None, None
            if chosen not in eos:
                old = cache_retained_prefix(edited_cache)
                count('edited_cached_argmax_observations')
                count('own_choice_tokens_consumed')
                probe, edited_cache = forward_record(model, ids.new_tensor([[chosen]]),
                    edited_cache, answers, step=1, extended=True, condition=condition)
                growth = validate_growth(edited_cache, old)
                probe['consumed_own_first_argmax'] = chosen
                if chosen == token and native_probe is not None:
                    if (probe['base_hidden'] != native_probe['base_hidden']
                            or probe['original15_cache'] != native_probe['cache']
                            or probe['full_h_sha256'] != native_probe['full_h_sha256']):
                        raise ValueError('Original35 same-own-history cached prefix changed')
            prefill_identity = (first['decision']['full_logits_sha256'] == source['decision']['full_logits_sha256']
                                and chosen == token)
            cached_identity = (probe is not None and native_probe is not None and chosen == token
                and probe['decision']['argmax'] == native_probe['decision']['argmax']
                and probe['decision']['full_logits_sha256'] == native_probe['decision']['full_logits_sha256']
                and probe['base_hidden'] == native_probe['base_hidden']
                and probe['original15_cache'] == native_probe['cache'])
            # Both EOS in the dedicated host fixture: the required behavior is no probe.
            host_joint_eos = not require_native_probe and chosen in eos and token in eos and probe is None
            cells[condition] = {'prefill': first, 'cached_probe': probe,
                'cached_probe_applicable': chosen not in eos, 'cache_growth': growth,
                'same_own_input_as_native': chosen == token,
                'prefill_full_FP32_argmax_identity': prefill_identity,
                'cached_full_FP32_argmax_prefix_identity': cached_identity,
                'host_both_EOS_no_probe': host_joint_eos}
        count('native_restoration_forwards')
        restored, unused_cache = forward_record(model, ids,
            DynamicCache(config=model.config.text_config), answers)
        source_check(restored, parent)
        actual_state = snapshot(model)
        if restored != source or actual_state != initial or not pool.check():
            raise ValueError('Actual native35 graph/source/state restoration failed')
        restorations[condition] = {'native': restored, 'actual_state': actual_state,
                                  'copy_pool_sha256': stable(pool.initial)}
    silent = cells['silent_new_native_KV_producer']
    return {'parent_binding': parent['binding'], 'carried_native': parent['original'],
        'native_source': source, 'native_cached_reference': native_probe,
        'cells': cells, 'native_restorations': restorations, 'condition_order': order(row_index),
        'silent_primary_components': {'prefill_identity': silent['prefill_full_FP32_argmax_identity'],
                                     'cached_identity': silent['cached_full_FP32_argmax_prefix_identity']},
        'joint_silent_primary_pass': silent['prefill_full_FP32_argmax_identity']
                                    and silent['cached_full_FP32_argmax_prefix_identity'],
        'copy_pool_sha256': stable(pool.initial), 'base_parameter_assignments': 0,
        'base_buffer_assignments': 0, 'backward_calls': 0, 'optimizer_steps': 0,
        'free_program_response_generations': 0}
