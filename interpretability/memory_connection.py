"""A fixed copied-projection memory connection. No training or fitted vectors.

This is a prototype for short causal contexts in the Gemma 4 implementation.
It adds a memory route, not a complete transformer layer. Construct the parent
with its original configuration before changing cache allocation metadata.
"""
from contextlib import contextmanager
import copy
import hashlib
import math
import torch
from torch import nn
from transformers import PreTrainedConfig
from transformers.cache_utils import DynamicCache
from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb


def tensor_hash(value):
    return hashlib.sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


class CopiedMemoryProjection(nn.Module):
    def __init__(self, source, target_index):
        super().__init__()
        for name in ['k_proj', 'v_proj']:
            module = getattr(source, name, None)
            if module is not None and (not isinstance(module, nn.Linear) or not module.weight.is_floating_point()
                                       or getattr(module.weight, 'quant_state', None) is not None):
                raise ValueError('Unpacked floating-point source projections required')
        self.k_proj = copy.deepcopy(source.k_proj)
        self.v_proj = copy.deepcopy(source.v_proj)
        self.k_norm = copy.deepcopy(source.k_norm)
        self.v_norm = copy.deepcopy(source.v_norm)
        self.head_dim = source.head_dim
        self.target_index = target_index
        self.requires_grad_(False)

    def forward(self, hidden, position_embeddings, cache):
        shape = (*hidden.shape[:-1], -1, self.head_dim)
        keys = self.k_proj(hidden).view(shape)
        values = self.v_proj(hidden).view(shape) if self.v_proj is not None else keys
        keys = self.k_norm(keys)
        keys = apply_rotary_pos_emb(keys, *position_embeddings, unsqueeze_dim=2).transpose(1, 2)
        values = self.v_norm(values).transpose(1, 2)
        if cache is not None:
            if not hasattr(cache, 'layers') or len(cache.layers) <= self.target_index:
                raise ValueError('Start a fresh sequence with the expanded cache allocation')
            keys, values = cache.update(keys, values, self.target_index)
        return keys, values


@contextmanager
def memory_connection(model, source_index=14, target_index=24, strength=.25, mode='value_history'):
    """Temporarily add copied K/V projections and a fixed native-memory read.

    Each sequence must start with a fresh cache. Do not reuse a cache across
    different policies or after leaving this context. Only causal full-attention
    source/target pairs are supported by this prototype.
    """
    modes = {'kv', 'key_only', 'value_history', 'value_current', 'value_permuted'}
    if mode not in modes or not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Choose a declared mode and a fixed strength between zero and one')
    decoder = model.model.language_model
    layers = decoder.layers
    if not 0 <= source_index < target_index < len(layers):
        raise ValueError('Ordered valid source and target layers required')
    source, target = layers[source_index].self_attn, layers[target_index].self_attn
    if (source.is_kv_shared_layer or not target.is_kv_shared_layer or
        source.layer_type != 'full_attention' or target.layer_type != 'full_attention' or
        source.head_dim != target.head_dim):
        raise ValueError('Require an original full-attention producer and a compatible shared reader')
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError('The parent model must be frozen with no parameter gradients')
    name = '_neuron_memory_connection'
    if hasattr(model, name):
        raise ValueError('A memory connection is already installed')
    widths = [layer.mlp.down_proj.in_features for layer in layers]
    flags = [(layer.self_attn.is_kv_shared_layer, layer.self_attn.store_full_length_kv) for layer in layers]
    original = {key: parameter._version for key, parameter in model.named_parameters()}
    configs = {id(config): config for config in [model.config.text_config, decoder.config]}
    shared_counts = {key: config.num_kv_shared_layers for key, config in configs.items()}
    branch = CopiedMemoryProjection(source, target_index).eval()
    copies = {}
    for key, parameter in branch.named_parameters():
        expected = source.get_parameter(key)
        assert torch.equal(parameter, expected)
        copies[key] = tensor_hash(parameter)
    receipt = {'source':source_index,'target':target_index,'mode':mode,'strength':strength,'calls':0,
        'projection_calls':0,'copied_parameter_hashes':copies,
        'added_parameters':sum(p.numel() for p in branch.parameters()),'optimizer_steps':0,
        'MLP_widths_changed':False,'fixed_input_path_scope':True}
    def hook(module, arguments, kwargs):
        receipt['calls'] += 1
        if strength == 0:
            return None
        hidden = kwargs.get('hidden_states', arguments[0] if arguments else None)
        positions = kwargs.get('position_embeddings')
        shared = kwargs.get('shared_kv_states')
        cache = kwargs.get('past_key_values')
        if hidden is None or positions is None or shared is None:
            raise ValueError('Expected explicit Gemma hidden states, positions and shared KV')
        old_keys, old_values = shared[target.layer_type]
        new_keys, new_values = branch(hidden, positions, cache)
        if new_keys.shape != old_keys.shape or new_values.shape != old_values.shape:
            raise ValueError('Copied memory and native shared memory shapes differ')
        if not torch.isfinite(new_keys).all() or not torch.isfinite(new_values).all():
            raise ValueError('Nonfinite copied memory')
        receipt['projection_calls'] += 1
        def blend(old, new):
            return new if strength == 1 else torch.lerp(old, new, strength)
        keys, values = old_keys, old_values
        if mode in ['kv', 'key_only']:
            keys = blend(old_keys, new_keys)
        if mode in ['kv', 'value_history']:
            values = blend(old_values, new_values)
        elif mode == 'value_permuted':
            values = blend(old_values, new_values.roll(shifts=1, dims=-1))
        elif mode == 'value_current':
            values = old_values.clone()
            values[:, :, -1:] = blend(old_values[:, :, -1:], new_values[:, :, -1:])
        shared[target.layer_type] = (keys, values)
        return None
    handle = None
    cache_handle = None
    cache_config = PreTrainedConfig(num_hidden_layers=len(layers),
        layer_types=list(decoder.config.layer_types),sliding_window=decoder.config.sliding_window)
    receipt['cache_expansions'] = 0
    def provision_cache(module, arguments, kwargs):
        if not kwargs.get('use_cache', True):
            return None
        cache = kwargs.get('past_key_values')
        if cache is None:
            cache = DynamicCache(config=cache_config)
            cache.layers = cache.layers[:target_index+1]
            kwargs = dict(kwargs)
            kwargs['past_key_values'] = cache
            receipt['cache_expansions'] += 1
            return arguments, kwargs
        if not isinstance(cache, DynamicCache):
            raise ValueError('This prototype requires a DynamicCache')
        if len(cache.layers) <= target_index:
            if cache.get_seq_length() != 0:
                raise ValueError('Cannot add historical memory to an already populated native cache')
            fresh = DynamicCache(config=cache_config)
            cache.layers.extend(fresh.layers[len(cache.layers):target_index+1])
            receipt['cache_expansions'] += 1
        if strength and cache.get_seq_length() != cache.get_seq_length(target_index):
            raise ValueError('Start a fresh sequence for each connection policy')
        return None
    try:
        # Cache-only metadata avoids both a zero-count slicing corner case and
        # accidental reconstruction of the parent's double-wide MLP shapes.
        model.add_module(name, branch)
        cache_handle = model.register_forward_pre_hook(provision_cache,with_kwargs=True)
        handle = target.register_forward_pre_hook(hook, with_kwargs=True)
        assert widths == [layer.mlp.down_proj.in_features for layer in layers]
        yield receipt
    finally:
        if handle is not None:
            handle.remove()
        if cache_handle is not None:
            cache_handle.remove()
        if hasattr(model, name):
            delattr(model, name)
        for key, config in configs.items():
            config.num_kv_shared_layers = shared_counts[key]
        assert widths == [layer.mlp.down_proj.in_features for layer in layers]
        assert flags == [(layer.self_attn.is_kv_shared_layer, layer.self_attn.store_full_length_kv) for layer in layers]
        assert original == {key:p._version for key,p in model.named_parameters()}
        receipt['restored_configuration_and_original_parameter_versions'] = True
