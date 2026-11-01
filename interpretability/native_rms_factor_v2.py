"""Exact native-state controls for one current-position Gemma 4 RMSNorm."""

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import inspect

import torch
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm


NATIVE_CLASS_SHA256 = 'bda3d4e437499d1a7d69b65d0093cc56f1e69b85f8c9a2195150910a886d66d9'

_NATIVE_FORWARD = Gemma4RMSNorm.forward
_NATIVE_NORM = Gemma4RMSNorm._norm


@lru_cache(maxsize=1)
def verify_class_source():
    if hashlib.sha256(inspect.getsource(Gemma4RMSNorm).rstrip().encode()).hexdigest() != NATIVE_CLASS_SHA256:
        raise ValueError('Pinned RMSNorm operator order changed')
    return True



def tensor_sha(value):
    raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def check_norm(norm):
    if type(norm) is not Gemma4RMSNorm:
        raise ValueError('Exact pinned native Gemma4RMSNorm required')
    if (getattr(norm.forward, '__func__', None) is not _NATIVE_FORWARD
            or getattr(norm._norm, '__func__', None) is not _NATIVE_NORM):
        raise ValueError('Native RMSNorm callable changed')
    verify_class_source()
    if not norm.with_scale or not isinstance(norm.weight, torch.nn.Parameter):
        raise ValueError('Native multiplicative scale required')
    if norm.weight.requires_grad or norm.weight.grad is not None:
        raise ValueError('Measurement is forward-only')
    if norm.weight.ndim != 1 or norm.weight.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError('Dense FP32/BF16 native scale required')
    if hasattr(norm.weight, 'quant_state') or not norm.weight.is_contiguous():
        raise ValueError('Packed or noncontiguous scale refused')
    if not isinstance(norm.eps, (int, float)) or not 0 < norm.eps < 1:
        raise ValueError('Positive native epsilon required')


def check_binding(binding):
    if (not isinstance(binding, tuple) or len(binding) != 4
            or not isinstance(binding[0], str) or not binding[0]
            or type(binding[1]) is not int or binding[1] < 0
            or any(not isinstance(v, str) or len(v) != 64 for v in binding[2:])):
        raise ValueError('Explicit case/step/native-cache/input binding required')


@dataclass(frozen=True)
class NativeNormState:
    binding: tuple
    layer: int
    kind: str
    input_shape: tuple
    raw: torch.Tensor
    reciprocal_RMS: torch.Tensor
    raw_sha256: str
    reciprocal_RMS_sha256: str
    output_sha256: str
    gamma_sha256: str
    mlp_input_sha256: str
    epsilon: float


def compose(norm, raw, reciprocal):
    """Use recorded native factors, the unchanged gamma and native operation order."""
    check_norm(norm)
    if raw.dtype not in (torch.float32, torch.bfloat16) or raw.requires_grad:
        raise ValueError('Detached FP32/BF16 raw native state required')
    if raw.ndim != 3 or raw.shape[:2] != (1, 1) or raw.shape[-1] != norm.weight.numel():
        raise ValueError('One explicit current-position branch required')
    if reciprocal.dtype != torch.float32 or reciprocal.shape != (1, 1, 1) or reciprocal.requires_grad:
        raise ValueError('Detached FP32 native reciprocal RMS required')
    if raw.device != norm.weight.device or reciprocal.device != raw.device:
        raise ValueError('Native factor devices must agree')
    if (not torch.isfinite(raw).all() or not torch.isfinite(reciprocal).all()
            or not torch.all(reciprocal > 0)):
        raise ValueError('Finite raw state and positive native reciprocal RMS required')
    with torch.no_grad():
        return ((raw.float() * reciprocal) * norm.weight.float()).type_as(raw)


@contextmanager
def capture_states(model, layers, binding, kind):
    """Observe native operations; repeated RMS expression must reproduce output bytes."""
    check_binding(binding)
    if kind not in ('native', 'physical_zero') or len(set(layers)) != len(layers) or not layers:
        raise ValueError('Unique explicit sites and native source kind required')
    states, inputs, handles = {}, {}, []
    successful = False
    try:
        for layer in layers:
            if type(layer) is not int or not 24 <= layer <= 34:
                raise ValueError('Declared late native layer required')
            block = model.model.language_model.layers[layer]
            norm = block.post_feedforward_layernorm
            check_norm(norm)
            if block.enable_moe_block:
                raise ValueError('Dense native branch only')

            def mlp_input(module, args, layer=layer):
                if layer in inputs or not args or args[0].requires_grad:
                    raise ValueError('One detached native MLP input required')
                inputs[layer] = tensor_sha(args[0])

            def observed(module, args, output, layer=layer):
                if layer in states or layer not in inputs:
                    raise ValueError('One aligned norm output required')
                source = args[0]
                if source.ndim != 3 or source.shape[0] != 1 or source.requires_grad:
                    raise ValueError('One detached batch with explicit position required')
                raw = source[:, -1:, :].detach().clone()
                with torch.no_grad():
                    reciprocal = torch.pow(raw.float().pow(2).mean(-1, keepdim=True) + module.eps, -0.5)
                reconstructed = compose(module, raw, reciprocal)
                actual = output[:, -1:, :]
                if tensor_sha(reconstructed) != tensor_sha(actual):
                    raise ValueError('Native RMS expression does not reconstruct actual output exactly')
                states[layer] = NativeNormState(binding, layer, kind, tuple(source.shape), raw,
                    reciprocal.detach().clone(), tensor_sha(raw), tensor_sha(reciprocal), tensor_sha(actual),
                    tensor_sha(module.weight), inputs[layer], float(module.eps))

            handles.append(block.mlp.register_forward_pre_hook(mlp_input))
            handles.append(norm.register_forward_hook(observed))
        yield states
        successful = True
    finally:
        for handle in reversed(handles):
            handle.remove()
        if successful and set(states) != set(layers):
            raise ValueError('Missing or duplicated native source capture')


def validate_source(norm, state, binding, layer):
    check_binding(binding)
    check_norm(norm)
    if not isinstance(state, NativeNormState) or state.binding != binding or state.layer != layer:
        raise ValueError('Native-state lineage mismatch')
    if state.epsilon != norm.eps or state.gamma_sha256 != tensor_sha(norm.weight):
        raise ValueError('Native normalization parameters changed')
    if tensor_sha(state.raw) != state.raw_sha256 or tensor_sha(state.reciprocal_RMS) != state.reciprocal_RMS_sha256:
        raise ValueError('Captured native factors changed')
    reconstructed = compose(norm, state.raw, state.reciprocal_RMS)
    if tensor_sha(reconstructed) != state.output_sha256:
        raise ValueError('Source diagonal no longer reconstructs native output')


@contextmanager
def norm_cell(model, layer, binding, native, removed, raw_bit, RMS_bit):
    """Patch one current output with a declared pair of native source factors."""
    if type(raw_bit) is not int or type(RMS_bit) is not int or raw_bit not in (0, 1) or RMS_bit not in (0, 1):
        raise ValueError('Exactly four prescribed native-state cells required')
    norm = model.model.language_model.layers[layer].post_feedforward_layernorm
    validate_source(norm, native, binding, layer)
    if native.kind != 'native':
        raise ValueError('Actual original native source required')
    if removed is None:
        if raw_bit or RMS_bit:
            raise ValueError('Physical factor-zero source required')
        removed = native
    else:
        validate_source(norm, removed, binding, layer)
        if removed.kind != 'physical_zero' or removed.input_shape != native.input_shape:
            raise ValueError('Exact physical source scope required')
        if removed.mlp_input_sha256 != native.mlp_input_sha256:
            raise ValueError('Physical source has different upstream MLP input')
    sources = (native, removed)
    replacement = compose(norm, sources[raw_bit].raw, sources[RMS_bit].reciprocal_RMS)
    receipt = {'calls': 0, 'activation_assignments': 0, 'other_positions_exact': True}

    def patch(module, args, output):
        source = args[0]
        if (tuple(source.shape) != native.input_shape
                or tensor_sha(source[:, -1:, :]) != native.raw_sha256
                or tensor_sha(output[:, -1:, :]) != native.output_sha256):
            raise ValueError('Hybrid recipient differs from original native context')
        if receipt['calls']:
            raise ValueError('One current-position replacement per context required')
        result = output.clone()
        with torch.no_grad():
            result[:, -1:, :].copy_(replacement)
        if tensor_sha(result[:, :-1, :]) != tensor_sha(output[:, :-1, :]):
            raise ValueError('Noncurrent norm positions changed')
        receipt['calls'] += 1
        receipt['activation_assignments'] += 1
        return result

    handle = norm.register_forward_hook(patch)
    successful = False
    try:
        yield receipt
        successful = True
    finally:
        handle.remove()
        if successful and receipt['calls'] != 1:
            raise ValueError('Hybrid context was not used exactly once')
