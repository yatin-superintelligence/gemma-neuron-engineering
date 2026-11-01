"""One fixed extra native FFN computation before the final per-layer-input branch.

Tied pretrained modules are reused. No parameter is assigned or trained.
This is an FFN residual insertion, not a full transformer-layer insertion.
"""
from contextlib import contextmanager
import hashlib
import inspect

import torch
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4RMSNorm, Gemma4TextDecoderLayer, Gemma4TextMLP,
)

MODULE_SHA = '64ecac478c7d11b9a6993ea194bdf98ce867d0e6d361295d976140e0ded53933'
_NATIVE_FORWARDS = {Gemma4RMSNorm: Gemma4RMSNorm.forward, Gemma4TextMLP: Gemma4TextMLP.forward}
_NATIVE_RMS_INNER = Gemma4RMSNorm._norm


def _bytes_equal(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype and a.device == b.device
            and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                            b.contiguous().reshape(-1).view(torch.uint8)))


@contextmanager
def reuse_final_ffn(model, mode):
    """Install identity, parallel_repeat or serial_repeat; remove every own hook."""
    if mode not in ('identity', 'parallel_repeat', 'serial_repeat'):
        raise ValueError('Unknown fixed FFN reuse condition')
    source = inspect.getsourcefile(Gemma4TextDecoderLayer)
    with open(source, 'rb') as stream:
        if hashlib.sha256(stream.read()).hexdigest() != MODULE_SHA:
            raise ValueError('Pinned Gemma4 implementation required')
    blocks = list(model.model.language_model.layers)
    block = blocks[-1]
    if (type(block) is not Gemma4TextDecoderLayer or block.layer_idx != len(blocks) - 1
            or block.enable_moe_block or not block.hidden_size_per_layer_input
            or type(block.mlp) is not Gemma4TextMLP
            or type(block.pre_feedforward_layernorm) is not Gemma4RMSNorm
            or type(block.post_feedforward_layernorm) is not Gemma4RMSNorm):
        raise ValueError('Final dense native FFN with per-layer input required')
    modules = [block.pre_feedforward_layernorm, block.mlp, block.post_feedforward_layernorm]
    if any(m._forward_hooks or m._forward_pre_hooks for m in modules + [block.per_layer_input_gate]):
        raise ValueError('Unscoped hooks on the native FFN site')
    original = [(m, _NATIVE_FORWARDS[type(m)], m.__dict__.get('forward')) for m in modules]
    if any(type(m).forward is not fn for m, fn, _ in original) or Gemma4RMSNorm._norm is not _NATIVE_RMS_INNER:
        raise ValueError('Pinned native class callables required')
    if any(custom is not None for _, _, custom in original):
        raise ValueError('Native callable identities required')
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('Parameter gradients must be absent')
    frame = {}; handles = []; busy = False
    receipt = {'mode': mode, 'site': len(blocks) - 1, 'extra_FFN_evaluations': 0,
               'activation_assignments': 0, 'native_frames': 0,
               'parameter_assignments': 0, 'optimizer_steps': 0}

    def native_input(module, args):
        if busy:
            return
        if frame:
            raise ValueError('Previous native FFN frame unfinished')
        x = args[0]
        if (x.ndim != 3 or x.shape[0] != 1 or x.shape[-1] != block.hidden_size
                or x.dtype not in (torch.float32, torch.bfloat16) or x.requires_grad):
            raise ValueError('Native batch-one FP32/BF16 inference required')
        frame['x'] = x.detach().clone()

    def native_output(module, args, output):
        if busy:
            return
        if set(frame) != {'x'}:
            raise ValueError('Unexpected native FFN call order')
        frame['delta'] = output.detach().clone()

    def insert(module, args):
        nonlocal busy
        if busy or set(frame) != {'x', 'delta'}:
            raise ValueError('Missing native FFN frame or reentrant insertion')
        if Gemma4RMSNorm._norm is not _NATIVE_RMS_INNER:
            raise ValueError('Native RMS callable changed')
        for m, forward, custom in original:
            if type(m).forward is not forward or m.__dict__.get('forward') is not custom:
                raise ValueError('Native FFN callable changed')
        y = args[0]
        x, delta = frame['x'], frame['delta']
        if not _bytes_equal(y, x + delta):
            raise ValueError('Insertion is not immediately after native FFN residual')
        if not torch.isfinite(y).all():
            raise ValueError('Nonfinite native residual')
        busy = True
        try:
            value = y if mode == 'serial_repeat' else x
            extra = block.post_feedforward_layernorm(block.mlp(block.pre_feedforward_layernorm(value)))
            receipt['extra_FFN_evaluations'] += 1
            if not torch.isfinite(extra).all():
                raise ValueError('Nonfinite extra native FFN output')
            if mode != 'serial_repeat' and not _bytes_equal(extra, delta):
                raise ValueError('Parallel native FFN recomputation differs')
            # The decoder saved this tensor as its PLE residual before the hook.
            # Assigning the declared update in-place keeps BOTH residual and gate input aligned.
            if mode != 'identity':
                y.copy_(y + extra)
            result = y
            receipt['native_frames'] += 1
            if mode != 'identity':
                receipt['activation_assignments'] += 1
            frame.clear()
            return (result,)
        finally:
            busy = False

    try:
        handles.append(block.pre_feedforward_layernorm.register_forward_pre_hook(native_input))
        handles.append(block.post_feedforward_layernorm.register_forward_hook(native_output))
        handles.append(block.per_layer_input_gate.register_forward_pre_hook(insert))
        yield receipt
        if frame or busy:
            raise ValueError('Unfinished FFN insertion at scope exit')
    finally:
        for handle in handles:
            handle.remove()
        frame.clear()
