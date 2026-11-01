"""Fixed sign masks at the current final native MLP down-projection input.

The sign is that of the post-gating product, not the gate pre-activation.
Random controls match cardinality only. No parameter or fitted scale is used.
"""
from contextlib import contextmanager
import hashlib
import inspect
import random

import torch
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer, Gemma4TextMLP
from research.experiments.n35.core import tensor_sha

MODES = ('original', 'identity', 'remove_negative', 'random_negative_count', 'remove_positive', 'random_positive_count')
MODULE_SHA = '64ecac478c7d11b9a6993ea194bdf98ce867d0e6d361295d976140e0ded53933'
MLP_FORWARD = Gemma4TextMLP.forward


def indices(value, mode):
    if mode not in MODES or value.ndim != 1 or value.dtype not in (torch.float32, torch.bfloat16) or value.requires_grad or not torch.isfinite(value).all():
        raise ValueError('Finite native one-dimensional inference state and fixed mode required')
    if mode in ('original', 'identity'):
        return []
    negative = 'negative' in mode
    sign = torch.nonzero(value < 0 if negative else value > 0).flatten().tolist()
    if mode.startswith('remove_'):
        return sign
    permutation = list(range(value.numel()))
    random.Random(4501 if negative else 4502).shuffle(permutation)
    return sorted(permutation[:len(sign)])


def transform(value, mode):
    if value.ndim != 3 or value.shape[0] != 1:
        raise ValueError('Batch-one native activation required')
    current = value[0, -1]
    selected = indices(current, mode)
    patched = value.clone()
    if mode not in ('original', 'identity'):
        patched[0, -1, selected] = 0
    if tensor_sha(patched[:, :-1]) != tensor_sha(value[:, :-1]):
        raise ValueError('Another sequence position changed')
    energy = current.double().square()
    removed = float(energy[selected].sum())
    total = float(energy.sum())
    return patched, {'native_h_sha256': tensor_sha(current), 'edited_h_sha256': tensor_sha(patched[0, -1]),
                     'indices': selected, 'mask_sha256': hashlib.sha256(str(selected).encode()).hexdigest(),
                     'negative_count': int((current < 0).sum()), 'positive_count': int((current > 0).sum()),
                     'width': current.numel(), 'removed_h_energy': removed, 'total_h_energy': total,
                     'removed_h_energy_fraction': removed / total if total else None,
                     'other_positions_byte_exact': True}


@contextmanager
def signed_mask(model, mode):
    if mode not in MODES:
        raise ValueError('Fixed mask mode required')
    source = inspect.getsourcefile(Gemma4TextDecoderLayer)
    if hashlib.sha256(open(source, 'rb').read()).hexdigest() != MODULE_SHA:
        raise ValueError('Pinned Gemma4 implementation required')
    blocks = model.model.language_model.layers
    block = blocks[-1]
    if type(block) is not Gemma4TextDecoderLayer or block.layer_idx != len(blocks)-1 or type(block.mlp) is not Gemma4TextMLP or block.enable_moe_block:
        raise ValueError('Final dense native MLP required')
    if type(block.mlp).forward is not MLP_FORWARD or 'forward' in block.mlp.__dict__:
        raise ValueError('Native MLP callable required')
    module = block.mlp.down_proj
    if module._forward_pre_hooks or module._forward_hooks or any(p.grad is not None for p in model.parameters()):
        raise ValueError('Clean hook site and absent parameter gradients required')
    receipt = {'mode': mode, 'site': len(blocks)-1, 'activation_assignments': 0, 'frames': []}
    def hook(module, args):
        if type(block.mlp).forward is not MLP_FORWARD or 'forward' in block.mlp.__dict__:
            raise ValueError('Native MLP callable changed')
        patched, witness = transform(args[0], mode)
        receipt['frames'].append(witness)
        if mode == 'original':
            return None
        receipt['activation_assignments'] += 1
        return (patched, *args[1:])
    handle = module.register_forward_pre_hook(hook)
    try:
        yield receipt
    finally:
        handle.remove()
