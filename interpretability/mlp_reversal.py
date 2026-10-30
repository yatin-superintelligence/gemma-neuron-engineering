"""Reversible dense MLP coordinate and sign transformations. No training."""
from contextlib import contextmanager
import torch
from interpretability.stored_neuron_gain import tensor_sha

OPERATIONS = {'aligned_channel_reverse', 'uncompensated_channel_reverse', 'sign_flip'}


def _transform(weight, axis):
    with torch.no_grad():
        if axis is None: weight.neg_()
        else: weight.copy_(weight.flip(axis))


@contextmanager
def reverse_native_mlps(model, operation):
    """Apply a fixed involution to every dense native text MLP, then restore.

    Aligned reversal changes hidden channel order and all corresponding
    connections. Uncompensated reversal changes only outgoing column order.
    Sign flip negates every outgoing matrix. These are distinct operations;
    none implements a semantic or logical inverse of the model's knowledge.
    """
    if operation not in OPERATIONS: raise ValueError('Unknown transformation')
    params = dict(model.named_parameters())
    if any(p.requires_grad or p.grad is not None for p in params.values()):
        raise ValueError('Freeze all parameters before direct edits')
    names = {id(p): name for name, p in params.items()}
    targets = []
    layers = model.model.language_model.layers
    for index, layer in enumerate(layers):
        mlp = layer.mlp
        gate, up, down = (getattr(mlp, name).weight for name in ['gate_proj', 'up_proj', 'down_proj'])
        if any(w.ndim != 2 or not w.is_floating_point() or hasattr(w, 'quant_state') for w in [gate, up, down]):
            raise ValueError('Dense floating matrices are required')
        if gate.shape != up.shape or down.shape != (gate.shape[1], gate.shape[0]):
            raise ValueError('Uncoordinated native MLP shape')
        if any(not torch.isfinite(w).all() for w in [gate, up, down]):
            raise ValueError('Nonfinite source weight')
        selected = [(gate, 0), (up, 0), (down, 1)] if operation == 'aligned_channel_reverse' else [(down, 1 if operation == 'uncompensated_channel_reverse' else None)]
        for weight, axis in selected:
            targets.append({'name': names[id(weight)], 'weight': weight, 'axis': axis,
                'layer': index, 'source_sha256': tensor_sha(weight)})
    if len({id(t['weight']) for t in targets}) != len(targets):
        raise ValueError('Shared target parameters require a separate protocol')
    versions = {n: p._version for n, p in params.items()}
    flags = {n: p.requires_grad for n, p in params.items()}
    applied = []; receipts = []
    receipt = {'operation': operation, 'native_layers': len(layers),
        'optimizer_steps': 0, 'actual_backward_passes': 0, 'targets': receipts}
    try:
        for target in targets:
            _transform(target['weight'], target['axis']); applied.append(target)
            receipts.append({k: v for k, v in target.items() if k != 'weight'})
            receipts[-1]['edited_sha256'] = tensor_sha(target['weight'])
        yield receipt
    finally:
        for target in reversed(applied):
            _transform(target['weight'], target['axis'])
            assert tensor_sha(target['weight']) == target['source_sha256']
        selected_names = {t['name'] for t in applied}
        assert all(p._version == versions[n] + (2 if n in selected_names else 0) for n, p in params.items())
        assert flags == {n: p.requires_grad for n, p in params.items()}
        assert all(p.grad is None for p in params.values())
        receipt.update(all_target_hashes_restored_exact=True,
            all_other_parameter_versions_unchanged=True, all_gradient_flags_unchanged=True,
            all_parameter_gradients_none=True)
