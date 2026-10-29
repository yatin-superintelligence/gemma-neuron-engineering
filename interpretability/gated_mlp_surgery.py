"""Local gated-MLP edits and reversible channel steering.

Dense structural edits return a new module and leave the source untouched.
Packed 4-bit weights are explicitly unsupported here. Quantized model work must
dequantize the selected layer deliberately, or use a documented sparse branch.
Exact-arithmetic invariants do not imply bitwise floating-point equivalence.
"""
from contextlib import contextmanager
import torch
from torch import nn


class DenseGatedMLP(nn.Module):
    def __init__(self, gate, up, down, activation):
        super().__init__()
        width, hidden = gate.shape
        if up.shape != gate.shape or down.shape != (hidden, width):
            raise ValueError('Aligned gate/up rows and down columns required')
        self.hidden_size, self.intermediate_size = hidden, width
        self.gate_proj = nn.Linear(hidden, width, bias=False, device=gate.device, dtype=gate.dtype)
        self.up_proj = nn.Linear(hidden, width, bias=False, device=gate.device, dtype=gate.dtype)
        self.down_proj = nn.Linear(width, hidden, bias=False, device=gate.device, dtype=gate.dtype)
        with torch.no_grad():
            self.gate_proj.weight.copy_(gate)
            self.up_proj.weight.copy_(up)
            self.down_proj.weight.copy_(down)
        self.act_fn = activation
        self.requires_grad_(False)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def _weights(mlp):
    projections = [getattr(mlp, name, None) for name in ['gate_proj', 'up_proj', 'down_proj']]
    if any(not isinstance(p, nn.Linear) or p.bias is not None for p in projections):
        raise ValueError('Plain bias-free dense projections required; unpack quantized/adapter weights explicitly')
    weights = [p.weight.detach() for p in projections]
    if any(not w.is_floating_point() or not torch.isfinite(w).all() for w in weights):
        raise ValueError('Finite unpacked floating-point matrices required')
    gate, up, down = weights
    if gate.ndim != 2 or up.shape != gate.shape or down.shape != (gate.shape[1], gate.shape[0]):
        raise ValueError('Inconsistent gated-MLP shapes')
    if len({(w.dtype, w.device) for w in weights}) != 1:
        raise ValueError('Shared dtype and device required')
    if not callable(getattr(mlp, 'act_fn', None)):
        raise ValueError('Explicit gated activation required')
    return gate, up, down


def _indices(values, width, duplicates=False):
    values = list(values)
    if not values or any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < width for x in values):
        raise ValueError('Nonempty valid integer channel indices required')
    if not duplicates and len(set(values)) != len(values):
        raise ValueError('Duplicate channels are ambiguous for this operation')
    return values


def delete_channels(mlp, channels):
    """Physically remove linked gate/up rows and down columns in a dense copy."""
    gate, up, down = _weights(mlp)
    deleted = set(_indices(channels, gate.shape[0]))
    keep = [i for i in range(gate.shape[0]) if i not in deleted]
    if not keep:
        raise ValueError('Retain at least one channel')
    return DenseGatedMLP(gate[keep], up[keep], down[:, keep], mlp.act_fn)


def duplicate_channels(mlp, channels, preserve_function=True):
    """Copy input vectors; optionally split outgoing weights among all replicas.

    preserve_function=False is an amplification control, not new knowledge.
    """
    gate, up, down = _weights(mlp)
    extra = _indices(channels, gate.shape[0], duplicates=True)
    mapping = torch.tensor(list(range(gate.shape[0])) + extra, device=gate.device)
    new_down = down[:, mapping].clone()
    if preserve_function:
        multiplicity = torch.bincount(mapping, minlength=gate.shape[0])
        new_down /= multiplicity[mapping].to(dtype=down.dtype).unsqueeze(0)
    return DenseGatedMLP(gate[mapping], up[mapping], new_down, mlp.act_fn)


def add_channels(mlp, gate_rows, up_rows, down_columns):
    """Add genuinely specified new input/output vectors to a dense layer copy."""
    gate, up, down = _weights(mlp)
    if (gate_rows.ndim != 2 or gate_rows.shape[0] == 0 or gate_rows.shape[1] != gate.shape[1]
            or up_rows.shape != gate_rows.shape or down_columns.shape != (gate.shape[1], gate_rows.shape[0])):
        raise ValueError('New gate/up rows and output columns must have matching dimensions')
    added = [gate_rows, up_rows, down_columns]
    if any(not w.is_floating_point() or not torch.isfinite(w).all() for w in added):
        raise ValueError('Finite new floating-point vectors required')
    gate_rows, up_rows, down_columns = [w.to(device=gate.device, dtype=gate.dtype) for w in added]
    return DenseGatedMLP(torch.cat([gate, gate_rows]), torch.cat([up, up_rows]),
                         torch.cat([down, down_columns], dim=1), mlp.act_fn)


def edit_output_vectors(mlp, channels, deltas):
    """Alter only selected columns that write channel contributions to the stream."""
    gate, up, down = _weights(mlp)
    channels = _indices(channels, gate.shape[0])
    if deltas.shape != (down.shape[0], len(channels)) or not deltas.is_floating_point() or not torch.isfinite(deltas).all():
        raise ValueError('Finite output-vector deltas with exact shape required')
    changed = down.clone()
    changed[:, channels] += deltas.to(device=down.device, dtype=down.dtype)
    return DenseGatedMLP(gate, up, changed, mlp.act_fn)


@contextmanager
def steer_channels(mlp, channels, gain=1.0, offset=0.0, token_scope='all'):
    """Reversible h_j <- gain*h_j + offset before down projection.

    Works through module hooks without editing packed weights. Caller must check
    that the hook ran; optimized/fused execution may bypass module hooks.
    """
    if token_scope not in ['all', 'last']:
        raise ValueError('Choose all or last token in each forward call')
    projection = getattr(mlp, 'down_proj', None)
    if projection is None or not hasattr(projection, 'in_features'):
        raise ValueError('An explicit down projection is required')
    channels = _indices(channels, projection.in_features)
    gain = torch.as_tensor(gain, dtype=torch.float64).reshape(-1)
    offset = torch.as_tensor(offset, dtype=torch.float64).reshape(-1)
    if any(v.numel() not in [1, len(channels)] or not torch.isfinite(v).all() for v in [gain, offset]):
        raise ValueError('Finite scalar or per-channel gain/offset required')
    identity = bool(torch.all(gain == 1) and torch.all(offset == 0))
    receipt = {'calls': 0, 'channels': channels, 'token_scope': token_scope, 'identity': identity}
    def hook(module, arguments):
        receipt['calls'] += 1
        if identity:
            return None
        source = arguments[0]
        if source.ndim != 3:
            raise ValueError('Expected batch/token/channel activations')
        changed = source.clone()
        g, b = [v.to(device=source.device, dtype=source.dtype) for v in [gain, offset]]
        if token_scope == 'all':
            changed[..., channels] = source[..., channels] * g + b
        else:
            changed[:, -1, channels] = source[:, -1, channels] * g + b
        return (changed, *arguments[1:])
    handle = projection.register_forward_pre_hook(hook)
    try:
        yield receipt
    finally:
        handle.remove()
