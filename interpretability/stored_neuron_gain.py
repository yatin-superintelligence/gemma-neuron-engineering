"""Prescribed dense native-column assignments, with exact restoration."""
from contextlib import contextmanager
import hashlib
import torch


def tensor_sha(value):
    return hashlib.sha256(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


@contextmanager
def stored_neuron_gain(model,layer,channel,factor):
    """Assign only one outgoing column. No optimizer, fitting or gradient update.

    Parameter version increments are expected for the approved assignment and
    restoration. They are distinguished from all other unchanged parameters.
    Packed quantized tensors are refused.
    """
    if factor not in [0,1,2,'zero_then_original']:raise ValueError('Only prescribed factors are supported')
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):raise ValueError('Freeze parameters first')
    layers=model.model.language_model.layers
    if not isinstance(layer,int)or not 0<=layer<len(layers):raise ValueError('Invalid native layer')
    weight=layers[layer].mlp.down_proj.weight
    if weight.ndim!=2 or not weight.is_floating_point()or hasattr(weight,'quant_state'):raise ValueError('A dense floating matrix is required')
    if not isinstance(channel,int)or not 0<=channel<weight.shape[1]:raise ValueError('Invalid native coordinate')
    others={n:p._version for n,p in model.named_parameters()if p is not weight}
    version=weight._version;original=weight.detach().clone();column=original[:,channel].clone()
    expected=column if factor=='zero_then_original'else column*factor
    assert torch.isfinite(expected).all()
    receipt={'layer':layer,'channel':channel,'factor':factor,'optimizer_steps':0,
        'source_column_sha256':tensor_sha(column),'expected_column_sha256':tensor_sha(expected),
        'source_matrix_sha256':tensor_sha(original),'target_parameter_version_before':version}
    try:
        with torch.no_grad():
            if factor=='zero_then_original':weight[:,channel].zero_();weight[:,channel].copy_(column)
            else:weight[:,channel].copy_(expected)
        assert torch.equal(weight[:,channel],expected)
        assert torch.equal(weight[:,:channel],original[:,:channel])and torch.equal(weight[:,channel+1:],original[:,channel+1:])
        receipt.update(edited_column_sha256=tensor_sha(weight[:,channel]),edited_matrix_sha256=tensor_sha(weight),
            other_columns_exact=True,target_parameter_version_during=weight._version)
        yield receipt
    finally:
        with torch.no_grad():weight[:,channel].copy_(column)
        assert torch.equal(weight,original)
        assert others=={n:p._version for n,p in model.named_parameters()if p is not weight}
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        receipt.update(original_matrix_restored_exact=True,restored_matrix_sha256=tensor_sha(weight),
            all_other_parameter_versions_unchanged=True,all_parameter_gradients_none=True,
            target_parameter_version_after=weight._version)
