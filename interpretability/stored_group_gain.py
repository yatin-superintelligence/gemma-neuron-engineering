"""Prescribed dense outgoing-column gains across native layers, without fitting."""
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import torch


def tensor_sha(value):
    value=value.detach().contiguous().cpu()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def outside_sha(weight,channels):
    value=weight.detach().cpu().clone();value[:,channels]=0
    return tensor_sha(value)


@contextmanager
def stored_group_gain(model,coordinates,factor):
    """Assign a fixed factor1 or2, preserving exact columns for restoration.

    Coordinates are explicit native (layer,channel) pairs or dictionaries.
    No activity/gradient measurement selects the factor inside this operation.
    Packed matrices, duplicate coordinates and unfrozen parameters are refused.
    Only compact original/edited columns need be saved by a caller.
    """
    if type(factor)is not int or factor not in(1,2):raise ValueError('Prescribed factor1 or2 required')
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):raise ValueError('Freeze all parameters first')
    layers=model.model.language_model.layers
    grouped=defaultdict(list);seen=set()
    for c in coordinates:
        layer,channel=(c['layer'],c['channel'])if isinstance(c,dict)else c
        if type(layer)is not int or type(channel)is not int or not 0<=layer<len(layers):raise ValueError('Invalid native coordinate')
        if(layer,channel)in seen:raise ValueError('Duplicate coordinate')
        seen.add((layer,channel));w=layers[layer].mlp.down_proj.weight
        if w.ndim!=2 or not w.is_floating_point()or hasattr(w,'quant_state'):raise ValueError('Dense floating matrix required')
        if not 0<=channel<w.shape[1]:raise ValueError('Invalid native channel')
        grouped[layer].append(channel)
    if not seen:raise ValueError('Nonempty group required')
    versions={n:p._version for n,p in model.named_parameters()}
    flags={n:p.requires_grad for n,p in model.named_parameters()}
    target_ids={id(layers[i].mlp.down_proj.weight)for i in grouped}
    if len(target_ids)!=len(grouped):raise ValueError('Shared down matrices are unsupported')
    target_names={n for n,p in model.named_parameters()if id(p)in target_ids}
    receipt={'factor':factor,'coordinates':len(seen),'optimizer_steps':0,'layers':[],'global_scope':True}
    saved=[]
    try:
        for layer in sorted(grouped):
            channels=sorted(grouped[layer]);w=layers[layer].mlp.down_proj.weight
            original=w[:,channels].detach().clone();expected=original*factor
            if not torch.isfinite(expected).all():raise ValueError('Nonfinite prescribed columns')
            full_before=tensor_sha(w);outside=outside_sha(w,channels);version=w._version
            item={'layer':layer,'channels':channels,'dtype':str(w.dtype),'source_columns_sha256':tensor_sha(original),'expected_columns_sha256':tensor_sha(expected),'source_matrix_sha256':full_before,'outside_columns_sha256':outside,'parameter_version_before':version}
            saved.append((w,channels,original,version,item))
            with torch.no_grad():w[:,channels]=expected
            assert w._version==version+1 and tensor_sha(w[:,channels])==tensor_sha(expected)
            assert outside_sha(w,channels)==outside
            item.update(edited_columns_exact=True,other_columns_exact=True,edited_matrix_sha256=tensor_sha(w));receipt['layers'].append(item)
        yield receipt
    finally:
        for w,channels,original,version,item in reversed(saved):
            with torch.no_grad():w[:,channels]=original
            assert w._version==version+2 and tensor_sha(w)==item['source_matrix_sha256']
            item.update(original_matrix_restored_exact=True,declared_version_increments=2)
        for n,p in model.named_parameters():assert p._version==versions[n]+(2 if n in target_names and any(id(p)==id(s[0])for s in saved)else 0)
        assert flags=={n:p.requires_grad for n,p in model.named_parameters()}
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        receipt.update(all_other_parameter_versions_unchanged=True,all_parameter_gradients_none=True,all_original_matrices_restored=True)
