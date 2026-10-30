"""Observe or edit native channels during exactly one cached token forward."""
from contextlib import contextmanager
import math
import torch
from interpretability.scoped_channel import bit_equal,digest


def frozen_versions(model):
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):raise ValueError('Freeze all model parameters before measurement')
    return {n:p._version for n,p in model.named_parameters()}


@contextmanager
def capture_decode_step(model,layer_indices,measurement_gradients=False):
    """Keep native values and, optionally, graph references for one current step.

    Gradient references are activation tensors only. Call autograd.grad inside
    the context; never apply a gradient to a parameter or an activation.
    Prompt-prefill inputs and repeated forwards are rejected.
    """
    versions=frozen_versions(model);native=model.model.language_model.layers;indices=list(layer_indices)
    if not indices or len(indices)!=len(set(indices))or any(not 0<=i<len(native)for i in indices):raise ValueError('Unique valid layers required')
    result={'values':{},'graph':{},'hits':{i:0 for i in indices},'optimizer_steps':0};handles=[];success=False
    try:
        for index in indices:
            def capture(module,args,index=index):
                value=args[0]
                if value.ndim!=3 or value.shape[:2]!=(1,1):raise ValueError('Exactly one cached token required')
                if result['hits'][index]:raise ValueError('Current-step context permits one forward only')
                result['hits'][index]+=1
                if measurement_gradients:
                    if not torch.is_grad_enabled():raise ValueError('Enable activation-gradient measurement explicitly')
                    if not value.requires_grad:value.requires_grad_(True)
                    result['graph'][index]=value
                result['values'][index]=value.detach().cpu().clone()
            handles.append(native[index].mlp.down_proj.register_forward_pre_hook(capture))
        yield result;success=True
    finally:
        for handle in handles:handle.remove()
        assert versions=={n:p._version for n,p in model.named_parameters()}and all(not p.requires_grad and p.grad is None for p in model.parameters())
        if success:assert all(n==1 for n in result['hits'].values())
        result['native_parameter_versions_unchanged']=True


@contextmanager
def edit_decode_step(model,coordinates,operation='gain',gain=1,donor=None):
    """Apply prescribed scalar values once, without changing native weights."""
    versions=frozen_versions(model);native=model.model.language_model.layers;groups={};seen=set()
    if operation not in['gain','swap','rescue']or not math.isfinite(gain):raise ValueError('Unknown operation or nonfinite fixed gain')
    for c in coordinates:
        layer,channel=c['layer'],c['channel']
        if (layer,channel)in seen or not 0<=layer<len(native)or not 0<=channel<native[layer].mlp.down_proj.in_features:raise ValueError('Unique native coordinates required')
        seen.add((layer,channel));groups.setdefault(layer,[]).append(c)
    if not groups:raise ValueError('Empty channel group')
    receipt={'operation':operation,'gain':gain,'coordinates':len(seen),'hits':{layer:0 for layer in groups},'scope':[],'optimizer_steps':0};handles=[];success=False
    try:
        for layer,cs in groups.items():
            channels=[c['channel']for c in cs];values=None
            if operation!='gain':
                if donor is None:raise ValueError('Actual unmodified donor values required')
                values=[donor[c['flat']]for c in cs]
                if not all(math.isfinite(v)for v in values):raise ValueError('Nonfinite donor')
            def change(module,args,layer=layer,channels=channels,values=values):
                before=args[0]
                if before.ndim!=3 or before.shape[:2]!=(1,1):raise ValueError('Exactly one cached token required')
                if receipt['hits'][layer]:raise ValueError('Current-step context permits one forward only')
                receipt['hits'][layer]+=1;after=before.clone()
                expected=before[:,:,channels]*gain if operation=='gain'else torch.tensor(values,device=before.device,dtype=before.dtype).view(1,1,-1)
                if operation=='rescue':after[:,:,channels]=0
                after[:,:,channels]=expected
                outside=after.clone();outside[:,:,channels]=before[:,:,channels]
                assert bit_equal(outside,before)and bit_equal(after[:,:,channels],expected)
                receipt['scope'].append({'layer':layer,'other_values_bit_exact':True,'before_target_sha256':digest(before[:,:,channels]),'after_target_sha256':digest(after[:,:,channels]),'expected_target_sha256':digest(expected)})
                return(after,*args[1:])
            handles.append(native[layer].mlp.down_proj.register_forward_pre_hook(change))
        yield receipt;success=True
    finally:
        for handle in handles:handle.remove()
        assert versions=={n:p._version for n,p in model.named_parameters()}and all(not p.requires_grad and p.grad is None for p in model.parameters())
        if success:assert all(n==1 for n in receipt['hits'].values())
        receipt['native_parameter_versions_unchanged']=True
