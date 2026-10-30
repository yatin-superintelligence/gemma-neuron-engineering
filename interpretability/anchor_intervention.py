"""Prescribed native-channel edits at a mapped prompt anchor or every position."""
from contextlib import contextmanager
import math
import torch


@contextmanager
def anchor_intervention(model, coordinates, anchor, operation='gain', gain=1,
                        scope='anchor', donor=None):
    """Apply fixed gains or actual cached donor values. No fitting or updates.

    Anchor-only policies apply during full prompt prefill and skip one-token
    cached decoding. The caller must start each sequence with a fresh cache.
    A rescue restores exact cached recipient values after zero assignment;
    this validates targeting and is not by itself a semantic explanation.
    """
    if operation not in ['gain','swap','rescue'] or scope not in ['anchor','all']:
        raise ValueError('Unknown operation or position scope')
    if not isinstance(anchor,int)or anchor<1 or not math.isfinite(gain):
        raise ValueError('A positive mapped anchor and finite gain are required')
    if scope=='all'and operation!='gain':
        raise ValueError('Donor values are only defined at the mapped anchor')
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError('Freeze the model before intervening')
    layers=model.model.language_model.layers
    grouped={}
    seen=set()
    for c in coordinates:
        key=(c['layer'],c['channel'])
        if key in seen or not 0<=key[0]<len(layers)or not 0<=key[1]<layers[key[0]].mlp.down_proj.in_features:
            raise ValueError('Unique valid native coordinates are required')
        seen.add(key);grouped.setdefault(key[0],[]).append(c)
    if not grouped:raise ValueError('Empty coordinate group')
    versions={n:p._version for n,p in model.named_parameters()}
    handles=[]
    receipt={'operation':operation,'scope':scope,'gain':gain,'coordinates':len(seen),
        'anchor_hits':0,'cached_anchor_skips':0,'all_position_calls':0,'optimizer_steps':0}
    try:
        for layer,cs in grouped.items():
            channels=[c['channel']for c in cs]
            values=None
            if operation!='gain':
                if donor is None:raise ValueError('Actual donor values are required')
                values=[donor[c['flat']]for c in cs]
                if not all(math.isfinite(v)for v in values):raise ValueError('Nonfinite donor')
            def hook(module,args,channels=channels,values=values):
                activation=args[0]
                if scope=='anchor'and activation.shape[1]<=anchor:
                    receipt['cached_anchor_skips']+=1
                    return None
                changed=activation.clone()
                if scope=='all':
                    changed[:,:,channels]*=gain;receipt['all_position_calls']+=1
                else:
                    receipt['anchor_hits']+=1
                    if operation=='gain':changed[:,anchor,channels]*=gain
                    else:
                        if operation=='rescue':changed[:,anchor,channels]=0
                        changed[:,anchor,channels]=torch.tensor(values,device=changed.device,dtype=changed.dtype)
                return (changed,*args[1:])
            handles.append(layers[layer].mlp.down_proj.register_forward_pre_hook(hook))
        yield receipt
    finally:
        for handle in handles:handle.remove()
        assert versions=={n:p._version for n,p in model.named_parameters()}
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        receipt['original_parameter_versions_unchanged']=True
