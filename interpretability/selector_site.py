"""Actual donor swaps at an MLP position or a shared attention read boundary."""
from contextlib import contextmanager
import torch


@contextmanager
def selector_site(model,position,donor,kind,mlp_layers=range(14),producer=14):
    """Clone shared K/V for later readers, leaving native producer output/cache intact.

    Native MLP swaps apply only during prefill. Shared K/V swaps reapply on each
    cached step because the native producer rebuilds the shared tuple each time.
    Donors must come from the same absolute position in an unmodified run.
    """
    if kind not in ['K','V','KV','MLP','both']or position<1:raise ValueError('Invalid site')
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):raise ValueError('Freeze parameters')
    layers=model.model.language_model.layers;handles=[]
    versions={n:p._version for n,p in model.named_parameters()}
    receipt={'kind':kind,'position':position,'MLP_prefill_hits':0,'MLP_decode_skips':0,'KV_calls':0,
        'native_cache_values_unchanged':True,'optimizer_steps':0}
    try:
        if kind in ['MLP','both']:
            for index in mlp_layers:
                def hook(module,args,index=index):
                    x=args[0]
                    if x.shape[1]<=position:receipt['MLP_decode_skips']+=1;return None
                    changed=x.clone();value=donor['MLP'][index].to(device=x.device,dtype=x.dtype)
                    assert value.shape==x[:,position].shape
                    changed[:,position]=value;receipt['MLP_prefill_hits']+=1
                    return (changed,*args[1:])
                handles.append(layers[index].mlp.down_proj.register_forward_pre_hook(hook))
        if kind in ['K','V','KV','both']:
            attention=layers[producer].self_attn
            if attention.is_kv_shared_layer or not attention.store_full_length_kv or attention.layer_type!='full_attention':
                raise ValueError('Expected native full-attention producer')
            def kv_hook(module,args,kwargs,output):
                states=kwargs['shared_kv_states'];key,value=states['full_attention']
                before=(key.clone(),value.clone())
                assert key.shape[-2]>position and value.shape[-2]>position
                k=key;v=value
                if kind in ['K','KV','both']:
                    k=key.clone();d=donor['K'].to(device=k.device,dtype=k.dtype)
                    assert d.shape==k[:,:,position:position+1].shape;k[:,:,position:position+1]=d
                if kind in ['V','KV','both']:
                    v=value.clone();d=donor['V'].to(device=v.device,dtype=v.dtype)
                    assert d.shape==v[:,:,position:position+1].shape;v[:,:,position:position+1]=d
                states['full_attention']=(k,v)
                assert torch.equal(before[0],key)and torch.equal(before[1],value)
                assert (k is key or k.data_ptr()!=key.data_ptr())and(v is value or v.data_ptr()!=value.data_ptr())
                receipt['KV_calls']+=1
            handles.append(attention.register_forward_hook(kv_hook,with_kwargs=True))
        yield receipt
    finally:
        for handle in handles:handle.remove()
        assert versions=={n:p._version for n,p in model.named_parameters()}
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        receipt['native_parameter_versions_unchanged']=True


@contextmanager
def capture_selector(model,position,mlp_layers=range(14),producer=14):
    """Capture one position in transient CPU memory. Callers save only hashes."""
    layers=model.model.language_model.layers;handles=[];donor={'MLP':{}}
    try:
        for index in mlp_layers:
            def mlp(module,args,index=index):donor['MLP'][index]=args[0][:,position].detach().cpu().clone()
            handles.append(layers[index].mlp.down_proj.register_forward_pre_hook(mlp))
        def kv(module,args,kwargs,output):
            key,value=kwargs['shared_kv_states']['full_attention']
            donor['K']=key[:,:,position:position+1].detach().cpu().clone()
            donor['V']=value[:,:,position:position+1].detach().cpu().clone()
            donor['producer_output']=output[0].detach().cpu().clone()
        handles.append(layers[producer].self_attn.register_forward_hook(kv,with_kwargs=True))
        yield donor
    finally:
        for h in handles:h.remove()
