"""Capture native gated MLP scalars at one prompt position without attention hooks."""
from contextlib import contextmanager


@contextmanager
def capture_native_mlp(model,position,layers):
    """Keep only selected native positions in transient host memory.

    No weights, gradients, attention states or cached values are changed.
    Callers choose whether to retain compact hashes rather than the values.
    """
    native=model.model.language_model.layers;indices=list(layers)
    if position<0 or not indices or len(set(indices))!=len(indices):raise ValueError('Invalid capture scope')
    if any(not isinstance(i,int)or not 0<=i<len(native)for i in indices):raise ValueError('Invalid layer')
    handles=[];states={'MLP':{}}
    try:
        for index in indices:
            def hook(module,args,index=index):
                value=args[0]
                if value.ndim!=3 or value.shape[1]<=position:raise ValueError('Capture requires prompt prefill')
                states['MLP'][index]=value[:,position].detach().cpu().clone()
            handles.append(native[index].mlp.down_proj.register_forward_pre_hook(hook))
        yield states
    finally:
        for handle in handles:handle.remove()
