"""Measure native activation influence while every parameter stays frozen."""
from contextlib import contextmanager


@contextmanager
def native_gradient_tap(model,layer_indices):
    """Create one activation leaf, then retain later activation gradients.

    Detaching each subsequent site would sever earlier causal paths. No model
    parameter requires gradients or receives an update from this measurement.
    """
    indices=sorted(layer_indices)
    if not indices or len(indices)!=len(set(indices)):raise ValueError('Unique native layers required')
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):raise ValueError('Freeze all parameters')
    layers=model.model.language_model.layers;versions={n:p._version for n,p in model.named_parameters()}
    captured={};handles=[]
    try:
        for index in indices:
            def tap(module,args,index=index):
                x=args[0]
                if index==indices[0]:x=x.detach().clone().requires_grad_(True)
                elif not x.requires_grad:raise RuntimeError('Tracked causal path was detached')
                x.retain_grad();captured[index]=x
                return (x,*args[1:])
            handles.append(layers[index].mlp.down_proj.register_forward_pre_hook(tap))
        yield captured
    finally:
        for h in handles:h.remove()
        assert versions=={n:p._version for n,p in model.named_parameters()}
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
