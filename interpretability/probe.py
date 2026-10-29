"""PyTorch MLP-channel measurement and reversible interventions.

Use ordinary eager Transformers/PEFT execution. Some optimized kernels bypass
module hooks; finish() fails if any expected hook was not observed.
"""
from contextlib import contextmanager
import re

import torch


def mlp_modules(model):
    candidates={name:module for name,module in model.named_modules()
                if name.endswith('.mlp.down_proj')}
    # Multimodal Gemma also has vision MLPs with the same projection names.
    # Prefer the explicitly named text decoder; text-only models need no prefix.
    text={name:module for name,module in candidates.items()
          if re.search(r'(^|\.)language_model\.layers\.\d+\.mlp\.down_proj$',name)}
    return text if text else candidates


class ActivationRecorder:
    def __init__(self,model):
        self.modules=mlp_modules(model)
        if not self.modules:raise ValueError('No MLP down projections found')
        self.handles=[];self.stats={};self.mask=None

    def set_mask(self,attention_mask):
        self.mask=attention_mask

    def _hook(self,name):
        def capture(module,args):
            values=args[0].detach().float()
            if values.ndim!=3:raise ValueError('Require [batch, tokens, channels] eager activations')
            if self.mask is None or tuple(self.mask.shape)!=tuple(values.shape[:2]):
                raise ValueError('Provide a matching token mask; free-generation traces are not aligned probes')
            selected=values[self.mask.to(device=values.device,dtype=torch.bool)]
            if not len(selected):raise ValueError('Probe contains no unmasked tokens')
            count=selected.shape[0]
            if name not in self.stats:
                self.stats[name]={'count':0,'sum':torch.zeros(values.shape[-1],device=values.device),
                    'sum_abs':torch.zeros(values.shape[-1],device=values.device),
                    'sum_sq':torch.zeros(values.shape[-1],device=values.device)}
            item=self.stats[name];item['count']+=count
            item['sum']+=selected.sum(dim=0)
            item['sum_abs']+=selected.abs().sum(dim=0)
            item['sum_sq']+=(selected*selected).sum(dim=0)
        return capture

    def __enter__(self):
        self.handles=[module.register_forward_pre_hook(self._hook(name)) for name,module in self.modules.items()]
        return self

    def finish(self):
        missing=set(self.modules)-set(self.stats)
        if missing:raise RuntimeError('Unobserved hooks; optimized execution may bypass instrumentation: '+str(sorted(missing)))
        result={}
        for name,item in self.stats.items():
            if any(not torch.isfinite(item[key]).all().item() for key in ('sum','sum_abs','sum_sq')):
                raise RuntimeError('Nonfinite activation statistics at '+name)
            n=item['count'];mean=item['sum']/n
            result[name]={'token_count':n,'mean':mean.cpu(),
                'mean_abs':(item['sum_abs']/n).cpu(),
                'rms':(item['sum_sq']/n).sqrt().cpu(),
                'variance':(item['sum_sq']/n-mean*mean).clamp_min(0).cpu()}
        return result

    def __exit__(self,*args):
        for handle in self.handles:handle.remove()
        self.handles=[]


@contextmanager
def scale_mlp_channels(model,layer,channels,scale):
    """Multiply post-gating MLP channels before down_proj, then restore hooks.

    This does not prune dense matrix multiplies and makes no speed guarantee.
    It affects all token positions during the context's lifetime.
    """
    if not 0<=scale<=2:raise ValueError('Initial intervention range is [0,2]')
    targets=[(name,module) for name,module in mlp_modules(model).items()
             if re.search(rf'layers\.{int(layer)}\.mlp\.down_proj$',name)]
    if len(targets)!=1:raise ValueError('Layer must identify exactly one MLP')
    indices=sorted(set(int(j) for j in channels))
    if not indices or min(indices)<0:raise ValueError('Nonempty nonnegative channel IDs required')
    def alter(module,args):
        source=args[0]
        if max(indices)>=source.shape[-1]:raise ValueError('Channel ID outside layer width')
        changed=source.clone();changed[...,indices]*=scale
        return (changed,*args[1:])
    handle=targets[0][1].register_forward_pre_hook(alter)
    try:yield
    finally:handle.remove()


@contextmanager
def select_adapter_modules(model,selected_names,mode='keep'):
    """Keep or drop selected LoRA modules while leaving base weights intact."""
    if mode not in ('keep','drop'):raise ValueError('mode must be keep or drop')
    modules={name:module for name,module in model.named_modules()
             if hasattr(module,'lora_A') and callable(getattr(module,'enable_adapters',None))}
    chosen=set(selected_names)
    if not modules or not chosen or chosen-set(modules):
        raise ValueError('Select existing, nonempty LoRA module names')
    previous={name:bool(module.disable_adapters) for name,module in modules.items()}
    try:
        for name,module in modules.items():
            enabled=(name in chosen) if mode=='keep' else (name not in chosen)
            module.enable_adapters(enabled and not previous[name])
        yield
    finally:
        for name,module in modules.items():module.enable_adapters(not previous[name])
