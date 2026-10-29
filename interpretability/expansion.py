"""Explicit new gated neurons and a normalized residual MLP insertion.

Uses a small separate dense branch; never resizes packed quantized matrices.
Zero outgoing vectors preserve the original function at initialization. A
residual MLP insertion is not a new transformer/attention decoder layer.
"""
from contextlib import contextmanager
import math
import torch
from torch import nn
from torch.nn import functional as F


class NewGatedNeurons(nn.Module):
    def __init__(self, hidden_size, width, *, seed=196, normalize=False, device=None):
        super().__init__()
        if not isinstance(hidden_size,int) or not isinstance(width,int) or min(hidden_size,width)<1:
            raise ValueError('Positive integer dimensions required')
        self.hidden_size,self.width,self.normalize=hidden_size,width,bool(normalize)
        # Isolated generator avoids changing global random state.
        generator=torch.Generator(device='cpu').manual_seed(seed)
        self.gate=nn.Parameter(torch.randn(width,hidden_size,generator=generator)/math.sqrt(hidden_size))
        self.up=nn.Parameter(torch.randn(width,hidden_size,generator=generator)/math.sqrt(hidden_size))
        self.down=nn.Parameter(torch.zeros(hidden_size,width))
        self.to(device=device,dtype=torch.float32)

    def forward(self,x):
        if x.shape[-1]!=self.hidden_size:raise ValueError('Residual dimension mismatch')
        z=x.to(dtype=self.gate.dtype)
        if self.normalize:z=z*torch.rsqrt(z.square().mean(dim=-1,keepdim=True)+1e-6)
        return F.linear(F.gelu(F.linear(z,self.gate),approximate='tanh')*F.linear(z,self.up),self.down).to(dtype=x.dtype)


@contextmanager
def insert_branch(module,branch,placement):
    """Restore hook even on failure; output tensor-only interface is deliberate.

    parallel: base_mlp(x)+new_neurons(x), computational width expansion.
    residual: decoder_output+new_MLP(decoder_output), new nonlinear block.
    The caller saves branch config/weights separately from the frozen base.
    """
    if placement not in ['parallel','residual']:raise ValueError('Unknown placement')
    receipt={'placement':placement,'calls':0,'new_parameters':sum(p.numel()for p in branch.parameters())}
    def hook(current,args,output):
        if not isinstance(output,torch.Tensor):raise ValueError('Expected a tensor module output')
        x=args[0] if placement=='parallel' else output
        delta=branch(x)
        if delta.shape!=output.shape or delta.device!=output.device or delta.dtype!=output.dtype:
            raise ValueError('Explicitly matching residual shape,device,dtype required')
        if not torch.isfinite(delta).all():raise ValueError('Nonfinite new branch output')
        receipt['calls']+=1
        return output+delta
    handle=module.register_forward_hook(hook)
    try:yield receipt
    finally:handle.remove()
