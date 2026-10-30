"""Actual whole-neuron donor states at one declared inference stage."""
from contextlib import contextmanager
import torch
from interpretability.decode_step import frozen_versions
from interpretability.scoped_channel import bit_equal
from interpretability.cache_proofs import check_cache
from research.experiments.n19.trajectory import tensor_witness,cache_witness


def cache_stage_witness(cache):
    full=cache_witness(cache);prior=[]
    for index,layer in enumerate(cache.layers):
        item={'layer':index}
        for key in ['keys','values']:
            value=getattr(layer,key,None)
            if not torch.is_tensor(value)or value.ndim!=4:raise ValueError('Complete native cache tensors required')
            item[key]=tensor_witness(value[:,:,:-1,:])
        prior.append(item)
    return {'full':full,'prior_tokens':prior}


def check_stage_cache_paths(edited,native,producer_count,unchanged_slots,dtype):
    check_cache(edited['full'],producer_count,dtype);check_cache(native['full'],producer_count,dtype)
    assert 1<=unchanged_slots<=producer_count
    assert edited['prior_tokens']==native['prior_tokens']
    assert edited['full']['layers'][:unchanged_slots]==native['full']['layers'][:unchanged_slots]


@contextmanager
def native_stage_site(model,indices,stage,position,donor):
    """Patch one full h vector per native layer with actual unmodified values.

    Current mode requires a one-token shape. The caller must separately create
    and validate a nonempty cache; shape alone does not prove cache presence.
    The context covers exactly one target forward and ends before continuation.
    Donor provenance is established by the caller's native capture hashes.
    """
    versions=frozen_versions(model);flags={n:p.requires_grad for n,p in model.named_parameters()}
    layers=model.model.language_model.layers;indices=list(indices)
    if not indices or len(set(indices))!=len(indices)or any(type(i)is not int or not 0<=i<len(layers)for i in indices):raise ValueError('Unique native layers required')
    if stage not in {'end','current'}or type(position)is not int or position<0 or(stage=='current'and position!=0):raise ValueError('Declared end/current position required')
    if set(donor)!=set(indices):raise ValueError('Exact donor layer set required')
    for i in indices:
        value=donor[i];width=layers[i].mlp.down_proj.in_features
        if not torch.is_tensor(value)or tuple(value.shape)not in[(1,width),(1,1,width)]or not torch.isfinite(value).all():raise ValueError('Native finite donor width required')
    receipt={'stage':stage,'position':position,'layers':indices,'hits':{i:0 for i in indices},'scope':[],'optimizer_steps':0};handles=[];success=False
    try:
        for index in indices:
            def swap(module,args,index=index):
                before=args[0]
                if before.ndim!=3 or before.shape[0]!=1:raise ValueError('Single native sequence required')
                if stage=='current'and before.shape[1]!=1:raise ValueError('Current mode refuses prefill')
                if stage=='end'and before.shape[1]!=position+1:raise ValueError('End position must be actual final input')
                if receipt['hits'][index]:raise ValueError('One target forward per context required')
                receipt['hits'][index]+=1
                if donor[index].dtype!=before.dtype:raise ValueError('Donor dtype must match native h exactly')
                expected=donor[index].detach().to(before.device).reshape(1,before.shape[-1]);after=before.clone();after[:,position,:]=expected
                outside=after.clone();outside[:,position,:]=before[:,position,:]
                assert bit_equal(outside,before)and bit_equal(after[:,position,:],expected)
                receipt['scope'].append({'layer':index,'input_shape':list(before.shape),'dtype':str(before.dtype),'before_target':tensor_witness(before[:,position,:]),'actual_donor':tensor_witness(expected),'after_target':tensor_witness(after[:,position,:]),'other_positions_bit_exact':True})
                return(after,*args[1:])
            handles.append(layers[index].mlp.down_proj.register_forward_pre_hook(swap))
        yield receipt;success=True
    finally:
        for handle in handles:handle.remove()
        assert versions=={n:p._version for n,p in model.named_parameters()}
        assert flags=={n:p.requires_grad for n,p in model.named_parameters()}and all(not p.requires_grad and p.grad is None for p in model.parameters())
        if success:assert all(v==1 for v in receipt['hits'].values())
        receipt.update(original_parameter_versions_unchanged=True,all_gradient_flags_unchanged=True,all_parameter_gradients_none=True,hooks_removed=True)
