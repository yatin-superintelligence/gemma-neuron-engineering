"""Actual native late-state swaps and scoped sparse comment-context edits."""

from contextlib import contextmanager, ExitStack
import torch
from interpretability.current_group import current_group
from interpretability.decode_step import frozen_versions
from interpretability.native_cache_branch import tensor_hash


@contextmanager
def full_late(model, donor):
    layers=model.model.language_model.layers; indices=list(range(24,35))
    versions=frozen_versions(model); hits={i:0 for i in indices}; scope=[]; handles=[]; success=False
    try:
        for i in indices:
            state=donor[i]
            if state.shape!=(layers[i].mlp.down_proj.in_features,) or not torch.isfinite(state).all():
                raise ValueError('Actual finite late-layer donor required')
            def hook(module,args,i=i):
                before=args[0]
                if before.shape[:2]!=(1,1) or hits[i]:raise ValueError('One current token per full donor scope')
                hits[i]+=1
                expected=donor[i].to(device=before.device,dtype=before.dtype).view_as(before)
                scope.append({'layer':i,'before_sha256':tensor_hash(before),'after_sha256':tensor_hash(expected),'expected_sha256':tensor_hash(expected)})
                return (expected.clone(),*args[1:])
            handles.append(layers[i].mlp.down_proj.register_forward_pre_hook(hook))
        yield {'hits':hits,'scope':scope,'optimizer_steps':0};success=True
    finally:
        for h in handles:h.remove()
        assert frozen_versions(model)==versions
        if success:assert all(v==1 for v in hits.values())


@contextmanager
def policy(model, op, group, native, opposite, same, offsets):
    with ExitStack() as stack:
        receipts={}
        if group is None:
            if op!='restore':receipts['full']=stack.enter_context(full_late(model,native if op=='full_self' else opposite if op=='full_opposite' else same))
        else:
            if op in ['conditional_self','conditional_cut']:receipts['full']=stack.enter_context(full_late(model,opposite))
            if op!='restore':
                source=opposite if op in ['opposite','conditional_self'] else same if op=='same' else native
                donor={c['flat']:float(source[c['layer']][c['channel']]) for c in group}
                operation='gain' if op in ['zero','gain2'] else 'rescue' if op=='rescue' else 'swap'
                receipts['group']=stack.enter_context(current_group(model,group,operation,donor,gain=0 if op=='zero' else 2 if op=='gain2' else 1))
        yield receipts
