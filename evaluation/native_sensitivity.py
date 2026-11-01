"""Local gradient-times-activation summaries; these measurements are not updates."""
import hashlib
import math
import random
import torch

EDITS=('remove_negative','random_negative_count','remove_positive','random_positive_count')


def sha_tensor(x):return hashlib.sha256(x.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)


def analyze(h,g,native_choice,native_margin,edits):
    require(h.ndim==g.ndim==1 and h.shape==g.shape and h.numel()>0 and h.dtype==g.dtype and h.dtype in [torch.float32,torch.bfloat16],'Aligned dense native h/VJP vectors required')
    require(not h.requires_grad and not g.requires_grad and torch.isfinite(h).all().item() and torch.isfinite(g).all().item(),'Finite detached observations required')
    require(native_choice in ['A','B'] and type(native_margin) in [int,float] and math.isfinite(native_margin),'Actual canonical native choice/margin required')
    require(set(edits)==set(EDITS),'All four unchanged completed finite masks required')
    h64,g64=h.double(),g.double();raw=h64*g64;support=raw*(1 if native_choice=='A' else -1)
    total_positive=float(support[support>0].sum());total_negative=-float(support[support<0].sum());total_abs=float(support.abs().sum())
    sign_masks={'negative':h<0,'positive':h>0,'zero':h==0};groups={}
    for name,mask in sign_masks.items():
        values=support[mask];positive=float(values[values>0].sum());negative=-float(values[values<0].sum());absolute=float(values.abs().sum())
        groups[name]={'unit_count':int(mask.sum()),'support_units':int((values>0).sum()),'suppression_units':int((values<0).sum()),'zero_contribution_units':int((values==0).sum()),'support_mass':positive,'suppression_mass':negative,'net':float(values.sum()),'absolute_mass':absolute,'positive_support_fraction':positive/total_positive if total_positive else None,'suppression_fraction':negative/total_negative if total_negative else None,'absolute_mass_fraction':absolute/total_abs if total_abs else None}
    finite={};width=h.numel()
    for mode in EDITS:
        old=edits[mode];a=old['activation'];indices=a['indices'];negative='negative' in mode;eligible=torch.nonzero(h<0 if negative else h>0).flatten().tolist()
        if mode.startswith('random_'):
            permutation=list(range(width));random.Random(4501 if negative else 4502).shuffle(permutation);eligible=sorted(permutation[:len(eligible)])
        require(indices==eligible and all(type(i) is int for i in indices),'Exact frozen native-sign or random-cardinality indices required')
        require(a['native_h_sha256']==sha_tensor(h),'Completed mask has different native h source')
        observed=old['metrics']['A_minus_B_margin']-native_margin;require(type(observed) in [int,float] and math.isfinite(observed),'Finite carried output contrast required')
        predicted=-float(raw[indices].sum());error=predicted-observed
        sign=lambda v:1 if v>0 else -1 if v<0 else 0
        finite[mode]={'selected_units':len(indices),'first_order_A_minus_B_delta':predicted,'carried_finite_A_minus_B_delta':observed,'signed_error':error,'absolute_error':abs(error),'predicted_sign':sign(predicted),'observed_sign':sign(observed),'sign_agrees':sign(predicted)==sign(observed)}
    return {'native_choice':native_choice,'width':width,'native_h_sha256':sha_tensor(h),'gradient_sha256':sha_tensor(g),'native_winner_support_groups':groups,'total_support_mass':total_positive,'total_suppression_mass':total_negative,'total_absolute_mass':total_abs,'net_native_winner_h_times_gradient':float(support.sum()),'all_four_carried_finite_comparisons':finite,'no_completeness_or_semantic_claim':True}
