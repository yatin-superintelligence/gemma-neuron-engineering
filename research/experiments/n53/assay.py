"""Exact native prefix-cache copies; no weights, gradients or learned alignment."""
import copy
import torch
from transformers.cache_utils import DynamicCache
from interpretability.native_cache_branch import fingerprint,_storages
from research.experiments.n37.continuation import decision
from research.experiments.n48.assay import choice_metrics

MODES=('all15_counterfactual_prefix_KV','last_shared_pair13_14_counterfactual_KV','preceding_pair8_9_counterfactual_KV')

def sites(model):
    c=model.config.text_config;entries=c.num_hidden_layers-c.num_kv_shared_layers
    if entries!=15:
        raise ValueError('Pinned native 15-entry sliding/full architecture required')
    if not (c.layer_types[8]==c.layer_types[13]=='sliding_attention' and c.layer_types[9]==c.layer_types[14]=='full_attention'):
        raise ValueError('Exact architectural shared/preceding producer pairs required')
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):raise ValueError('Frozen native parameters required')
    return {'all15_counterfactual_prefix_KV':list(range(15)),'last_shared_pair13_14_counterfactual_KV':[13,14],'preceding_pair8_9_counterfactual_KV':[8,9]}


def forward(model,ids,answers,cache=None):
    c=model.config.text_config;entries=c.num_hidden_layers-c.num_kv_shared_layers
    if cache is None:cache=DynamicCache(config=c)
    with torch.no_grad():value=model(input_ids=ids,past_key_values=cache,use_cache=True,logits_to_keep=1)
    d,w=decision(value,0,entries,str(next(model.parameters()).dtype))
    return {'decision':d,'metrics':choice_metrics(value.logits[0,-1],answers),'cache':w},value.past_key_values


def capture(model,ids,answers,count):
    sites(model);count('native_prefix_source_forwards');record,cache=forward(model,ids,answers)
    witness=fingerprint(cache,15,ids.numel(),next(model.parameters()).dtype)
    return {'record':record,'prefix_fingerprint':witness,'prefix_tokens':ids.numel()},cache


def branch_run(model,ids,answers,recipient,donor,mode,length,count):
    mapping=sites(model);dtype=next(model.parameters()).dtype
    before=fingerprint(recipient,15,length,dtype);other=fingerprint(donor,15,length,dtype);count('cache_clone_constructions');branch=copy.deepcopy(recipient)
    if fingerprint(branch,15,length,dtype)!=before or _storages(branch)&(_storages(recipient)|_storages(donor)):
        raise ValueError('Actual native nonaliasing cache clone required')
    indices=[] if mode=='identity' else mapping[mode]
    try:
        for i in indices:
            for name in ['keys','values']:
                a,b=getattr(branch.layers[i],name),getattr(donor.layers[i],name)
                if a.shape!=b.shape or a.dtype!=b.dtype or a.device!=b.device:raise ValueError('Aligned native source shapes required,no fitted alignment')
                setattr(branch.layers[i],name,b.detach().clone());count('prefix_KV_tensor_assignments')
        edited=fingerprint(branch,15,length,dtype)
        if any(edited['layers'][i]!=(other if i in indices else before)['layers'][i] for i in range(15)):
            raise ValueError('Selected donor and nonselected recipient cache bytes differ')
        if _storages(branch)&(_storages(recipient)|_storages(donor)):raise ValueError('Edited branch aliases native sources')
        record,cache=forward(model,ids,answers,branch)
        fingerprint(cache,15,length+ids.numel(),dtype)
        return record,{'mode':mode,'selected_entries':indices,'K_V_tensor_assignments':2*len(indices),'recipient_prefix_sha256':before['sha256'],'donor_prefix_sha256':other['sha256'],'edited_prefix_sha256':edited['sha256'],'selected_donor_nonselected_recipient_exact':True,'source_nonaliasing':True,'prefix_tokens':length,'suffix_tokens':ids.numel()}
    finally:
        if fingerprint(recipient,15,length,dtype)!=before or fingerprint(donor,15,length,dtype)!=other:raise ValueError('Native source cache mutated')


def native_case(model,full,suffix,answers,cache,length,count):
    count('native_monolithic_guards');full_record,_=forward(model,full,answers)
    count('native_segmented_source_forwards');record,w=branch_run(model,suffix,answers,cache,cache,'identity',length,count)
    if full_record['decision']['argmax']!=record['decision']['argmax'] or full_record['metrics']['choice']!=record['metrics']['choice']:
        raise ValueError('Every native segmented/monolithic full-vocabulary choice must match')
    return {'original':record,'monolithic':full_record,'native_branch':w,'full_logits_hash_equal_descriptive':full_record['decision']['full_logits_sha256']==record['decision']['full_logits_sha256'],'full_cache_hash_equal_descriptive':full_record['decision']['cache_sha256']==record['decision']['cache_sha256']}


def edited_case(model,suffix,answers,recipient,donor,length,own_native,donor_native,count):
    count('identity_suffix_guards');identity,identity_w=branch_run(model,suffix,answers,recipient,recipient,'identity',length,count)
    if identity!=own_native['original']:raise ValueError('Identity cached suffix differs from actual native source')
    records={};restores={}
    for mode in MODES:
        count('edited_suffix_decisions');record,w=branch_run(model,suffix,answers,recipient,donor,mode,length,count)
        if mode==MODES[0] and record!=donor_native['original']:raise ValueError('Whole-prefix control differs from actual donor native source')
        records[mode]={'record':record,'cache_edit':w};count('native_restoration_forwards');restored,restore_w=branch_run(model,suffix,answers,recipient,recipient,'identity',length,count)
        if restored!=own_native['original']:raise ValueError('Actual native cached restoration differs')
        restores[mode]={'record':restored,'cache_branch':restore_w}
    return {'carried_native':own_native,'identity':{'record':identity,'cache_branch':identity_w},'edits':records,'restorations':restores,'all15_positive_control_exact':True,'parameter_assignments':0,'MLP_activation_assignments':0,'backward_calls':0,'optimizer_steps':0,'new_emitted_tokens':0}
