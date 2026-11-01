"""Native R/P contribution cuts with the recipient FFN vector held fixed."""
from contextlib import contextmanager
import torch
from transformers.cache_utils import DynamicCache
from interpretability.signed_native_mask import signed_mask
from research.experiments.n35.core import tensor_sha
from research.experiments.n37.continuation import decision
from research.experiments.n48.assay import choice_metrics
from research.experiments.n49.assay import site

DONORS=('paired_order_swap','unrelated_same_order')
XORS=dict(zip(DONORS,[1,2]))
CELLS=('10','01','11')


def combine(block,r,f,p):
    if any(x.requires_grad or x.shape!=f.shape or x.dtype!=f.dtype or x.device!=f.device for x in [r,f,p]):
        raise ValueError('Aligned detached actual native components required')
    if f.ndim!=3 or f.shape[:2]!=(1,1) or not all(torch.isfinite(x).all() for x in [r,f,p]):
        raise ValueError('Finite single-current native components required')
    value=(r+f)+p
    value*=block.layer_scalar
    return value


@contextmanager
def observe(model,count):
    block,norm=site(model)
    if not block.hidden_size_per_layer_input or block.post_per_layer_input_norm._forward_hooks or block.pre_feedforward_layernorm._forward_pre_hooks or block._forward_hooks:
        raise ValueError('Native per-layer input path and clean observation sites required')
    saved={};handles=[]
    def residual(module,args):
        if 'R' in saved:raise ValueError('Duplicate native residual capture')
        saved['R']=args[0][:,-1:,:].detach().clone()
    def observed(name):
        def hook(module,args,output):
            if name in saved:raise ValueError('Duplicate native component capture')
            saved[name]=output[:,-1:,:].detach().clone()
        return hook
    def final(module,args,output):
        if set(saved)!={'R','F','P'}:raise ValueError('Missing aligned native additive component')
        count('native_component_reconstruction_checks')
        rebuilt=combine(module,saved['R'],saved['F'],saved['P'])
        if tensor_sha(rebuilt)!=tensor_sha(output[:,-1:,:]):raise ValueError('Native R+F+P/scalar operator order differs')
        saved['output']=output[:,-1:,:].detach().clone();saved['shape']=list(output.shape)
    try:
        handles.append(block.pre_feedforward_layernorm.register_forward_pre_hook(residual))
        handles.append(norm.register_forward_hook(observed('F')))
        handles.append(block.post_per_layer_input_norm.register_forward_hook(observed('P')))
        handles.append(block.register_forward_hook(final))
        yield saved
        if set(saved)!={'R','F','P','output','shape'}:raise ValueError('Incomplete native component observation')
    finally:
        for h in reversed(handles):h.remove()


def native_forward(model,encoded,answer_ids,count=lambda name:None):
    slots=model.config.text_config.num_hidden_layers-model.config.text_config.num_kv_shared_layers
    with torch.no_grad(),observe(model,count) as actual,signed_mask(model,'original') as h:
        out=model(**encoded,past_key_values=DynamicCache(config=model.config.text_config),use_cache=True,logits_to_keep=1)
        d,cache=decision(out,0,slots,str(next(model.parameters()).dtype));metrics=choice_metrics(out.logits[0,-1],answer_ids)
    if len(h['frames'])!=1 or h['activation_assignments']!=0:raise ValueError('Exactly one unmodified native h witness required')
    record={'decision':d,'activation':h['frames'][0],'metrics':metrics}
    return {'record':record,'cache':cache,'vectors':{k:actual[k] for k in ['R','F','P','output']},
            'component_sha256':{k:tensor_sha(actual[k]) for k in ['R','F','P','output']},'shape':actual['shape'],
            'native_scalar_sha256':tensor_sha(model.model.language_model.layers[-1].layer_scalar)}


def capture(model,encoded,answer_ids,parent,count=lambda name:None):
    actual=native_forward(model,encoded,answer_ids,count)
    if (actual['record']!=parent['original'] or actual['cache']!=parent['native_first_full_cache']
            or actual['component_sha256']['F']!=parent['native_F_sha256']):
        raise ValueError('Frozen N48 native decision/cache/h or N49 native F source differs')
    actual['parent_binding']=parent['binding'];return actual


def public_capture(source):return {k:v for k,v in source.items() if k!='vectors'}


@contextmanager
def patch(model,own,donor,donor_id,cell,current,count):
    if cell not in CELLS:raise ValueError('Only the three new native-factor cells are eligible')
    block=model.model.language_model.layers[-1]
    for source in [own,donor]:
        if any(tensor_sha(source['vectors'][k])!=source['component_sha256'][k] for k in ['R','F','P','output']):
            raise ValueError('Actual native source bytes changed')
    r=(donor if cell[0]=='1' else own);p=(donor if cell[1]=='1' else own)
    count('hybrid_component_constructions');replacement=combine(block,r['vectors']['R'],own['vectors']['F'],p['vectors']['P'])
    receipt={'calls':0,'activation_assignments':0,'donor_id':donor_id,'cell':cell,
        'R_sha256':r['component_sha256']['R'],'F_sha256':own['component_sha256']['F'],'P_sha256':p['component_sha256']['P'],
        'replacement_sha256':tensor_sha(replacement),'all_native_internal_components_exact':False,'other_positions_byte_exact':False}
    def replace(module,args,output):
        if (receipt['calls'] or list(output.shape)!=own['shape'] or tensor_sha(output[:,-1:,:])!=own['component_sha256']['output']
                or set(current)!={'R','F','P','output','shape'}
                or any(tensor_sha(current[k])!=own['component_sha256'][k] for k in ['R','F','P','output'])
                or tensor_sha(module.layer_scalar)!=own['native_scalar_sha256']):
            raise ValueError('Native recipient internal components/output/scalar differ')
        result=output.clone();count('activation_assignments');result[:,-1:,:].copy_(replacement)
        if tensor_sha(result[:,:-1])!=tensor_sha(output[:,:-1]):raise ValueError('Noncurrent block output changed')
        receipt.update(calls=1,activation_assignments=1,all_native_internal_components_exact=True,other_positions_byte_exact=True)
        return result
    handle=block.register_forward_hook(replace)
    try:
        yield receipt
        if receipt['calls']!=1:raise ValueError('Exactly one factor output replacement required')
    finally:handle.remove()


def one_case(model,encoded,answer_ids,row,sources,by_index,parents,count=lambda name:None):
    own=sources[row['id']];slots=model.config.text_config.num_hidden_layers-model.config.text_config.num_kv_shared_layers
    work=[(d,c) for d in DONORS for c in CELLS];offset=row['row_index']%len(work);work=work[offset:]+work[:offset]
    cells={d:{} for d in DONORS};restorations={d:{} for d in DONORS}
    for donor_kind,cell in work:
        donor_row=by_index[row['row_index']^XORS[donor_kind]];donor=sources[donor_row['id']]
        with torch.no_grad(),observe(model,count) as current,signed_mask(model,'original') as h:
            with patch(model,own,donor,donor_row['id'],cell,current,count) as receipt:
                out=model(**encoded,past_key_values=DynamicCache(config=model.config.text_config),use_cache=True,logits_to_keep=1)
                d,cache=decision(out,0,slots,str(next(model.parameters()).dtype));metrics=choice_metrics(out.logits[0,-1],answer_ids)
        if cache!=own['cache'] or h['frames']!=[own['record']['activation']]:raise ValueError('Native h or upstream whole cache changed')
        cells[donor_kind][cell]={'decision':d,'metrics':metrics,'activation':receipt,'donor_binding':donor['parent_binding']}
        count('native_restoration_forwards');restored=capture(model,encoded,answer_ids,parents[row['id']],count)
        if public_capture(restored)!=public_capture(own):raise ValueError('Actual complete native R/F/P restoration differs')
        restorations[donor_kind][cell]=public_capture(restored)
    return {'parent_binding':parents[row['id']]['binding'],'carried_native':parents[row['id']]['original'],
            'native_source':public_capture(own),'cells':cells,'restorations':restorations,
            'order':[list(v) for v in work],'answer_token_ids':answer_ids,'input_tokens':int(encoded['input_ids'].numel()),
            'new_emitted_tokens':0,'parameter_assignments':0,'backward_calls':0,'optimizer_steps':0}
