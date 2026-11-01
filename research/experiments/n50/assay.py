"""Native full-h donor transplants with fixed answer-order binding controls."""
from contextlib import contextmanager
import torch
from transformers.cache_utils import DynamicCache
from research.experiments.n35.core import tensor_sha
from research.experiments.n37.continuation import decision
from research.experiments.n48.assay import choice_metrics
from research.experiments.n49.assay import source_forward, site

CONDITIONS=('paired_order_swap','unrelated_same_order','unrelated_opposite_order')
XORS=dict(zip(CONDITIONS,[1,2,3]))


def capture(model, encoded, answer_ids, parent):
    record,cache,actual=source_forward(model,encoded,answer_ids)
    if record!=parent['original'] or cache!=parent['native_first_full_cache']:
        raise ValueError('Native donor capture differs from frozen N48 source')
    current=actual['h'][:, -1:, :].detach().clone()
    return {'record':record,'cache':cache,'full_h_sha256':tensor_sha(actual['h']),
            'current_h':current,'current_h_sha256':tensor_sha(current),'parent_binding':parent['binding']}


def public_capture(source):
    return {k:v for k,v in source.items() if k!='current_h'}


@contextmanager
def transplant(model, recipient, donor, donor_id, assigned):
    block,norm=site(model)
    if (donor['current_h'].requires_grad or donor['current_h'].ndim!=3
            or donor['current_h'].shape[:2]!=(1,1) or tensor_sha(donor['current_h'])!=donor['current_h_sha256']):
        raise ValueError('Intact actual detached native donor required')
    receipt={'calls':0,'activation_assignments':0,'donor_id':donor_id,'donor_h_sha256':donor['current_h_sha256'],
             'native_h_exact':False,'other_positions_byte_exact':False,'assigned_h_sha256':None}
    def patch(module,args):
        value=args[0]
        if (receipt['calls'] or value.requires_grad or tensor_sha(value)!=recipient['full_h_sha256']
                or value.dtype!=donor['current_h'].dtype or value.device!=donor['current_h'].device
                or value.shape[-1]!=donor['current_h'].shape[-1]):
            raise ValueError('Native recipient/donor scope mismatch')
        receipt['native_h_exact']=True;result=value.clone();assigned()
        result[:, -1:, :].copy_(donor['current_h'])
        if tensor_sha(result[:, :-1])!=tensor_sha(value[:, :-1]):raise ValueError('Noncurrent h changed')
        receipt.update(calls=1,activation_assignments=1,other_positions_byte_exact=True,assigned_h_sha256=tensor_sha(result[:, -1:, :]))
        return (result,*args[1:])
    handle=block.mlp.down_proj.register_forward_pre_hook(patch)
    try:
        yield receipt
        if receipt['calls']!=1:raise ValueError('Exactly one donor use required')
    finally:handle.remove()


def one_case(model,encoded,answer_ids,row,sources,rows_by_index,parents,count=lambda name:None):
    own=sources[row['id']];slots=model.config.text_config.num_hidden_layers-model.config.text_config.num_kv_shared_layers
    order=list(CONDITIONS);offset=row['row_index']%3;order=order[offset:]+order[:offset]
    cells={};restorations={}
    for condition in order:
        donor_row=rows_by_index[row['row_index']^XORS[condition]];donor=sources[donor_row['id']]
        with torch.no_grad(),transplant(model,own,donor,donor_row['id'],lambda:count('activation_assignments')) as patch:
            output=model(**encoded,past_key_values=DynamicCache(config=model.config.text_config),use_cache=True,logits_to_keep=1)
            d,cache=decision(output,0,slots,str(next(model.parameters()).dtype));metrics=choice_metrics(output.logits[0,-1],answer_ids)
        if cache!=own['cache']:raise ValueError('Final-h donor changed upstream whole cache')
        cells[condition]={'decision':d,'metrics':metrics,'activation':patch,'donor_binding':donor['parent_binding']}
        count('native_restoration_forwards');restored=capture(model,encoded,answer_ids,parents[row['id']])
        if public_capture(restored)!=public_capture(own):raise ValueError('Actual complete native restoration failed')
        restorations[condition]=public_capture(restored)
    return {'parent_binding':parents[row['id']]['binding'],'carried_original':parents[row['id']]['original'],
            'carried_identity':parents[row['id']]['identity'],'native_source':public_capture(own),
            'cells':cells,'restorations':restorations,'order':order,'answer_token_ids':answer_ids,
            'input_tokens':int(encoded['input_ids'].numel()),'new_emitted_tokens':0,'parameter_assignments':0,'backward_calls':0,'optimizer_steps':0}
