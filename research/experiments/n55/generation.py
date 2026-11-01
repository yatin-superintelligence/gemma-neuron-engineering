"""Own-history native-EOS generation with prescribed numerator-only masks."""
from contextlib import contextmanager
import hashlib
import torch
from transformers.cache_utils import DynamicCache
from torch.nn import functional as F
from interpretability.native_rms_factor_v2 import compose
from interpretability.signed_native_mask import indices
from interpretability.cache_proofs import check_cache
from research.experiments.n19.trajectory import cache_witness
from research.experiments.n35.core import tensor_sha
from research.experiments.n49.assay import site

EDITS=('remove_negative','random_negative_count','remove_positive','random_positive_count')


def cache_record(model,cache):
    value=cache_witness(cache)
    check_cache(value,model.config.text_config.num_hidden_layers-model.config.text_config.num_kv_shared_layers,str(next(model.parameters()).dtype))
    if any(v['keys'] is None for v in value['layers']):raise ValueError('All actual native KV producers required')
    return value


@contextmanager
def numerator_policy(model,mode,count):
    if mode not in EDITS:raise ValueError('Fixed numerator policy required')
    block,norm=site(model);pending={};frames=[]
    def capture(module,args):
        if pending or args[0].requires_grad:raise ValueError('One aligned native h required')
        h=args[0][:,-1:,:].detach();selected=indices(h[0,0],mode);edited=h.clone();edited[0,0,selected]=0
        count('extra_down_projection_evaluations')
        raw=F.linear(edited,block.mlp.down_proj.weight,block.mlp.down_proj.bias)
        pending.update(edited_raw=raw,mask={'native_h_sha256':tensor_sha(h),'mask_sha256':hashlib.sha256(str(selected).encode()).hexdigest(),'negative_count':int((h<0).sum()),'positive_count':int((h>0).sum()),'width':h.numel(),'mask_count':len(selected)})
    def replace(module,args,output):
        if not pending:raise ValueError('Missing aligned native h')
        native=args[0][:,-1:,:].detach();q=(native.float().pow(2).mean(-1,keepdim=True)+module.eps).pow(-.5)
        if tensor_sha(compose(module,native,q))!=tensor_sha(output[:,-1:,:]):raise ValueError('Native RMS reconstruction failed')
        replacement=compose(module,pending['edited_raw'],q);result=output.clone();count('activation_assignments');result[:,-1:,:].copy_(replacement)
        other=tensor_sha(result[:,:-1])==tensor_sha(output[:,:-1])
        if not other:raise ValueError('Noncurrent norm output changed')
        frames.append({**pending['mask'],'native_raw_sha256':tensor_sha(native),'native_reciprocal_sha256':tensor_sha(q),'native_norm_output_sha256':tensor_sha(output[:,-1:,:]),'replacement_sha256':tensor_sha(replacement),'other_positions_byte_exact':True});pending.clear()
        return result
    handles=[block.mlp.down_proj.register_forward_pre_hook(capture),norm.register_forward_hook(replace)]
    try:yield frames
    finally:
        for handle in reversed(handles):handle.remove()
        if pending:raise ValueError('Unfinished native RMS operation')


def generate(model,tokenizer,encoded,eos,mode,*,resource_guard,on_decision,count):
    if mode not in EDITS or any(p.requires_grad or p.grad is not None for p in model.parameters()) or any(m.training for m in model.modules()):raise ValueError('Fixed frozen inference policy required')
    c=model.config.text_config;n=int(encoded['input_ids'].numel());maximum=c.max_position_embeddings
    if encoded['input_ids'].shape!=(1,n) or n<1 or type(maximum)is not int or not eos or any(type(t)is not int or not 0<=t<c.vocab_size for t in eos):raise ValueError('Exact single prompt/native boundaries required')
    versions={k:p._version for k,p in model.named_parameters()};cache=DynamicCache(config=c);ids=[];first=last=None;attempts=exposures=0;interruption=None;termination='interrupted'
    try:
        with torch.no_grad(),numerator_policy(model,mode,count)as frames:
            while True:
                interruption=resource_guard()
                if interruption is not None:
                    if not isinstance(interruption,dict)or 'reason'not in interruption or 'evidence'not in interruption:raise ValueError('Actual resource evidence required')
                    break
                step=len(ids)
                if n+step>maximum:interruption={'reason':'context_exhausted','evidence':{'configured_context':maximum,'next_required_input_length':n+step}};break
                args=encoded if step==0 else{'input_ids':torch.tensor([[ids[-1]]],device=encoded['input_ids'].device)}
                attempts+=1;exposures+=n if step==0 else 1
                try:
                    output=model(**args,past_key_values=cache,use_cache=True,logits_to_keep=1);cache=output.past_key_values
                    logits=output.logits[0,-1].detach()
                    if not torch.isfinite(logits).all()or str(logits.dtype)!=str(next(model.parameters()).dtype):raise ValueError('Finite native raw logits required')
                    if len(frames)!=1:raise ValueError('One actual current numerator assignment perforward required')
                    token=int(logits.argmax());event={'step':step,'argmax':token,'full_logits_sha256':tensor_sha(logits.float()),'raw_logits_dtype':str(logits.dtype),'consumed_previous_token':None if step==0 else ids[-1],'consumed_input_length':n+step,'mask':frames.pop()}
                    if step==0:first=cache_record(model,cache)
                except torch.OutOfMemoryError:
                    interruption={'reason':'memory_exhausted','evidence':{'during_forward_or_witness':True}};break
                ids.append(token);count('fresh_choices');on_decision(event)
                if token in eos:termination='eos';interruption=None;break
            if ids:last=cache_record(model,cache)
    finally:
        if versions!={k:p._version for k,p in model.named_parameters()}or any(p.requires_grad or p.grad is not None for p in model.parameters()):raise ValueError('Frozen parameter state changed')
    return {'generated_ids':ids,'text':tokenizer.decode(ids,skip_special_tokens=True),'generated_tokens':len(ids),'fresh_choices':len(ids),'old_cell_carries':0,'first_cache':first,'terminal_cache':last,'complete':termination=='eos','termination':termination,'interruption':interruption,'new_response_token_ceiling':None,'configured_context':maximum,'input_tokens':n,'attempted_forward_calls':attempts,'input_token_exposures':exposures,'per_token_whole_cache_authenticated':False}
