"""Fresh own-history native generation with the fixed N60 value connection."""
from contextlib import nullcontext
import base64
import hashlib
import inspect
import random

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.gemma4 import modeling_gemma4 as native_module
from research.experiments.n19.trajectory import cache_witness
from research.experiments.n35.core import tensor_sha,stable
from research.experiments.n59.assay import cache_retained_prefix,validate_growth

POLICIES=('original','remove_BOS_value_link','remove_one_nonBOS_value_link')
NATIVE_EAGER=native_module.eager_attention_forward
EAGER_SHA='7f8506e7461c8fb3a702c2c83535c1cd8e4381ce029ccc8a29bc4e0a37fb293e'
OPERATIONS=('fresh_choices','native_fresh_choices','edited_fresh_choices','native_sourceguards','native_restoration_forwards',
 'edited_attention_callbacks','native_value_BMM_replay_attempts','edited_value_BMM_attempts','scratch_V_clone_attempts',
 'scratch_V_zero_assignments','selected_position_absent_callbacks','current_head_output_assignments',
 'zero_assigned_elements','zero_assigned_bytes','current_output_assigned_elements','current_output_assigned_bytes',
 'global_function_reference_assignments','cached_prefix_observations')

def native_check(model):
    text=model.model.language_model
    if (len(text.layers)!=35 or text.config.num_hidden_layers!=35 or text.config.num_kv_shared_layers!=20
        or native_module.eager_attention_forward is not NATIVE_EAGER
        or hashlib.sha256(inspect.getsource(NATIVE_EAGER).rstrip('\n').encode()).hexdigest()!=EAGER_SHA
        or 'eager'in native_module.ALL_ATTENTION_FUNCTIONS
        or any(b.self_attn.config._attn_implementation!='eager'for b in text.layers)):
        raise ValueError('Unchanged native35/unshadowed eager source reference required')
    return text

def function_witness(model):
    native_check(model)
    return {'native_eager_source_sha256':EAGER_SHA,'actual_global_is_original_reference':True,'eager_registry_shadow_absent':True}

def control_position(length,row_index):
    if type(length)is not int or length<3 or type(row_index)is not int or row_index<0:raise ValueError('New exact case index/length')
    return random.Random(6001+row_index).choice(range(1,length-1))

def noncurrent_prefix(cache):
    if len(cache.layers)!=15:raise ValueError('Original15 native physical KV entries')
    return [{'layer':i,'keys':tensor_sha(v.keys[...,:-1,:]),'values':tensor_sha(v.values[...,:-1,:]),
             'shape':list(v.keys[...,:-1,:].shape),'dtype':str(v.keys.dtype)}for i,v in enumerate(cache.layers)]

def retained_after(cache,before):
    growth=validate_growth(cache,before)
    left=[];right=[];shapes=[]
    for layer,old in zip(cache.layers,before,strict=True):
        if old['empty']:raise ValueError('All15 initialized own native KV entries required')
        n=old['retained_positions'];shapes.append([old['old_seen'],old['old_stored'],n,growth[old['layer']]['new_stored']])
        for key in ('keys','values'):
            a=None if n==0 else tensor_sha(getattr(layer,key)[...,:n,:])
            b=None if n==0 else old[key]['sha256']
            if a!=b:raise ValueError('Own retained native cache prefix changed')
            left.append('0'*64 if b is None else b);right.append('0'*64 if a is None else a)
    pack=lambda xs:base64.b64encode(b''.join(bytes.fromhex(x)for x in xs)).decode()
    return {'before_KV_SHA256_base64':pack(left),'after_KV_SHA256_base64':pack(right),'shape_counts':shapes,
            'both_actual_compared_prefixes_exact':True,'growth_exact':True,'compared_hash_order':'15physical_layers;keys,values;zero hash denotes empty retained prefix'}

class ValueOperator:
    def __init__(self,model,condition,control,count):
        self.model=model;self.condition=condition;self.control=control;self.count=count
        self.modules=[b.self_attn for b in native_check(model).layers]
        if condition not in POLICIES[1:]:raise ValueError('Exact fixed edited policy')
        self.frames=[];self.query_length=None;self.total=None;self.complete=True
    def begin(self,query_length,total):
        if not self.complete:raise ValueError('Previous35 attention callbacks incomplete')
        self.frames=[];self.query_length=query_length;self.total=total;self.complete=False
    def wrapper(self,module,query,key,value,attention_mask,dropout=0.0,scaling=None,softcap=None,**kwargs):
        i=len(self.frames);q=self.query_length
        if (i>=35 or module is not self.modules[i] or module.training or dropout!=0.0
            or query.shape[0]!=1 or query.shape[-2]!=q or key.shape[-2]!=value.shape[-2]
            or self.total is None or value.shape[-2]>self.total):raise ValueError('Actual35 ordered same-call native attention bindings')
        self.count('edited_attention_callbacks')
        output,weights=NATIVE_EAGER(module,query,key,value,attention_mask,dropout=dropout,scaling=scaling,softcap=softcap,**kwargs)
        if (not torch.isfinite(output).all()or not torch.isfinite(weights).all()
            or output.shape!=(1,q,query.shape[1],query.shape[-1])
            or weights.shape!=(1,query.shape[1],q,value.shape[-2])):raise ValueError('Actual finite native full-shape weights/output')
        repeated=native_module.repeat_kv(value,module.num_key_value_groups)
        native_sha=tensor_sha(output);value_sha=tensor_sha(value);weights_sha=tensor_sha(weights)
        self.count('native_value_BMM_replay_attempts')
        replay=torch.matmul(weights,repeated).transpose(1,2).contiguous()
        if replay.dtype!=output.dtype or replay.shape!=output.shape or tensor_sha(replay)!=native_sha:raise ValueError('Native FULLSHAPE value BMM replay differs; whole panel censored')
        selected=0 if self.condition==POLICIES[1]else self.control
        key_start=self.total-value.shape[-2];local=selected-key_start;available=0<=local<value.shape[-2]
        self.count('scratch_V_clone_attempts');scratch=repeated.clone()
        if scratch.untyped_storage().data_ptr()==repeated.untyped_storage().data_ptr():raise ValueError('Scratch V must not alias actual KV')
        if available:
            self.count('scratch_V_zero_assignments');scratch[:,:,local,:]=0
            zero=scratch[:,:,local,:];self.count('zero_assigned_elements',zero.numel());self.count('zero_assigned_bytes',zero.numel()*zero.element_size())
            if torch.count_nonzero(zero):raise ValueError('Literal zero selected scratch value row')
            if tensor_sha(scratch[:,:,:local,:])!=tensor_sha(repeated[:,:,:local,:])or tensor_sha(scratch[:,:,local+1:,:])!=tensor_sha(repeated[:,:,local+1:,:]):raise ValueError('Unselected scratch rows changed')
        else:
            self.count('selected_position_absent_callbacks')
            if tensor_sha(scratch)!=tensor_sha(repeated):raise ValueError('Unavailable native key must not be replaced or reinserted')
        self.count('edited_value_BMM_attempts');edited=torch.matmul(weights,scratch).transpose(1,2).contiguous()
        assigned=output.clone();self.count('current_head_output_assignments');assigned[:,-1,:,:]=edited[:,-1,:,:]
        self.count('current_output_assigned_elements',assigned[:,-1].numel());self.count('current_output_assigned_bytes',assigned[:,-1].numel()*assigned.element_size())
        if tensor_sha(assigned[:,:-1])!=tensor_sha(output[:,:-1])or tensor_sha(value)!=value_sha:raise ValueError('Noncurrent headoutput or actual native V changed')
        self.frames.append({'layer':i,'key_length':value.shape[-2],'key_start':key_start,'selected_absolute':selected,
          'local':local if available else None,'available':available,'query_length':q,'heads':query.shape[1],'head_dim':query.shape[-1],
          'dtype':str(output.dtype),'hashes':[weights_sha,value_sha,native_sha,tensor_sha(assigned[:,-1])],
          'native_replay_exact':True,'actual_V_unchanged':True,'noncurrent_headoutput_exact':True,'unselected_scratch_exact':True})
        return assigned,weights
    def finish(self):
        if len(self.frames)!=35:raise ValueError('Exactly35 callback frames required')
        frames=self.frames;self.complete=True
        packed=base64.b64encode(b''.join(bytes.fromhex(h)for f in frames for h in f['hashes'])).decode()
        return {'hash_order':'weights,unrepeated_V,native_full_head_equal_replay,assigned_current_head',
          'four_SHA256_per_layer_base64':packed,'hash_bytes':35*4*32,'frame_shapes':[[f[k]for k in ('key_length','key_start','selected_absolute','local','available','query_length','heads','head_dim')]for f in frames],
          'dtype':frames[0]['dtype'],'all35_native_fullshape_replays_exact':True,'all35_actualV_unselected_scratch_noncurrent_exact':True}
    def __enter__(self):
        self.count('global_function_reference_assignments');native_module.eager_attention_forward=self.wrapper;return self
    def __exit__(self,*exc):
        self.count('global_function_reference_assignments');native_module.eager_attention_forward=NATIVE_EAGER;native_check(self.model)

def root_forward(model,ids,cache,first=False):
    observed={};handle=None
    if first:
        def h(module,args):
            if observed or len(args)!=1:raise ValueError('One actual lastMLP h input')
            observed['full_h_sha256']=tensor_sha(args[0])
        handle=model.model.language_model.layers[34].mlp.down_proj.register_forward_pre_hook(h)
    try:
        with torch.no_grad():output=model(input_ids=ids,past_key_values=cache,use_cache=True,logits_to_keep=1)
    finally:
        if handle is not None:handle.remove()
    logits=output.logits[0,-1].detach()
    if logits.ndim!=1 or logits.dtype!=next(model.parameters()).dtype or not torch.isfinite(logits).all():raise ValueError('Actual finite full-vocabulary native-dtype logits')
    record={'argmax':int(logits.argmax()),'full_FP32_logits_sha256':tensor_sha(logits.float()),'raw_logits_dtype':str(logits.dtype)}
    if first:
        record.update(observed,cache=cache_witness(output.past_key_values),noncurrent_prefix=noncurrent_prefix(output.past_key_values))
        if 'full_h_sha256'not in record:raise ValueError('Actual fullh observation missing')
    return record,output.past_key_values

def native_first(model,encoded):
    native_check(model)
    record,_=root_forward(model,encoded['input_ids'],DynamicCache(config=model.config.text_config),True)
    return record

def generate(model,tokenizer,encoded,eos,condition,row_index,*,resource_guard,on_decision,count):
    native_check(model)
    ids=encoded['input_ids'];n=ids.shape[-1];maximum=model.config.text_config.max_position_embeddings
    if ids.shape!=(1,n)or n<3 or int(ids[0,0])!=2 or condition not in POLICIES or not eos:raise ValueError('New actual BOS prompt and fixed policy/EOS required')
    if any(p.requires_grad or p.grad is not None for p in model.parameters())or any(m.training for m in model.modules()):raise ValueError('Frozen inference only')
    emitted=[];cache=DynamicCache(config=model.config.text_config);first=None;terminal=None;attempts=exposures=0
    interruption=None;termination='interrupted';control=control_position(n,row_index);chain='0'*64
    operator=None if condition=='original'else ValueOperator(model,condition,control,count)
    with operator if operator is not None else nullcontext():
        while True:
            interruption=resource_guard()
            if interruption is not None:
                if not isinstance(interruption,dict)or set(interruption)!={'reason','evidence'}:raise ValueError('Actual resource interruption evidence')
                break
            step=len(emitted);total=n+step
            if total>maximum:interruption={'reason':'context_exhausted','evidence':{'configured_context':maximum,'next_required_input_length':total}};break
            current=ids if step==0 else torch.tensor([[emitted[-1]]],device=ids.device,dtype=torch.long)
            before=None if step==0 else cache_retained_prefix(cache)
            if operator is not None:operator.begin(current.shape[-1],total)
            attempts+=1;exposures+=current.numel()
            event,cache=root_forward(model,current,cache,step==0)
            prefix=None if before is None else retained_after(cache,before)
            if before is not None:count('cached_prefix_observations')
            callbacks=None if operator is None else operator.finish()
            if step==0:first=event
            token=event['argmax'];event={k:v for k,v in event.items()if k not in ('cache','noncurrent_prefix','full_h_sha256')}
            event.update(step=step,consumed_previous_token=None if step==0 else emitted[-1],actual_input_length=current.numel(),total_consumed_length=total,
                         own_input_sha256=tensor_sha(current),callbacks=callbacks,retained_own_prefix=prefix,previous_event_sha256=chain)
            chain=stable(event);event['event_sha256']=chain
            emitted.append(token);count('fresh_choices');count('native_fresh_choices'if condition=='original'else'edited_fresh_choices');on_decision(event)
            if token in eos:termination='eos';interruption=None;break
        if emitted:terminal=cache_witness(cache)
    native_check(model)
    return {'condition':condition,'generated_ids':emitted,'text':tokenizer.decode(emitted,skip_special_tokens=True),'generated_tokens':len(emitted),
      'fresh_choices':len(emitted),'old_cell_carries':0,'first_native_or_edited':first,'terminal_cache':terminal,'complete':termination=='eos','termination':termination,
      'interruption':interruption,'new_response_token_ceiling':None,'configured_context':maximum,'input_tokens':n,'control_absolute_position':control,
      'attempted_forward_calls':attempts,'input_token_exposures':exposures,'own_choice_tokens_consumed':max(0,len(emitted)-1),
      'last_event_sha256':chain,'per_token_whole_cache_authenticated':False,'full_vocabulary_argmax':True}
