"""Actual pretrained block copies with silent and FFN-only internal controls."""
from contextlib import contextmanager
import copy
import torch
from transformers.cache_utils import DynamicCache
from research.experiments.n35.core import tensor_sha,stable
from research.experiments.n49.assay import site,source_forward
from research.experiments.n37.continuation import decision
from research.experiments.n48.assay import choice_metrics

CONDITIONS=('silent_copied_block','FFN_only_copied_block','full_copied_block')


def module_state(module):
    return {'parameters':{n:{'sha256':tensor_sha(p),'version':p._version,'shape':list(p.shape),'dtype':str(p.dtype),'requires_grad':p.requires_grad,'grad_absent':p.grad is None}for n,p in module.named_parameters()},'buffers':{n:{'sha256':tensor_sha(p),'version':p._version,'shape':list(p.shape),'dtype':str(p.dtype)}for n,p in module.named_buffers()}}


def shared_state(mapping):
    if set(mapping)!={'sliding_attention','full_attention'}:raise ValueError('Actual shared attention-type KV source required')
    return {k:[{'sha256':tensor_sha(t),'shape':list(t.shape),'dtype':str(t.dtype)}for t in pair]for k,pair in mapping.items()}


class CopyPool:
    """Copies remain outside the base model's registered module tree."""
    def __init__(self,model,count=lambda name:None):
        block,_=site(model)
        if not block.hidden_size_per_layer_input or not block.self_attn.is_kv_shared_layer or block.self_attn.store_full_length_kv:raise ValueError('Actual late shared-KV native attention+FFN+PLE block required')
        if any(m._forward_pre_hooks or m._forward_hooks for m in block.modules()):raise ValueError('Clean native copied source required')
        self.block=block;self.count=count;self.source=module_state(block);self.base_ids={n:id(p)for n,p in model.named_parameters()};self.clones={};self.handles=[]
        for condition in CONDITIONS:
            count('cloned_block_constructions');clone=copy.deepcopy(block).eval().requires_grad_(False)
            if module_state(clone)['parameters'].keys()!=self.source['parameters'].keys():raise ValueError('Exact native clone structure required')
            native=dict(block.named_parameters());nativebuffers=dict(block.named_buffers())
            for n,p in clone.named_parameters():
                count('cloned_parameter_objects')
                if tensor_sha(p)!=self.source['parameters'][n]['sha256']or p.untyped_storage().data_ptr()==native[n].untyped_storage().data_ptr():raise ValueError('Byte-exact nonaliasing native Parameter copy required')
            for n,b in clone.named_buffers():
                count('cloned_buffer_objects')
                if tensor_sha(b)!=self.source['buffers'][n]['sha256']or b.untyped_storage().data_ptr()==nativebuffers[n].untyped_storage().data_ptr():raise ValueError('Byte-exact nonaliasing native buffer copy required')
            norms=[]
            if condition=='silent_copied_block':norms=[clone.post_attention_layernorm,clone.post_feedforward_layernorm,clone.post_per_layer_input_norm]
            elif condition=='FFN_only_copied_block':norms=[clone.post_attention_layernorm,clone.post_per_layer_input_norm]
            with torch.no_grad():
                for norm in norms:count('manual_clone_parameter_zero_assignments');norm.weight.zero_()
                if condition=='silent_copied_block':count('manual_clone_buffer_one_assignments');clone.layer_scalar.fill_(1)
            self.clones[condition]=clone
            for module,name in [(clone,'extra_full_block_evaluations'),(clone.self_attn,'extra_attention_evaluations'),(clone.mlp,'extra_FFN_evaluations'),(clone.per_layer_projection,'extra_PLE_evaluations')]:
                self.handles.append(module.register_forward_pre_hook(lambda module,args,name=name:count(name)))
        storages=[p.untyped_storage().data_ptr()for clone in self.clones.values()for p in clone.parameters()]
        if len(storages)!=len(set(storages)):raise ValueError('Different clone Parameters alias')
        self.prepared={k:module_state(v)for k,v in self.clones.items()};self.source_witness=self.witness()
        if self.base_ids!={n:id(p)for n,p in model.named_parameters()}:raise ValueError('Copy registered into native model')
    def witness(self):
        return {'native_source':module_state(self.block),'clones':{k:module_state(v)for k,v in self.clones.items()},'clone_Parameter_elements':sum(p.numel()for v in self.clones.values()for p in v.parameters()),'clone_Parameter_bytes':sum(p.numel()*p.element_size()for v in self.clones.values()for p in v.parameters()),'clone_buffer_bytes':sum(b.numel()*b.element_size()for v in self.clones.values()for b in v.buffers()),'all_clone_grad_absent':all(not p.requires_grad and p.grad is None for v in self.clones.values()for p in v.parameters()),'source_and_clones_nonaliasing':True}
    def check(self,model):
        if self.base_ids!={n:id(p)for n,p in model.named_parameters()}or self.witness()!=self.source_witness:raise ValueError('Actual base references or cloned source bytes/versions changed')
        return True
    def remove(self):
        for handle in self.handles:handle.remove()


@contextmanager
def composed(model,pool,condition):
    if condition not in CONDITIONS:raise ValueError('Fixed copied-block condition required')
    if not pool.check(model):raise ValueError('Intact actual copies required')
    seen={}
    def hook(module,args,kwargs,output):
        if seen or output.requires_grad or output.ndim!=3 or output.shape[0]!=1:raise ValueError('Exactly one native final-block output required')
        if len(args)!=2 or 'hidden_states'in kwargs or 'per_layer_input'in kwargs:raise ValueError('Pinned native positional hidden/PLE decoder invocation required')
        actual=dict(kwargs);actual['per_layer_input']=args[1];shared_before=shared_state(actual['shared_kv_states']);input_sha=tensor_sha(output);source_input=output.clone();copied=pool.clones[condition](hidden_states=source_input,**actual)
        if not torch.isfinite(copied).all()or tensor_sha(output)!=input_sha or shared_state(actual['shared_kv_states'])!=shared_before:raise ValueError('Copied block changed native recipient or shared KV source')
        result=output.clone();pool.count('current_output_assignments');result[:,-1:,:].copy_(copied[:,-1:,:]);other=tensor_sha(result[:,:-1])==tensor_sha(output[:,:-1])
        if not other:raise ValueError('Noncurrent final-block output changed')
        seen.update(condition=condition,calls=1,native_full_block_output_sha256=input_sha,copied_full_input_sha256=input_sha,copied_full_output_sha256=tensor_sha(copied),assigned_current_output_sha256=tensor_sha(copied[:,-1:,:]),silent_full_input_output_byte_exact=tensor_sha(copied)==input_sha,other_positions_byte_exact=True,shared_KV_before_after_exact=True,actual_shared_KV=shared_before,source_pool_sha256=stable(pool.source_witness),clone_attention_KV_shared_read_only=True)
        return result
    handle=pool.block.register_forward_hook(hook,with_kwargs=True)
    try:
        yield seen
        if seen.get('calls')!=1:raise ValueError('Missing copied block operation')
    finally:handle.remove()


def native(model,encoded,answers,parent):
    record,cache,captured=source_forward(model,encoded,answers)
    if record!=parent['original']or cache!=parent['native_first_full_cache']:raise ValueError('Actual original N48 source changed')
    return {'record':record,'cache':cache,'full_h_sha256':tensor_sha(captured['h'])}


def order(row_index):
    k=(row_index//2)%3
    return list(CONDITIONS[k:]+CONDITIONS[:k])


def one_case(model,encoded,answers,parent,row_index,pool,count=lambda name:None):
    count('native_sourceguards');source=native(model,encoded,answers,parent);cells={};restores={};slots=model.config.text_config.num_hidden_layers-model.config.text_config.num_kv_shared_layers
    for condition in order(row_index):
        with torch.no_grad(),composed(model,pool,condition)as witness:
            output=model(**encoded,past_key_values=DynamicCache(config=model.config.text_config),use_cache=True,logits_to_keep=1);d,cache=decision(output,0,slots,str(next(model.parameters()).dtype));metrics=choice_metrics(output.logits[0,-1],answers)
        if cache!=source['cache']:raise ValueError('Extra late shared-KV block changed whole native cache')
        cells[condition]={'decision':d,'metrics':metrics,'composition':witness,'silent_logits_exact':d['full_logits_sha256']==source['record']['decision']['full_logits_sha256'],'cache_exact':True}
        count('native_restoration_forwards');restored=native(model,encoded,answers,parent)
        if restored!=source or not pool.check(model):raise ValueError('Actual full native/copy restoration failed')
        restores[condition]=restored
    silent=cells['silent_copied_block'];primary={'full_input_output_byte_identity':silent['composition']['silent_full_input_output_byte_exact'],'full_FP32_logits_unchanged':silent['silent_logits_exact'],'whole_KV_cache_unchanged':silent['cache_exact']}
    return {'parent_binding':parent['binding'],'carried_native':parent['original'],'native_source':source,'cells':cells,'native_restorations':restores,'condition_order':order(row_index),'silent_primary_components':primary,'joint_silent_primary_pass':all(primary.values()),'copy_pool_sha256':stable(pool.source_witness),'new_emitted_tokens':0,'base_parameter_assignments':0,'base_buffer_assignments':0,'backward_calls':0,'optimizer_steps':0}
