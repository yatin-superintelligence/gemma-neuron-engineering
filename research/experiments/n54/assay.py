"""Paired prescribed up/down sign gauge with original Parameter restoration."""
from contextlib import contextmanager
import torch
from transformers.cache_utils import DynamicCache
from research.experiments.n35.core import tensor_sha
from research.experiments.n49.assay import site,observe
from research.experiments.n37.continuation import decision
from research.experiments.n48.assay import choice_metrics

class UniformSignGauge:
    def __init__(self,model,count=lambda name:None):
        block,_=site(model);self.block=block;self.count=count;self.original={k:getattr(block.mlp,k).weight for k in ['up_proj','down_proj']};self.hashes={k:tensor_sha(p) for k,p in self.original.items()};self.versions={k:p._version for k,p in self.original.items()};self.negative={}
        for k,p in self.original.items():
            if p.requires_grad or p.grad is not None:raise ValueError('Frozen native original parameters required')
            count('negative_weight_tensor_negations');value=p.detach().neg();count('negative_weight_Parameter_constructions');self.negative[k]=torch.nn.Parameter(value,requires_grad=False)
        self.negative_hashes={k:tensor_sha(p) for k,p in self.negative.items()};self.gate_hash=tensor_sha(block.mlp.gate_proj.weight)
        if any(self.negative[k].untyped_storage().data_ptr()==p.untyped_storage().data_ptr() for k,p in self.original.items()):raise ValueError('Native and prescribed negative weights alias')
    def witness(self):
        return {'original_parameter_sha256':self.hashes,'original_parameter_versions':self.versions,'negative_parameter_sha256':self.negative_hashes,'negative_parameter_versions':{k:p._version for k,p in self.negative.items()},'gate_parameter_sha256':self.gate_hash,'all_parameter_grad_buffers_absent':all(p.grad is None and not p.requires_grad for p in [*self.original.values(),*self.negative.values()]),'negative_parameter_storage_nonaliasing':True}
    def native_ok(self):
        return all(getattr(self.block.mlp,k).weight is p and tensor_sha(p)==self.hashes[k] and p._version==self.versions[k] and not p.requires_grad and p.grad is None for k,p in self.original.items()) and tensor_sha(self.block.mlp.gate_proj.weight)==self.gate_hash
    @contextmanager
    def installed(self):
        if not self.native_ok():raise ValueError('Actual native parameter references/bytes/versions changed')
        try:
            for k,p in self.negative.items():
                if tensor_sha(p)!=self.negative_hashes[k] or p.requires_grad or p.grad is not None:raise ValueError('Prescribed negative source mutated')
                self.count('manual_parameter_assignments');getattr(self.block.mlp,k).weight=p
            if tensor_sha(self.block.mlp.gate_proj.weight)!=self.gate_hash:raise ValueError('Gate parameter changed')
            yield
        finally:
            for k,p in self.original.items():self.count('manual_parameter_assignments');getattr(self.block.mlp,k).weight=p
            if not self.native_ok():raise ValueError('Actual original parameter references/bytes/versions not restored')

def run(model,encoded,answers):
    slots=model.config.text_config.num_hidden_layers-model.config.text_config.num_kv_shared_layers
    with torch.no_grad(),observe(model) as captured:
        output=model(**encoded,past_key_values=DynamicCache(config=model.config.text_config),use_cache=True,logits_to_keep=1);d,cache=decision(output,0,slots,str(next(model.parameters()).dtype));metrics=choice_metrics(output.logits[0,-1],answers);logits=output.logits[0,-1].detach().float().cpu()
    if not torch.isfinite(captured['h']).all():raise ValueError('Finite observed actual h required')
    return {'decision':d,'activation':captured['activation'],'metrics':metrics},cache,captured['h'],logits

def one_case(model,encoded,answers,parent,gauge,count=lambda name:None):
    caught=None;gauged=None
    try:
        with gauge.installed():
            count('gauged_forwards');gauged,gcache,h,glogits=run(model,encoded,answers);count('auxiliary_measured_h_negations');back=h.detach().neg();observed={'gauged_full_h_sha256':tensor_sha(h),'negative_back_full_h_sha256':tensor_sha(back),'negative_back_current_h_sha256':tensor_sha(back[0,-1]),'actual_gauged_parameter_witness':gauge.witness()}
    except BaseException as error:caught=error
    finally:
        count('native_restoration_forwards');restored,cache,native_h,nlogits=run(model,encoded,answers)
        if restored!=parent['original'] or cache!=parent['native_first_full_cache'] or tensor_sha(native_h)!=parent['native_full_h_sha256'] or not gauge.native_ok():raise ValueError('Actual native full restoration failed')
    if caught is not None:raise caught
    primary={'full_FP32_logits_hash_unchanged':gauged['decision']['full_logits_sha256']==parent['original']['decision']['full_logits_sha256'],'full_KV_cache_unchanged':gcache==cache,'full_h_sign_reversal_exact':observed['negative_back_full_h_sha256']==parent['native_full_h_sha256'],'current_h_sign_reversal_exact':observed['negative_back_current_h_sha256']==parent['original']['activation']['native_h_sha256']}
    diff=(glogits-nlogits).abs()
    return {'parent_binding':parent['binding'],'carried_native':parent['original'],'gauged':gauged,'gauged_cache':gcache,'restored_native':restored,'native_cache':cache,'observed':observed,'primary_components':primary,'joint_primary_pass':all(primary.values()),'full_FP32_logit_delta':{'max_absolute':float(diff.max()),'mean_absolute':float(diff.mean()),'nonzero_logits':int(torch.count_nonzero(diff))},'actual_original_parameter_references_restored':gauge.native_ok(),'parameter_source_witness':gauge.witness(),'new_candidate_grades':0,'new_emitted_tokens':0,'backward_calls':0,'optimizer_steps':0,'activation_value_assignments':0}
