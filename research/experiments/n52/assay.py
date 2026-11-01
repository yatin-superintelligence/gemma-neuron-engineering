"""Read-only native final-h VJP; no gradient is ever applied as an update."""
from contextlib import contextmanager
import torch
from transformers.cache_utils import DynamicCache
from interpretability.signed_native_mask import transform
from research.experiments.n35.core import tensor_sha
from research.experiments.n37.continuation import decision
from research.experiments.n48.assay import choice_metrics
from research.experiments.n49.assay import site,source_forward


@contextmanager
def native_leaf(model,count):
    block,norm=site(model);saved={}
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError('Frozen parameters and absent parameter gradients required')
    def hook(module,args):
        source=args[0]
        if saved or source.requires_grad:raise ValueError('One actual unmodified upstream native h required')
        _,activation=transform(source,'original')
        before=tensor_sha(source);leaf=source.detach().requires_grad_(True)
        if tensor_sha(leaf)!=before:raise ValueError('Autograd leaf changed activation values')
        count('same_value_input_leaf_substitutions');count('activation_assignments')
        saved.update(leaf=leaf,activation=activation,full_h_sha256=before,version_before=leaf._version)
        return (leaf,*args[1:])
    handle=block.mlp.down_proj.register_forward_pre_hook(hook)
    try:
        yield saved
        if set(saved)!={'leaf','activation','full_h_sha256','version_before'}:raise ValueError('Missing native measurement leaf')
    finally:handle.remove()


def one_case(model,encoded,answer_ids,parent,count=lambda name:None):
    slots=model.config.text_config.num_hidden_layers-model.config.text_config.num_kv_shared_layers
    with torch.enable_grad(),native_leaf(model,count) as captured:
        output=model(**encoded,past_key_values=DynamicCache(config=model.config.text_config),use_cache=True,logits_to_keep=1)
        d,cache=decision(output,0,slots,str(next(model.parameters()).dtype));metrics=choice_metrics(output.logits[0,-1],answer_ids)
        native={'decision':d,'activation':captured['activation'],'metrics':metrics}
        if native!=parent['original'] or cache!=parent['native_first_full_cache'] or captured['full_h_sha256']!=parent['native_full_h_sha256']:
            raise ValueError('Gradient forward native values/cache/h differ from frozen source')
        leaf=captured['leaf'];target=output.logits[0,-1,answer_ids['A']].float()-output.logits[0,-1,answer_ids['B']].float()
        if not target.requires_grad or float(target.detach())!=metrics['A_minus_B_margin']:
            raise ValueError('Native A-minus-B measurement target required')
        count('autograd_grad_API_attempts');count('measurement_backward_engine_requests')
        gradient,=torch.autograd.grad(target,leaf,create_graph=False,retain_graph=False,allow_unused=False)
        count('autograd_grad_API_returns')
        if (gradient.shape!=leaf.shape or gradient.dtype!=leaf.dtype or gradient.requires_grad
                or not torch.isfinite(gradient).all() or torch.count_nonzero(gradient[:,:-1,:]).item()!=0):
            raise ValueError('Finite native-dtype current-only local VJP required')
        if (leaf.grad is not None or tensor_sha(leaf)!=captured['full_h_sha256'] or leaf._version!=captured['version_before']
                or any(p.requires_grad or p.grad is not None for p in model.parameters())):
            raise ValueError('Read-only measurement changed values/versions or accumulated parameter gradients')
        native_h=leaf[0,-1].detach().clone();current_gradient=gradient[0,-1].detach().clone()
        observation={'target':'native_A_minus_B_logits','target_value':float(target.detach()),
            'native_h_sha256':tensor_sha(native_h),'gradient_sha256':tensor_sha(current_gradient),
            'full_h_sha256_before':captured['full_h_sha256'],'full_h_sha256_after':tensor_sha(leaf),
            'full_gradient_sha256':tensor_sha(gradient),'full_shape':list(leaf.shape),'gradient_dtype':str(gradient.dtype),
            'leaf_version_before':captured['version_before'],'leaf_version_after':leaf._version,
            'leaf_requires_grad':leaf.requires_grad,'gradient_requires_grad':gradient.requires_grad,
            'leaf_grad_buffer_absent':leaf.grad is None,'all_parameter_grad_buffers_absent':all(p.grad is None for p in model.parameters()),
            'noncurrent_gradient_all_zero':True,'activation_values_unchanged':True}
    count('native_restoration_forwards');restored,restored_cache,restored_state=source_forward(model,encoded,answer_ids)
    if restored!=parent['original'] or restored_cache!=cache or tensor_sha(restored_state['h'])!=parent['native_full_h_sha256']:
        raise ValueError('Actual native full restoration failed')
    return {'parent_binding':parent['binding'],'native_value_guard':native,'restored_native':restored,
            'native_first_full_cache':cache,'gradient_observation':observation,'answer_token_ids':answer_ids,
            'input_tokens':int(encoded['input_ids'].numel()),'new_emitted_tokens':0,'parameter_assignments':0,
            'gradient_value_updates':0,'optimizer_steps':0,'Tensor_backward_method_calls':0},native_h,current_gradient
