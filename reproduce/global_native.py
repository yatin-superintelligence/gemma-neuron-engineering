"""Exact permanent native-column recipes,without training or full base copies."""
from pathlib import Path
import json
import torch
from safetensors.torch import load_file
from reproduce.model_edits import load_n02_base,file_sha256,tensor_sha256


def apply_native_column_recipe(model,folder,condition):
    """Validate all source columns before assigning the selected saved variant.

    The caller supplies a frozen dense parent. Only declared down columns change.
    Failure after an assignment restores the original matrices before raising.
    This is a weight reconstruction utility,not an independent inference replay.
    """
    folder=Path(folder);recipe=json.loads((folder/'recipe.json').read_text())
    if recipe['schema_version']!=1 or recipe['kind']!='native_column_patch'or recipe['optimizer_steps']!=0:raise ValueError('Unsupported native recipe')
    choices={c['name']:c for c in recipe['conditions']}
    if condition not in choices:raise ValueError('Unknown fixed condition')
    for name,expected in recipe['files'].items():
        path=(folder/name).resolve()
        if not path.is_relative_to(folder.resolve())or file_sha256(path)!=expected:raise ValueError('Bundle hash mismatch')
    params=dict(model.named_parameters())
    if any(p.requires_grad or p.grad is not None for p in params.values()):raise ValueError('Frozen parent required')
    versions={n:p._version for n,p in params.items()};layers=model.model.language_model.layers
    tensors=load_file(str(folder/'native_columns.safetensors'));expected_keys=set();seen=set()
    for item in recipe['coordinates']:
        layer,channel=item['layer'],item['channel'];key=f'L{layer}_C{channel}'
        if(layer,channel)in seen:raise ValueError('Duplicate native coordinate')
        seen.add((layer,channel));w=layers[layer].mlp.down_proj.weight
        if hasattr(w,'quant_state')or w.dtype!=torch.bfloat16:raise ValueError('Dense native BF16 required')
        original,doubled=tensors[key+'_original'],tensors[key+'_double'];expected_keys.update([key+'_original',key+'_double'])
        if list(original.shape)!=item['shape']or original.dtype!=torch.bfloat16 or doubled.dtype!=torch.bfloat16 or not torch.isfinite(original).all()or not torch.equal(original*2,doubled):raise ValueError('Invalid saved variants')
        if tensor_sha256(original)!=item['original_sha256']or tensor_sha256(doubled)!=item['double_sha256']:raise ValueError('Variant tensor hash mismatch')
        if not torch.equal(w[:,channel].detach().cpu(),original):raise ValueError('Original native column mismatch')
    if set(tensors)!=expected_keys:raise ValueError('Unexpected tensor keys')
    for layer,expected in recipe['source_down_matrix_sha256'].items():
        if tensor_sha256(layers[int(layer)].mlp.down_proj.weight)!=expected:raise ValueError('Original matrix mismatch')
    chosen=choices[condition]
    if type(chosen['factor'])is not int or chosen['factor']not in(1,2):raise ValueError('Prescribed integer factor1/2 required')
    group=recipe['groups'][chosen['group']]if chosen['group']is not None else[];by_layer={}
    for c in group:
        if(c['layer'],c['channel'])not in seen:raise ValueError('Group outside validated basis')
        by_layer.setdefault(c['layer'],[]).append(c['channel'])
    if any(len(v)!=len(set(v))for v in by_layer.values()):raise ValueError('Duplicate group coordinates')
    touched=[]
    try:
        for layer,channels in sorted(by_layer.items()):
            channels=sorted(channels);w=layers[layer].mlp.down_proj.weight;original=w[:,channels].detach().clone();touched.append((w,channels,original))
            variant='double'if chosen['factor']==2 else'original'if chosen['factor']==1 else None
            if variant is None:raise ValueError('Only prescribed factor1/2 variants')
            values=torch.stack([tensors[f'L{layer}_C{ch}_{variant}']for ch in channels],dim=1).to(w.device)
            with torch.no_grad():w[:,channels]=values
        actual={layer:tensor_sha256(layers[int(layer)].mlp.down_proj.weight)for layer in recipe['source_down_matrix_sha256']}
        if actual!=recipe['expected_down_matrix_sha256'][condition]:raise ValueError('Edited matrix bytes differ from GPU witness')
        target_ids={id(w)for w,_,_ in touched}
        assert all(p._version==versions[n]+int(id(p)in target_ids)for n,p in params.items())
        assert all(not p.requires_grad and p.grad is None for p in params.values())
        return {'condition':condition,'edited_down_matrix_sha256':actual,'all_expected_projection_hashes_exact':True,'optimizer_steps':0,'independent_clean_package_inference_replay':False}
    except BaseException:
        for w,channels,original in reversed(touched):
            with torch.no_grad():w[:,channels]=original
        assert all(tensor_sha256(layers[int(layer)].mlp.down_proj.weight)==expected for layer,expected in recipe['source_down_matrix_sha256'].items())
        raise


def reconstruct_n21(folder,condition,device='cuda:0',verify_source=True):
    """Load the pinned N02 BF16 parent and apply one validated N21 weight recipe."""
    folder=Path(folder);recipe=json.loads((folder/'recipe.json').read_text())
    if recipe.get('study')!='N21':raise ValueError('Expected N21 bundle')
    model,source=load_n02_base(folder/recipe['base_source_bundle'],device=device,verify_source=verify_source)
    if source['base']['checkpoint']!=recipe['parent']or source['base']['revision']!=recipe['revision']:raise ValueError('Parent identity mismatch')
    report=apply_native_column_recipe(model,folder,condition)
    return model,report
