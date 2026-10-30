"""Two-layer synthetic weight tests,no checkpoint download or inference."""
import copy,json,tempfile,unittest
from pathlib import Path
import torch
from torch import nn
from safetensors.torch import save_file
from reproduce.global_native import apply_native_column_recipe
from reproduce.model_edits import tensor_sha256,file_sha256


class NativeColumnBundleTests(unittest.TestCase):
    def fixture(self,folder):
        model=nn.Module();model.model=nn.Module();model.model.language_model=nn.Module();layers=[];tensors={};coordinates=[];source={};edited={}
        for index,channel in[(0,1),(1,2)]:
            layer=nn.Module();layer.mlp=nn.Module();layer.mlp.down_proj=nn.Linear(4,3,bias=False,dtype=torch.bfloat16);layers.append(layer)
            with torch.no_grad():layer.mlp.down_proj.weight.copy_((torch.arange(12).reshape(3,4)+index)/16)
            w=layer.mlp.down_proj.weight;original=w[:,channel].detach().clone();doubled=original*2;source[str(index)]=tensor_sha256(w);expected=w.detach().clone();expected[:,channel]=doubled;edited[str(index)]=expected
            key=f'L{index}_C{channel}';tensors[key+'_original']=original;tensors[key+'_double']=doubled;coordinates.append({'layer':index,'channel':channel,'shape':[3],'original_sha256':tensor_sha256(original),'double_sha256':tensor_sha256(doubled)})
        model.model.language_model.layers=nn.ModuleList(layers);model.requires_grad_(False)
        save_file(tensors,str(folder/'native_columns.safetensors'))
        recipe={'schema_version':1,'kind':'native_column_patch','optimizer_steps':0,'conditions':[{'name':'double','group':'selected','factor':2}],'groups':{'selected':[{'layer':c['layer'],'channel':c['channel']}for c in coordinates]},'coordinates':coordinates,'files':{'native_columns.safetensors':file_sha256(folder/'native_columns.safetensors')},'source_down_matrix_sha256':source,'expected_down_matrix_sha256':{'double':{k:tensor_sha256(v)for k,v in edited.items()}}}
        (folder/'recipe.json').write_text(json.dumps(recipe));return model,recipe,edited

    def test_exact_saved_variant_and_other_columns(self):
        with tempfile.TemporaryDirectory()as temporary:
            folder=Path(temporary);model,recipe,expected=self.fixture(folder);report=apply_native_column_recipe(model,folder,'double')
            self.assertTrue(all(torch.equal(model.model.language_model.layers[int(i)].mlp.down_proj.weight,w)for i,w in expected.items()));self.assertTrue(report['all_expected_projection_hashes_exact']);self.assertEqual(report['optimizer_steps'],0)

    def test_later_source_mismatch_refused_before_any_assignment(self):
        with tempfile.TemporaryDirectory()as temporary:
            folder=Path(temporary);model,recipe,_=self.fixture(folder);w=model.model.language_model.layers[1].mlp.down_proj.weight
            with torch.no_grad():w[0,2]+=1
            before=copy.deepcopy(model.state_dict());versions={n:p._version for n,p in model.named_parameters()}
            with self.assertRaises(ValueError):apply_native_column_recipe(model,folder,'double')
            self.assertTrue(all(torch.equal(v,before[k])for k,v in model.state_dict().items()));self.assertEqual(versions,{n:p._version for n,p in model.named_parameters()})

    def test_failed_output_hash_restores_both_layers(self):
        with tempfile.TemporaryDirectory()as temporary:
            folder=Path(temporary);model,recipe,_=self.fixture(folder);before=copy.deepcopy(model.state_dict());recipe['expected_down_matrix_sha256']['double']['1']='0'*64;(folder/'recipe.json').write_text(json.dumps(recipe))
            with self.assertRaises(ValueError):apply_native_column_recipe(model,folder,'double')
            self.assertTrue(all(torch.equal(v,before[k])for k,v in model.state_dict().items()));self.assertTrue(all(p.grad is None for p in model.parameters()))
