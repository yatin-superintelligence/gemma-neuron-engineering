"""Small actual-Gemma anchor-hook checks, with no pretrained download."""
from pathlib import Path
import datetime,json,hashlib
import torch
from transformers import Gemma4Config,Gemma4TextConfig,Gemma4ForConditionalGeneration
from interpretability.anchor_intervention import anchor_intervention


def main():
    torch.manual_seed(1505);torch.set_num_threads(1)
    text=Gemma4TextConfig(vocab_size=32,vocab_size_per_layer_input=32,hidden_size=16,
        intermediate_size=24,num_hidden_layers=6,num_attention_heads=2,num_key_value_heads=2,
        head_dim=8,global_head_dim=8,num_global_key_value_heads=2,num_kv_shared_layers=2,
        hidden_size_per_layer_input=4,use_double_wide_mlp=True,sliding_window=16,
        max_position_embeddings=64,layer_types=['sliding_attention','full_attention']*3,
        rope_parameters={k:{'rope_type':'default','rope_theta':10000.0}for k in ['sliding_attention','full_attention']})
    config=Gemma4Config(text_config=text.to_dict());config.vision_config=None;config.audio_config=None
    model=Gemma4ForConditionalGeneration(config).double().eval().requires_grad_(False)
    ids=torch.tensor([[2,3,4,5,6]]);coordinates=[{'layer':1,'channel':c,'flat':24+c}for c in [0,1,2]]
    donor={}
    def capture(module,args):
        for c in coordinates:donor[c['flat']]=float(args[0][0,2,c['channel']])
    handle=model.model.language_model.layers[1].mlp.down_proj.register_forward_pre_hook(capture)
    with torch.inference_mode():base=model(ids,use_cache=False).logits.detach().clone()
    handle.remove();cases=[]
    for scope,gain in [('anchor',1),('anchor',0),('anchor',2),('all',0),('all',2)]:
        with anchor_intervention(model,coordinates,2,gain=gain,scope=scope)as receipt:
            with torch.inference_mode():changed=model(ids,use_cache=False).logits.detach().clone()
        assert torch.equal(base,changed)==(gain==1)
        with torch.inference_mode():assert torch.equal(base,model(ids,use_cache=False).logits)
        cases.append({'scope':scope,'gain':gain,'exact':torch.equal(base,changed),'receipt':receipt})
    for op in ['swap','rescue']:
        with anchor_intervention(model,coordinates,2,operation=op,donor=donor)as receipt:
            with torch.inference_mode():assert torch.equal(base,model(ids,use_cache=False).logits)
        cases.append({'operation':op,'exact':True,'receipt':receipt})
    with anchor_intervention(model,coordinates,2,gain=0)as receipt:
        with torch.inference_mode():
            output=model.generate(ids,max_new_tokens=3,do_sample=False,use_cache=True,pad_token_id=0)
    assert receipt['anchor_hits']==1 and receipt['cached_anchor_skips']==2
    cases.append({'cached_generation_tokens':output[0].tolist(),'receipt':receipt})
    root=Path(__file__).resolve().parents[3];out=root/'runs/n05/host';out.mkdir(parents=True,exist_ok=True)
    result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'complete','cases':cases,
        'all_gates_passed':True,'optimizer_steps':0,'pretrained_model_tested':False,
        'source_sha256':hashlib.sha256((root/'interpretability/anchor_intervention.py').read_bytes()).hexdigest()}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('N05host:8cases passed,zero training')


if __name__=='__main__':main()
