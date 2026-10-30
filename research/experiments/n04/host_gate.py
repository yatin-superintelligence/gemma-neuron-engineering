"""Actual-Gemma fixture for copied memory connections, no pretrained model."""
from pathlib import Path
import datetime
import hashlib
import json
import time
import traceback
import torch
import transformers
from transformers import Gemma4Config, Gemma4TextConfig, Gemma4ForConditionalGeneration
from interpretability.memory_connection import memory_connection

ROOT = Path(__file__).resolve().parents[3]


def main():
    start = time.monotonic()
    protocol_path = Path(__file__).with_name('host_protocol.json')
    protocol = json.loads(protocol_path.read_text())
    output = ROOT/'runs/n04/host/version2'
    output.mkdir(parents=True,exist_ok=True)
    assert not (output/'result.json').exists()
    assert transformers.__version__ == '5.8.0'
    torch.set_num_threads(1)
    ids = torch.tensor([protocol['tokens']])
    records = []
    try:
        for seed in protocol['seeds']:
            torch.manual_seed(seed)
            text = Gemma4TextConfig(vocab_size=32,vocab_size_per_layer_input=32,hidden_size=16,
                intermediate_size=24,num_hidden_layers=6,num_attention_heads=2,num_key_value_heads=2,
                head_dim=8,global_head_dim=8,num_global_key_value_heads=2,num_kv_shared_layers=2,
                hidden_size_per_layer_input=4,use_double_wide_mlp=True,
                layer_types=['sliding_attention','full_attention']*3,sliding_window=16,
                max_position_embeddings=64,rope_parameters={
                  'full_attention':{'rope_type':'default','rope_theta':10000.0},
                  'sliding_attention':{'rope_type':'default','rope_theta':10000.0}})
            config = Gemma4Config(text_config=text.to_dict())
            config.vision_config = None
            config.audio_config = None
            model = Gemma4ForConditionalGeneration(config).double().eval().requires_grad_(False)
            decoder = model.model.language_model
            widths = [layer.mlp.down_proj.in_features for layer in decoder.layers]
            assert widths == [24]*4+[48]*2
            def logits(sequence=ids,cache=None,use_cache=False):
                return model(sequence,past_key_values=cache,use_cache=use_cache,logits_to_keep=1)
            def past_zero():
                def hook(module,args):
                    value=args[0].clone();value[:,:-1]=0
                    return (value,*args[1:])
                return decoder.layers[4].mlp.down_proj.register_forward_pre_hook(hook)
            with torch.no_grad():
                base = logits().logits.detach().clone()
                generated = model.generate(ids,max_new_tokens=3,do_sample=False,use_cache=True,pad_token_id=0)
                handle = past_zero()
                try:
                    assert torch.equal(logits().logits,base)
                finally:
                    handle.remove()
            for mode in protocol['modes']:
                for strength in protocol['strengths']:
                    with memory_connection(model,source_index=3,target_index=5,strength=strength,mode=mode) as receipt:
                        assert decoder.config.num_kv_shared_layers == 2
                        assert widths == [layer.mlp.down_proj.in_features for layer in decoder.layers]
                        with torch.no_grad():
                            full = logits().logits.detach().clone()
                            if strength == 0:
                                assert torch.equal(full,base)
                                assert torch.equal(model.generate(ids,max_new_tokens=3,do_sample=False,use_cache=True,pad_token_id=0),generated)
                            cache = None
                            incremental_errors = []
                            for i in range(ids.shape[1]):
                                step = logits(ids[:,i:i+1],cache=cache,use_cache=True)
                                cache = step.past_key_values
                                prefix = logits(ids[:,:i+1]).logits
                                torch.testing.assert_close(step.logits,prefix,atol=1e-10,rtol=1e-10)
                                incremental_errors.append(float((step.logits-prefix).abs().max()))
                            handle = past_zero()
                            try:
                                changed = logits().logits.detach().clone()
                            finally:
                                handle.remove()
                            if strength == 0 or mode == 'value_current':
                                assert torch.equal(changed,full)
                            else:
                                assert not torch.equal(changed,full), 'Historical connection failed to open predicted path'
                        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
                    assert model.config.text_config.num_kv_shared_layers == 2
                    with torch.no_grad():
                        assert torch.equal(logits().logits,base)
                    record={'seed':seed,'mode':mode,'strength':strength,
                        'full_incremental_max_abs':max(incremental_errors),
                        'past_intervention_max_abs':float((changed-full).abs().max()),
                        'past_intervention_exact_null':torch.equal(changed,full),
                        'receipt':receipt,'restored_exact':True,'optimizer_steps':0}
                    records.append(record)
                    with (output/'cases.jsonl').open('a') as stream:
                        stream.write(json.dumps(record)+'\n')
        result={'status':'complete','experiment':'N04H','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'cases':len(records),'seeds':protocol['seeds'],'all_gates_passed':True,'optimizer_steps':0,
            'pretrained_model_tested':False,'seconds':time.monotonic()-start,
            'protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            'connection_source_sha256':hashlib.sha256((ROOT/'interpretability/memory_connection.py').read_bytes()).hexdigest(),
            'interpretation':'Random-weight implementation feasibility only. Off-mode preserves the baseline;historical connections open an internal path;current-only controls retain the null. No coding or semantic-role claim.'}
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    except Exception as error:
        (output/'failure.json').write_text(json.dumps({'error':repr(error),'traceback':traceback.format_exc(),
            'completed_cases':len(records),'optimizer_steps':0},indent=2)+'\n')
        raise


if __name__=='__main__':
    main()
