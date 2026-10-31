"""Native greedy generation with an independent first-prefill scope witness."""
import hashlib
import json
import torch
from transformers.cache_utils import DynamicCache
from interpretability.cache_proofs import check_cache
from research.experiments.n19.trajectory import cache_witness


def float_sha(value):
    return hashlib.sha256(value.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()


def encode(tokenizer,row,device):
    text=tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    encoded=tokenizer(text,add_special_tokens=False,return_tensors='pt')
    ids=encoded['input_ids'][0].tolist()
    metadata={'id':row['id'],'input_tokens':len(ids),'template_text_sha256':hashlib.sha256(text.encode()).hexdigest(),'input_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
    assert tokenizer.decode(ids,skip_special_tokens=False)==text
    return encoded.to(device),metadata


def observe(model,encoded):
    text_config=model.config.text_config
    cache=DynamicCache(config=text_config)
    with torch.no_grad():value=model(**encoded,past_key_values=cache,use_cache=True,logits_to_keep=1)
    logits=value.logits[0,-1].detach().float().cpu()
    assert torch.isfinite(logits).all()
    witness=cache_witness(value.past_key_values)
    check_cache(witness,text_config.num_hidden_layers-text_config.num_kv_shared_layers,str(next(model.parameters()).dtype))
    return {'final_logits_sha256':float_sha(logits),'comparison_dtype':'torch.float32','raw_logits_dtype':str(value.logits.dtype),'top1_token_id':int(logits.argmax()),'vocabulary':int(logits.numel()),'cache':witness}


def native_generate(model,tokenizer,encoded,cap,scope):
    config,leftover=model._prepare_generation_config(None,do_sample=False,use_cache=True,max_new_tokens=cap)
    assert not leftover and config.do_sample is False and config.num_beams==1 and config.num_return_sequences==1 and config.use_cache
    eos=config.eos_token_id
    eos={int(eos)}if isinstance(eos,int)else set(map(int,eos))
    assert eos
    with torch.no_grad():value=model.generate(**encoded,do_sample=False,use_cache=True,max_new_tokens=cap)
    n=int(encoded['input_ids'].shape[1]);ids=value[0,n:].detach().cpu().tolist()
    assert 0<len(ids)<=cap
    termination='eos'if ids[-1]in eos else'cap'
    assert termination=='eos'or len(ids)==cap
    return {'token_ids':ids,'text':tokenizer.decode(ids,skip_special_tokens=True),'generated_tokens':len(ids),'max_new_tokens':cap,'at_cap':len(ids)==cap,'termination':termination,'first_token_matches_scope':ids[0]==scope['top1_token_id']},config.to_dict()
