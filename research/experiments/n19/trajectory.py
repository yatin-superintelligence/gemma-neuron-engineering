"""Fresh-cache greedy paths and byte fingerprints for a single decision."""
import hashlib,json
import torch


def tensor_witness(value):
    raw=value.detach().contiguous().cpu()
    return{'shape':list(raw.shape),'dtype':str(raw.dtype),'sha256':hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest()}


def cache_witness(cache):
    layers=[]
    for index,layer in enumerate(cache.layers):
        layers.append({'layer':index,**{name:tensor_witness(getattr(layer,name))if torch.is_tensor(getattr(layer,name,None))else None for name in ['keys','values']}})
    return{'layers':layers,'sha256':hashlib.sha256(json.dumps(layers,sort_keys=True,separators=(',',':')).encode()).hexdigest()}


def prepare(model,encoded,prefix):
    assert prefix
    with torch.no_grad():value=model(**encoded,use_cache=True,logits_to_keep=1)
    for token in prefix[:-1]:
        with torch.no_grad():value=model(input_ids=torch.tensor([[token]],device=encoded['input_ids'].device),past_key_values=value.past_key_values,use_cache=True,logits_to_keep=1)
    return value.past_key_values


def current(model,encoded,prefix,cache):
    return model(input_ids=torch.tensor([[prefix[-1]]],device=encoded['input_ids'].device),past_key_values=cache,use_cache=True,logits_to_keep=1)


def continue_tail(model,encoded,prefix,value,eos_ids,cap=64):
    tail=[]
    while len(tail)<cap-len(prefix):
        token=int(value.logits[0,-1].argmax());tail.append(token)
        if token in eos_ids:break
        if len(tail)==cap-len(prefix):break
        with torch.no_grad():value=model(input_ids=torch.tensor([[token]],device=encoded['input_ids'].device),past_key_values=value.past_key_values,use_cache=True,logits_to_keep=1)
    return tail
