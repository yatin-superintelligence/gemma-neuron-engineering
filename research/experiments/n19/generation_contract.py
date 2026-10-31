"""Resolve the pinned library defaults before validating manual greedy decoding."""


def resolve_generation_contract(model):
    config,leftover=model._prepare_generation_config(None,do_sample=False,use_cache=True,max_new_tokens=64)
    assert not leftover,leftover
    result=config.to_dict()
    neutral={'repetition_penalty':1.0,'encoder_repetition_penalty':1.0,'no_repeat_ngram_size':0,'encoder_no_repeat_ngram_size':0,'min_length':0,'num_beams':1,'num_return_sequences':1,'do_sample':False,'use_cache':True,'remove_invalid_values':False}
    for key,value in neutral.items():assert result.get(key)==value,(key,result.get(key))
    for key in ['renormalize_logits','token_healing']:assert result.get(key)in[None,False],(key,result.get(key))
    for key in ['bad_words_ids','forced_bos_token_id','forced_eos_token_id','suppress_tokens','begin_suppress_tokens','sequence_bias','watermarking_config','min_new_tokens','exponential_decay_length_penalty','guidance_scale']:assert result.get(key)is None,(key,result.get(key))
    assert result['eos_token_id']==[1,106,50]and result['bos_token_id']==2 and result['pad_token_id']==0
    return result
