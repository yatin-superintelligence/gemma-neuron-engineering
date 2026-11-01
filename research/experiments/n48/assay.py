"""One fresh answer decision per policy; no response generation or labels."""
import torch
from transformers.cache_utils import DynamicCache
from interpretability.signed_native_mask import signed_mask
from research.experiments.n35.core import tensor_sha
from research.experiments.n37.continuation import decision

EDITS = ('remove_negative', 'random_negative_count', 'remove_positive', 'random_positive_count')
CONDITIONS = ('original', *EDITS)


def choice_metrics(logits, answer_ids):
    if set(answer_ids) != {'A', 'B'} or any(type(v) is not int or not 0 <= v < logits.numel() for v in answer_ids.values()) or answer_ids['A'] == answer_ids['B']:
        raise ValueError('Two distinct canonical answer token IDs required')
    x = logits.detach().float()
    if x.ndim != 1 or not torch.isfinite(x).all(): raise ValueError('Finite full-vocabulary vector required')
    token = int(x.argmax()); probabilities = x.softmax(-1)
    return {'argmax': token, 'choice': next((k for k, v in answer_ids.items() if v == token), 'other'),
        'answer_logits': {k: float(x[v]) for k, v in answer_ids.items()},
        'answer_probabilities': {k: float(probabilities[v]) for k, v in answer_ids.items()},
        'A_minus_B_margin': float(x[answer_ids['A']]-x[answer_ids['B']])}


def one_case(model, encoded, answer_ids, on_assignment=lambda: None):
    slots = model.config.text_config.num_hidden_layers-model.config.text_config.num_kv_shared_layers
    dtype = str(next(model.parameters()).dtype)
    def forward(mode):
        with torch.no_grad(), signed_mask(model, mode) as scope:
            if mode != 'original': on_assignment()
            output = model(**encoded, past_key_values=DynamicCache(config=model.config.text_config), use_cache=True, logits_to_keep=1)
            item, cache = decision(output, 0, slots, dtype)
            logits = output.logits[0, -1].detach().float().clone()
        assert len(scope['frames']) == 1 and scope['activation_assignments'] == int(mode != 'original')
        return {'decision': item, 'activation': scope['frames'][0], 'metrics': choice_metrics(logits, answer_ids)}, cache
    original, cache = forward('original'); identity, identity_cache = forward('identity')
    assert original['decision'] == identity['decision'] and identity_cache == cache
    assert original['activation']['native_h_sha256'] == identity['activation']['edited_h_sha256']
    edits, restorations = {}, {}
    for mode in EDITS:
        value, other_cache = forward(mode)
        assert other_cache == cache and value['decision']['cache_sha256'] == original['decision']['cache_sha256']
        assert value['activation']['native_h_sha256'] == original['activation']['native_h_sha256']
        assert value['activation']['other_positions_byte_exact'] is True
        restored, restored_cache = forward('original')
        assert restored == original and restored_cache == cache
        edits[mode], restorations[mode] = value, restored
    return {'original': original, 'identity': identity, 'edits': edits,
        'restorations': restorations, 'native_first_full_cache': cache,
        'answer_token_ids': answer_ids, 'input_tokens': int(encoded['input_ids'].numel()),
        'new_emitted_tokens': 0, 'parameter_assignments': 0, 'backward_calls': 0, 'optimizer_steps': 0}
