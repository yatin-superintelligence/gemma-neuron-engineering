"""Current-position witnesses include the exact unaffected earlier cache."""
import hashlib
import json

import torch

from interpretability.mlp_composition import compose_mlp_donors
from interpretability.scoped_channel import digest
from research.experiments.n27.measure import snapshot as full_snapshot


def snapshot(output, slots=15):
    value = full_snapshot(output, slots)
    rows = []
    for index, layer in enumerate(output.past_key_values.layers):
        row = {'layer': index}
        for name in ['keys', 'values']:
            past = getattr(layer, name)[:, :, :-1]
            assert past.shape[-2] > 0
            row[name] = {'shape': list(past.shape), 'dtype': str(past.dtype), 'sha256': digest(past)}
        rows.append(row)
    value['past_cache'] = {'layers': rows, 'sha256': hashlib.sha256(json.dumps(
        rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}
    return value


def forward(model, encoded, slots=15):
    with torch.no_grad():
        output = model(**encoded, use_cache=True, logits_to_keep=1)
    return output, snapshot(output, slots)


def assert_current_path(edited, original):
    assert edited['past_cache'] == original['past_cache']
    assert edited['cache']['layers'][0] == original['cache']['layers'][0]


def paired_self(model, encoded, donor, reference):
    """Compare fresh live native/self caches without sharing their storage."""
    original, first = forward(model, encoded)
    native, edited_states, nonalias = [first], [], []
    position = encoded['input_ids'].shape[1]-1
    with compose_mlp_donors(model, position,
                           [{'layers': list(range(35)), 'donor': donor}]) as proof:
        edited, value = forward(model, encoded)
        edited_states.append(value)
        for index in range(len(reference)+1):
            pointers = {getattr(layer, name).untyped_storage().data_ptr()
                        for layer in original.past_key_values.layers for name in ['keys', 'values']}
            assert all(getattr(layer, name).untyped_storage().data_ptr() not in pointers
                       for layer in edited.past_key_values.layers for name in ['keys', 'values'])
            assert native[-1] == edited_states[-1]
            nonalias.append(True)
            if index == len(reference):
                break
            ids = torch.tensor([[reference[index]]], device=encoded['input_ids'].device)
            with torch.no_grad():
                original = model(input_ids=ids, past_key_values=original.past_key_values,
                                 use_cache=True, logits_to_keep=1)
                edited = model(input_ids=ids, past_key_values=edited.past_key_values,
                               use_cache=True, logits_to_keep=1)
            native.append(snapshot(original))
            edited_states.append(snapshot(edited))
    return {'reference_tokens': reference, 'native': native, 'self': edited_states,
            'nonalias': nonalias, 'proof': proof}
