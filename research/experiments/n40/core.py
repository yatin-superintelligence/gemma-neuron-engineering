"""Fixed-input lineage and explicit parameter-version accounting for N40."""

from collections import Counter as Counts
from contextlib import contextmanager
import copy

import torch
from transformers.cache_utils import DynamicCache

from interpretability.prescribed_columns import prescribed_columns
from research.experiments.n35.core import snapshot
from research.experiments.n37.continuation import decision

CONDITIONS = ['selected_zero', 'randomA_zero', 'randomB_zero', 'prompt_only']
GROUPS = {'selected_zero': 'selected32', 'randomA_zero': 'randomA32', 'randomB_zero': 'randomB32'}


def expected_work(rows, metadata, references, predictions):
    s = sum(len(references[r['id']]['trace']) for r in rows)
    p = sum(metadata['original'][r['id']]['input_tokens'] + len(references[r['id']]['trace']) - 1 for r in rows)
    coding = next(r for r in rows if r['role'] == 'coding')
    retention = next(r for r in rows if r['role'] == 'retention')
    sentinels = sum(metadata['original'][r['id']]['input_tokens'] for r in [coding, retention])
    return {'calls': 8 * s + 14 + sum(r['generated_tokens'] for r in predictions),
            'input_token_exposures': 8 * p + 7 * sentinels + sum(r['input_tokens'] + r['generated_tokens'] - 1 for r in predictions)}


def teacher(model, encoded, reference):
    """Measure at most three decisions on explicit native reference inputs.

    Edited decisions are never fed back here. Free generation is a separate
    function and always starts with an empty cache and the declared user prompt.
    """
    ids = reference['generated_prefix_ids']
    if not 1 <= len(ids) <= 3 or len(ids) != len(reference['trace']):
        raise ValueError('One to three aligned native reference decisions required')
    cache = DynamicCache(config=model.config.text_config)
    trace, first = [], None
    with torch.no_grad():
        for j in range(len(ids)):
            args = encoded if j == 0 else {'input_ids': torch.tensor([[ids[j - 1]]], device=encoded['input_ids'].device)}
            value = model(**args, past_key_values=cache, use_cache=True, logits_to_keep=1)
            item, witness = decision(value, j, 15, str(next(model.parameters()).dtype))
            trace.append(item)
            if j == 0: first = witness
            cache = value.past_key_values
    return {'trace': trace, 'first_cache': first,
            'expected_counter': {'calls': len(ids), 'input_token_exposures': int(encoded['input_ids'].numel()) + len(ids) - 1}}


def assert_native(record, reference):
    assert record['trace'] == reference['trace'] and record['first_cache'] == reference['first_cache']


def assert_fixed_cache(record, reference):
    assert record['first_cache'] == reference['first_cache']
    assert [r['cache_sha256'] for r in record['trace']] == [r['cache_sha256'] for r in reference['trace']]


class VersionLedger:
    """Count every declared assignment and restore, never optimizer updates."""
    def __init__(self, model):
        self.before = snapshot(model)
        self.deltas = Counts()
        self.contexts = []
        self.names = {id(p): n for n, p in model.named_parameters()}

    def assert_restored(self, model):
        after = snapshot(model)
        expected = copy.deepcopy(self.before)
        for name, count in self.deltas.items(): expected['parameters'][name]['version'] += count
        assert after == expected, 'Undeclared native state or parameter version change'

    @contextmanager
    def edit(self, model, coordinates, factor, context_id):
        with prescribed_columns(model, coordinates, factor) as receipt:
            yield receipt
        for item in receipt['layers']:
            weight = model.model.language_model.layers[item['layer']].mlp.down_proj.weight
            name = self.names[id(weight)]
            self.deltas[name] += item['declared_version_increments']
        self.contexts.append({'context_id': context_id, **receipt})
        self.assert_restored(model)
