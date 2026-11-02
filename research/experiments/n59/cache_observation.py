"""Read-only persistence of both compared retained-cache prefix hashes."""
import torch
from research.experiments.n19.trajectory import cache_witness
from research.experiments.n35.core import tensor_sha
from research.experiments.n59.assay import cache_retained_prefix, validate_growth, tensor_record


class CachedPrefixObserver:
    def __init__(self, count):
        self.count = count
        self.records = []
        self.pending = None
        self.handles = []

    def before(self, module, args, kwargs):
        ids, cache = kwargs['input_ids'], kwargs.get('past_key_values')
        if cache is None or cache.get_seq_length() == 0:
            return
        if self.pending is not None or ids.shape != (1, 1) or ids.dtype != torch.long:
            raise ValueError('Exactly one actual own-choice cached token required')
        self.pending = {'input_sha256': tensor_sha(ids), 'input_token': int(ids[0, 0]),
                        'before_whole_cache': cache_witness(cache),
                        'retained_before': cache_retained_prefix(cache)}

    def after(self, module, args, kwargs, output):
        if self.pending is None:
            return
        cache = output.past_key_values
        old = self.pending
        if len(cache.layers) != len(old['retained_before']):
            raise ValueError('Cached layer cardinality changed')
        growth = validate_growth(cache, old['retained_before'])
        compared = []
        for layer, before in zip(cache.layers, old['retained_before'], strict=True):
            if before['empty']:
                compared.append({'layer': before['layer'], 'empty_unchanged': True})
                continue
            n = before['retained_positions']
            after = {key: tensor_record(getattr(layer, key)[..., :n, :]) if n else None
                     for key in ('keys', 'values')}
            if any(after[key] != before[key] for key in ('keys', 'values')):
                raise ValueError('Both persisted compared prefix hashes must match')
            compared.append({'layer': before['layer'], 'retained_positions': n,
                'before': {key: before[key] for key in ('keys', 'values')}, 'after': after,
                'retained_BF16_bytes_per_K_or_V': 0 if not n else
                    after['keys']['shape'][0]*after['keys']['shape'][1]*n*after['keys']['shape'][3]*2})
        self.records.append({**old, 'after_whole_cache': cache_witness(cache),
                             'growth': growth, 'compared_retained_prefixes': compared})
        self.count('cached_prefix_observations')
        self.pending = None

    def attach(self, model):
        if self.handles:
            raise ValueError('Observer attachment exactly once')
        self.handles = [model.register_forward_pre_hook(self.before, with_kwargs=True),
                        model.register_forward_hook(self.after, with_kwargs=True)]

    def remove(self):
        for handle in self.handles:
            handle.remove()

