"""Native-EOS persistent policies with exact, conditional completed-cell carries."""
import copy
import torch
from transformers.cache_utils import DynamicCache
from interpretability.signed_native_mask import signed_mask
from research.experiments.n37.continuation import decision

EDITS = ('remove_negative', 'random_negative_count', 'remove_positive', 'random_positive_count')


def eligible(ids, reference, points):
    step = len(ids)
    return str(step) in points and ids == reference['generated_ids'][:step]


def generate(model, tokenizer, encoded, eos, mode, reference, points, *, resource_guard, on_decision, on_scope, on_source_guard):
    if mode not in EDITS or not callable(resource_guard) or not callable(on_decision) or not callable(on_scope) or not callable(on_source_guard):
        raise ValueError('Fixed policy and actual resource/stream callbacks required')
    if any(p.requires_grad or p.grad is not None for p in model.parameters()) or any(m.training for m in model.modules()):
        raise ValueError('Frozen evaluation model required')
    config = model.config.text_config
    maximum = config.max_position_embeddings
    if type(maximum) is not int or maximum < 1 or not eos or any(type(v) is not int or not 0 <= v < config.vocab_size for v in eos):
        raise ValueError('Native context/EOS boundaries required')
    n = int(encoded['input_ids'].numel())
    if encoded['input_ids'].ndim != 2 or encoded['input_ids'].shape[0] != 1 or n < 1:
        raise ValueError('One actual encoded prompt required')
    versions = {k: p._version for k, p in model.named_parameters()}
    slots = config.num_hidden_layers-config.num_kv_shared_layers
    dtype = str(next(model.parameters()).dtype)
    cache = DynamicCache(config=config)
    ids, trace = [], []
    first = None
    attempted = exposures = fresh = carried = assignments = source_attempts = 0
    native_prefix = True
    interruption = None
    termination = 'interrupted'
    try:
        with torch.no_grad():
            while True:
                interruption = resource_guard()
                if interruption is not None:
                    if interruption.get('reason') not in {'wall_time_interrupt', 'memory_reservation_interrupt', 'output_storage_interrupt', 'external_interrupt'} or 'evidence' not in interruption:
                        raise ValueError('Observed actual resource witness required')
                    break
                step = len(ids)
                if n+step > maximum:
                    interruption = {'reason': 'context_exhausted', 'evidence': {'configured_context': maximum, 'next_required_input_length': n+step}}
                    break
                args = encoded if step == 0 else {'input_ids': torch.tensor([[ids[-1]]], device=encoded['input_ids'].device)}
                carry = native_prefix and eligible(ids, reference, points)
                scope = None
                attempted += 1
                exposures += n if step == 0 else 1
                try:
                    if carry:
                        source_attempts += 1
                        on_source_guard()
                        output = model(**args, past_key_values=cache, use_cache=True, logits_to_keep=1)
                        native, witness = decision(output, step, slots, dtype)
                        if native != reference['decision_trace'][step]: raise ValueError('Carry native source history/logits/cache changed')
                        source = points[str(step)]
                        item = copy.deepcopy(source['edits'][mode]['decision'])
                        if type(item['step']) is not int or item['step'] != step or item['cache_sha256'] != native['cache_sha256']:
                            raise ValueError('Exact completed decision and same-input cache required')
                        event = {'kind': 'carry', 'decision': item, 'source': copy.deepcopy(source['provenance'][mode])}
                    else:
                        try:
                            with signed_mask(model, mode) as scope:
                                output = model(**args, past_key_values=cache, use_cache=True, logits_to_keep=1)
                                item, witness = decision(output, step, slots, dtype)
                        finally:
                            if scope is not None:
                                assignments += scope['activation_assignments']
                                on_scope(scope['activation_assignments'])
                        if len(scope['frames']) != 1 or scope['activation_assignments'] != 1: raise ValueError('One actual current-position mask per new call required')
                        frame = scope['frames'][0]
                        compact = {k: frame[k] for k in ['native_h_sha256', 'mask_sha256', 'negative_count', 'positive_count', 'width', 'other_positions_byte_exact']}
                        compact['mask_count'] = len(frame['indices'])
                        event = {'kind': 'fresh', 'decision': item, 'mask': compact}
                    cache = output.past_key_values
                    if step == 0:
                        if witness != reference['first_cache']: raise ValueError('Actual same-input first cache changed')
                        first = witness
                    if native_prefix and step < len(reference['decision_trace']) and item['cache_sha256'] != reference['decision_trace'][step]['cache_sha256']:
                        raise ValueError('Same consumed history cache changed')
                except torch.OutOfMemoryError:
                    interruption = {'reason': 'memory_exhausted', 'evidence': {'during_forward_or_witness': True}}
                    break
                ids.append(item['argmax']); trace.append(event)
                if carry: carried += 1
                else: fresh += 1
                on_decision(copy.deepcopy(event))
                native_prefix = native_prefix and step < len(reference['generated_ids']) and item['argmax'] == reference['generated_ids'][step]
                if item['argmax'] in eos:
                    termination = 'eos'; interruption = None; break
    finally:
        if versions != {k: p._version for k, p in model.named_parameters()} or any(p.grad is not None or p.requires_grad for p in model.parameters()):
            raise ValueError('Parameter state changed during inference')
    return {'generated_ids': ids, 'text': tokenizer.decode(ids, skip_special_tokens=True), 'decision_trace': trace,
        'first_cache': first, 'generated_tokens': len(ids), 'new_generated_tokens': fresh, 'carried_generated_tokens': carried,
        'attempted_forward_calls': attempted, 'input_token_exposures': exposures, 'activation_assignments': assignments,
        'carry_native_source_guard_forwards': source_attempts, 'fresh_policy_forwards': attempted-source_attempts,
        'complete': termination == 'eos', 'termination': termination, 'interruption': interruption,
        'new_response_token_ceiling': None, 'configured_context': maximum, 'input_tokens': n,
        'optimizer_steps': 0, 'backward_calls': 0, 'parameter_assignments': 0}
