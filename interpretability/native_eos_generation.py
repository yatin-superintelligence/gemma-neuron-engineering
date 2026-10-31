"""Greedy native-EOS generation with streamed evidence and resumable history.

There is no response-token ceiling. The caller supplies actual resource checks;
the model's configured context is a separate boundary, not a completion signal.
An interrupted record remains incomplete. Pretrained use requires a prospective
protocol binding the prompt, model, edit and any saved native reference history.
"""

import copy

import torch
from transformers.cache_utils import DynamicCache

from research.experiments.n37.continuation import decision

RESOURCE_REASONS = {'wall_time_interrupt', 'memory_reservation_interrupt',
                    'output_storage_interrupt', 'external_interrupt'}


def generate_native_eos(model, tokenizer, encoded, eos, *, resource_guard,
                        on_decision, reference=None):
    """Continue to native EOS, recording actual resource/context interruptions.

    ``resource_guard`` returns None or a JSON-compatible resource witness with
    ``reason`` and ``evidence``. It must measure resources, not impose a disguised
    token limit. ``on_decision`` persists every new token and its exact hashes.
    A reference is unfinished own-history evidence, never a donor or an answer.
    Every reference choice/logit/cache must replay exactly before a new suffix.
    """
    config = model.config.text_config
    maximum = config.max_position_embeddings
    if type(maximum) is not int or maximum < 1:
        raise ValueError('Actual configured context boundary required')
    if not callable(resource_guard) or not callable(on_decision):
        raise ValueError('Resource and streaming callbacks required')
    if not eos or any(type(i) is not int or not 0 <= i < config.vocab_size for i in eos):
        raise ValueError('Explicit valid native EOS IDs required')
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError('Freeze model parameters before inference')
    if any(module.training for module in model.modules()):
        raise ValueError('Evaluation mode required')
    versions = {n: p._version for n, p in model.named_parameters()}
    n = int(encoded['input_ids'].numel())
    if encoded['input_ids'].ndim != 2 or encoded['input_ids'].shape[0] != 1 or n < 1:
        raise ValueError('One nonempty encoded native prompt required')
    prefix = [] if reference is None else list(reference['generated_ids'])
    old_trace = [] if reference is None else copy.deepcopy(reference['decision_trace'])
    if reference is not None:
        if not prefix or len(prefix) != len(old_trace) or any(i in eos for i in prefix):
            raise ValueError('Only unfinished non-EOS own histories can resume')
        if prefix != [r['argmax'] for r in old_trace]:
            raise ValueError('Reference token choices and trace disagree')
    ids, trace, replay = list(prefix), list(old_trace), []
    first = None if reference is None else copy.deepcopy(reference['first_cache'])
    layers = config.num_hidden_layers - config.num_kv_shared_layers
    dtype = str(next(model.parameters()).dtype)
    cache = DynamicCache(config=config)
    attempted = exposures = completed = emitted = 0
    step = 0
    interruption = None
    termination = 'interrupted'
    try:
        with torch.no_grad():
            while True:
                interruption = resource_guard()
                if interruption is not None:
                    if not isinstance(interruption, dict) or interruption.get('reason') not in RESOURCE_REASONS or 'evidence' not in interruption:
                        raise ValueError('Explicit actual-resource interruption witness required')
                    break
                if n + step > maximum:
                    interruption = {'reason': 'context_exhausted',
                                    'evidence': {'configured_context': maximum,
                                                 'next_required_input_length': n + step}}
                    break
                args = encoded if step == 0 else {'input_ids': torch.tensor(
                    [[prefix[step - 1] if step < len(prefix) else ids[-1]]],
                    device=encoded['input_ids'].device)}
                attempted += 1
                exposures += n if step == 0 else 1
                try:
                    output = model(**args, past_key_values=cache, use_cache=True, logits_to_keep=1)
                    item, witness = decision(output, step, layers, dtype)
                except torch.OutOfMemoryError as error:
                    interruption = {'reason': 'memory_exhausted', 'evidence': {
                        'exception_type': type(error).__name__, 'during_forward_or_witness': True}}
                    break
                completed += 1
                cache = output.past_key_values
                if step == 0:
                    if reference is not None:
                        assert witness == reference['first_cache'], 'Native reference first cache changed'
                    first = witness
                if step < len(prefix):
                    assert item == old_trace[step], 'Native own-history choice/logit/cache changed'
                    replay.append(item)
                else:
                    assert len(replay) == len(prefix)
                    ids.append(item['argmax'])
                    trace.append(item)
                    emitted += 1
                    on_decision(copy.deepcopy(item))
                    if item['argmax'] in eos:
                        termination = 'eos'
                        interruption = None
                        break
                step += 1
    finally:
        assert versions == {n: p._version for n, p in model.named_parameters()}
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    return {'generated_ids': ids, 'text': tokenizer.decode(ids, skip_special_tokens=True),
            'generated_tokens': len(ids), 'new_generated_tokens': emitted,
            'decision_trace': trace, 'first_cache': first, 'replayed_trace': replay,
            'reference_tokens_replayed_exact': len(replay), 'reference_tokens_required': len(prefix),
            'prefix_exact': len(replay) == len(prefix), 'termination': termination,
            'complete': termination == 'eos', 'interruption': interruption,
            'input_tokens': n, 'attempted_forward_calls': attempted,
            'completed_forward_calls': completed, 'input_token_exposures': exposures,
            'new_response_token_ceiling': None, 'configured_context': maximum,
            'optimizer_steps': 0, 'backward_calls': 0, 'parameter_assignments': 0}
