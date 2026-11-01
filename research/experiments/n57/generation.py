"""Own-history generation with a fixed native intermediate-state readout branch."""
from contextlib import contextmanager
import math

import torch
from transformers.cache_utils import DynamicCache

from interpretability.cache_proofs import check_cache
from research.experiments.n19.trajectory import cache_witness
from research.experiments.n35.core import tensor_sha

POLICIES = ('subtract_native_boundary', 'add_native_boundary')
ALPHA = 0.1


def cache_record(model, cache):
    value = cache_witness(cache)
    config = model.config.text_config
    check_cache(value, config.num_hidden_layers-config.num_kv_shared_layers,
                str(next(model.parameters()).dtype))
    if any(v['keys'] is None for v in value['layers']):
        raise ValueError('All actual native KV producers required')
    return value


def native_parts(model):
    config = model.config.text_config
    language = model.model.language_model
    boundary = config.num_hidden_layers-config.num_kv_shared_layers-1
    if (config.num_hidden_layers != 35 or config.num_kv_shared_layers != 20
            or len(language.layers) != 35 or boundary != 14
            or language.layers[boundary].self_attn.is_kv_shared_layer):
        raise ValueError('Pinned native last-KV-producer boundary14 required')
    return language, boundary


@contextmanager
def observe_native_states(model):
    language, boundary = native_parts(model)
    frames = {}

    def earlier(module, args, output):
        if ('early' in frames or output.requires_grad or output.ndim != 3
                or output.shape[0] != 1 or output.shape[1] < 1):
            raise ValueError('One actual native block14 output required')
        frames['early'] = output[:, -1:, :].detach().clone()

    def mature(module, args):
        if (len(args) != 1 or 'mature' in frames or args[0].requires_grad
                or args[0].ndim != 3 or args[0].shape[:2] != (1, 1)):
            raise ValueError('One actual current native head input required')
        frames['mature'] = args[0].detach().clone()

    handles = [language.layers[boundary].register_forward_hook(earlier),
               model.lm_head.register_forward_pre_hook(mature)]
    try:
        yield frames
        if set(frames) != {'early', 'mature'}:
            raise ValueError('Missing aligned native source states')
    finally:
        # Cleanup must preserve the original exception rather than replace it.
        for handle in reversed(handles):
            handle.remove()


def project(model, hidden, count):
    count('extra_native_head_evaluations')
    logits = model.lm_head(hidden)
    cap = model.config.get_text_config().final_logit_softcapping
    if cap is not None:
        if type(cap) not in (int, float) or not math.isfinite(cap) or cap <= 0:
            raise ValueError('Actual finite positive native softcap required')
        logits = logits / cap
        logits = torch.tanh(logits)
        logits = logits * cap
    return logits[0, -1]


def constrained_scores(native, branch, count):
    if (native.ndim != 1 or native.shape != branch.shape
            or native.dtype != torch.float32 or branch.dtype != torch.float32
            or not torch.isfinite(native).all() or not torch.isfinite(branch).all()):
        raise ValueError('Finite full-vocabulary FP32 native and branch logits required')
    threshold = native.max() + math.log(ALPHA)
    eligible = native >= threshold
    native_choice = int(native.argmax())
    if not bool(eligible[native_choice]) or not bool(eligible.any()):
        raise ValueError('Native maximum must remain in nonempty support')
    scores = branch.clone()
    count('policy_support_assignments')
    scores.masked_fill_(~eligible, -float('inf'))
    choice = int(scores.argmax())
    if not bool(eligible[choice]) or not bool(torch.isfinite(scores[choice])):
        raise ValueError('Actual constrained full-vocabulary choice must be eligible')
    return scores, {
        'alpha': ALPHA, 'threshold': float(threshold),
        'support_count': int(eligible.sum()), 'vocabulary_size': native.numel(),
        'support_sha256': tensor_sha(eligible),
        'native_argmax': native_choice, 'native_max_in_support': True,
        'chosen_in_support': True, 'chosen_native_logit': float(native[choice]),
        'chosen_branch_score': float(scores[choice]),
        'constrained_argmax': choice,
        'full_constrained_scores_sha256': tensor_sha(scores),
        'unrestricted_native_argmax_claimed': False,
    }


def readout(model, frames, actual_native_logits, policy, count):
    if policy not in POLICIES:
        raise ValueError('Fixed subtractive or additive native connection required')
    language, boundary = native_parts(model)
    early, mature = frames['early'], frames['mature']
    dtype = next(model.parameters()).dtype
    if (early.shape != mature.shape or mature.shape != (1, 1, model.config.text_config.hidden_size)
            or early.dtype != dtype or mature.dtype != dtype
            or not torch.isfinite(early).all() or not torch.isfinite(mature).all()):
        raise ValueError('Aligned finite native BF16/fixture FP32 states required')
    before = (tensor_sha(early), tensor_sha(mature))
    replay = project(model, mature.clone(), count)
    if (replay.dtype != actual_native_logits.dtype
            or tensor_sha(replay.float()) != tensor_sha(actual_native_logits.float())):
        raise ValueError('Actual native head/softcap replay failed')
    count('extra_native_final_RMS_evaluations')
    normalized = language.norm(early)
    if normalized.dtype != dtype or not torch.isfinite(normalized).all():
        raise ValueError('Actual native final RMS output required')
    sign = -1 if policy == 'subtract_native_boundary' else 1
    count('branch_input_constructions')
    mixed = (mature.float() + sign*normalized.float()).to(dtype)
    branch = project(model, mixed, count)
    if branch.dtype != dtype or not torch.isfinite(branch).all():
        raise ValueError('Finite actual native branch head output required')
    if before != (tensor_sha(early), tensor_sha(mature)):
        raise ValueError('Native captured sources changed')
    scores, support = constrained_scores(actual_native_logits.float(), branch.float(), count)
    event = {
        'source_layer_index': boundary, 'literal_coefficient': sign,
        'early_state_sha256': before[0], 'mature_state_sha256': before[1],
        'early_normalized_sha256': tensor_sha(normalized),
        'mixed_state_sha256': tensor_sha(mixed), 'state_dtype': str(dtype),
        'native_head_replay_full_FP32_sha256': tensor_sha(replay.float()),
        'actual_native_full_FP32_sha256': tensor_sha(actual_native_logits.float()),
        'native_head_replay_exact': True, 'source_inputs_unchanged': True,
        'branch_raw_logits_dtype': str(branch.dtype),
        'branch_full_FP32_sha256': tensor_sha(branch.float()),
        **support,
    }
    return scores, event


def generate(model, tokenizer, encoded, eos, policy, *, resource_guard, on_decision, count):
    if (policy not in POLICIES
            or any(p.requires_grad or p.grad is not None for p in model.parameters())
            or any(m.training for m in model.modules())):
        raise ValueError('Fixed frozen native inference policy required')
    native_parts(model)
    config = model.config.text_config
    n = int(encoded['input_ids'].numel())
    maximum = config.max_position_embeddings
    if (encoded['input_ids'].shape != (1, n) or n < 1 or type(maximum) is not int
            or not eos or any(type(t) is not int or not 0 <= t < config.vocab_size for t in eos)):
        raise ValueError('Exact single prompt and configured native boundaries required')
    versions = {k: p._version for k, p in model.named_parameters()}
    cache = DynamicCache(config=config)
    ids = []
    first = last = None
    attempts = exposures = 0
    interruption = None
    termination = 'interrupted'
    try:
        with torch.no_grad():
            while True:
                interruption = resource_guard()
                if interruption is not None:
                    if (not isinstance(interruption, dict) or 'reason' not in interruption
                            or 'evidence' not in interruption):
                        raise ValueError('Actual resource evidence required')
                    break
                step = len(ids)
                if n+step > maximum:
                    interruption = {'reason': 'context_exhausted',
                                    'evidence': {'configured_context': maximum,
                                                 'next_required_input_length': n+step}}
                    break
                args = encoded if step == 0 else {
                    'input_ids': torch.tensor([[ids[-1]]], device=encoded['input_ids'].device)}
                attempts += 1
                exposures += n if step == 0 else 1
                try:
                    with observe_native_states(model) as frames:
                        output = model(**args, past_key_values=cache, use_cache=True, logits_to_keep=1)
                    cache = output.past_key_values
                    native = output.logits[0, -1].detach()
                    if native.dtype != next(model.parameters()).dtype or not torch.isfinite(native).all():
                        raise ValueError('Finite actual native raw logits required')
                    scores, witness = readout(model, frames, native, policy, count)
                    token = int(scores.argmax())
                    if step == 0:
                        first = cache_record(model, cache)
                    event = {'step': step, 'argmax': token,
                             'raw_logits_dtype': str(native.dtype),
                             'full_logits_sha256': tensor_sha(scores),
                             'consumed_previous_token': None if step == 0 else ids[-1],
                             'consumed_input_length': n+step, 'readout': witness}
                except torch.OutOfMemoryError:
                    interruption = {'reason': 'memory_exhausted',
                                    'evidence': {'during_native_forward_or_readout': True}}
                    break
                ids.append(token)
                count('fresh_choices')
                on_decision(event)
                if token in eos:
                    termination = 'eos'
                    interruption = None
                    break
            if ids:
                last = cache_record(model, cache)
    finally:
        if (versions != {k: p._version for k, p in model.named_parameters()}
                or any(p.requires_grad or p.grad is not None for p in model.parameters())):
            raise ValueError('Frozen native parameter state changed')
    return {
        'generated_ids': ids, 'text': tokenizer.decode(ids, skip_special_tokens=True),
        'generated_tokens': len(ids), 'fresh_choices': len(ids), 'old_cell_carries': 0,
        'first_cache': first, 'terminal_cache': last, 'complete': termination == 'eos',
        'termination': termination, 'interruption': interruption,
        'new_response_token_ceiling': None, 'configured_context': maximum,
        'input_tokens': n, 'attempted_forward_calls': attempts,
        'input_token_exposures': exposures, 'per_token_whole_cache_authenticated': False,
        'support_constrained_full_vocabulary_policy': True,
    }
