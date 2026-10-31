"""Compare saved cache paths only while consumed token histories match."""

import re


def compare_shared_input_history(native, edited, *, eos=frozenset({1, 106, 50})):
    """Include the first differing prediction: its consumed inputs still match.

    A later decision consumes the differing emitted token and is outside this
    comparison. Different prompts are refused rather than treated as a cache
    failure. Callers must separately establish checkpoint/edit provenance.
    """
    if native['input_metadata'] != edited['input_metadata']:
        raise ValueError('Exact input metadata equality required')
    metadata = native['input_metadata']
    if type(metadata.get('input_tokens')) is not int or metadata['input_tokens'] < 1:
        raise ValueError('Positive measured input length required')
    if any(not re.fullmatch(r'[a-f0-9]{64}', metadata.get(field, ''))
           for field in ['input_ids_sha256', 'template_text_sha256']):
        raise ValueError('Exact token and rendered-template input hashes required')
    for record in [native, edited]:
        ids, trace = record['generated_ids'], record['decision_trace']
        if not record['complete'] or record['termination'] != 'eos' or not ids:
            raise ValueError('Complete native-EOS records required')
        if ids[-1] not in eos or any(i in eos for i in ids[:-1]):
            raise ValueError('Exactly one terminal native EOS required')
        if any(type(i) is not int or i < 0 for i in ids):
            raise ValueError('Nonnegative integer token IDs required')
        if len(ids) != len(trace) or ids != [r['argmax'] for r in trace]:
            raise ValueError('Generated IDs and decisions must align')
        if [r['step'] for r in trace] != list(range(len(ids))):
            raise ValueError('Contiguous decision steps required')
        if any(not re.fullmatch(r'[a-f0-9]{64}', r[field])
               for r in trace for field in ['cache_sha256', 'full_logits_sha256']):
            raise ValueError('Exact cache and full-logit hashes required')
    left, right = native['generated_ids'], edited['generated_ids']
    prefix = 0
    while prefix < min(len(left), len(right)) and left[prefix] == right[prefix]:
        prefix += 1
    if prefix == min(len(left), len(right)) and len(left) != len(right):
        raise ValueError('Complete EOS sequence cannot extend an identical complete sequence')
    divergent = None if left == right else prefix
    steps = len(left) if divergent is None else divergent + 1
    a, b = native['decision_trace'], edited['decision_trace']
    mismatches = [i for i in range(steps) if a[i]['cache_sha256'] != b[i]['cache_sha256']]
    logit_differences = [i for i in range(steps) if a[i]['full_logits_sha256'] != b[i]['full_logits_sha256']]
    return {'same_consumed_input_decisions': steps, 'shared_emitted_token_prefix': prefix,
            'first_different_predicted_token_decision': divergent,
            'first_different_full_logit_decision': logit_differences[0] if logit_differences else None,
            'identical_complete_emitted_tokens': left == right,
            'all_shared_input_caches_exact': not mismatches,
            'cache_mismatch_decisions': mismatches,
            'decisions_after_divergent_token_consumption_excluded': True}
