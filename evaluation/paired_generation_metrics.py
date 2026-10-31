"""Account for every paired generation from caller-supplied labels and counts.

No model, tokenizer, grader or data lookup is performed. Exact count arithmetic
separates output reduction on all coding cases from reduction on programs that
were already correct. Unknown comment spans remain explicit uncertainty.
"""

from fractions import Fraction
import hashlib
import json

FIELDS = {'id', 'correct', 'at_cap', 'generated_tokens', 'comment_tokens', 'unknown_comment_tokens'}


def _exact(value):
    return None if value is None else {'numerator': value.numerator, 'denominator': value.denominator}


def _fraction(value):
    return Fraction(value['numerator'], value['denominator'])


def _mean(values):
    return sum(values, Fraction()) / len(values) if values else None


def _display(value):
    return None if value is None else float(value)


def _manifest(rows):
    output = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'id', 'role', 'family'}:
            raise ValueError('Only explicit case identity, role and family required')
        if any(not isinstance(row[k], str) or not row[k].strip() for k in row):
            raise ValueError('Nonempty text identity, role and family required')
        if row['role'] not in {'coding', 'retention'} or row['id'] in output:
            raise ValueError('Unique cases with a declared coding or retention role required')
        output[row['id']] = dict(row)
    if not output:
        raise ValueError('Nonempty case manifest required')
    return dict(sorted(output.items()))


def _records(rows, cases):
    output = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != FIELDS or not isinstance(row['id'], str) or row['id'] not in cases or row['id'] in output:
            raise ValueError('Exactly one complete record per immutable case ID required')
        if type(row['correct']) is not bool or type(row['at_cap']) is not bool:
            raise ValueError('Actual Boolean correctness and cap labels required')
        count = row['generated_tokens']
        if type(count) is not int or count <= 0:
            raise ValueError('A positive generated-token count required')
        for key in ['comment_tokens', 'unknown_comment_tokens']:
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError('Nonnegative integer classified and unknown counts required')
        if row['comment_tokens'] + row['unknown_comment_tokens'] > count:
            raise ValueError('Classified and uncertain comment counts must be disjoint and bounded')
        if row['correct'] and row['at_cap']:
            raise ValueError('An unfinished capped response cannot be labelled correct')
        output[row['id']] = dict(row)
    if set(output) != set(cases):
        raise ValueError('Every manifest case must remain in every condition')
    return dict(sorted(output.items()))


def _paired(ids, baseline, treatment):
    ratios, lower, upper = [], [], []
    gains, losses = [], []
    for identity in ids:
        b, t = baseline[identity], treatment[identity]
        ratios.append(Fraction(t['generated_tokens'], b['generated_tokens']))
        lower.append(Fraction(b['comment_tokens'], b['generated_tokens']) - Fraction(t['comment_tokens'] + t['unknown_comment_tokens'], t['generated_tokens']))
        upper.append(Fraction(b['comment_tokens'] + b['unknown_comment_tokens'], b['generated_tokens']) - Fraction(t['comment_tokens'], t['generated_tokens']))
        if t['correct'] and not b['correct']: gains.append(identity)
        if b['correct'] and not t['correct']: losses.append(identity)
    ratio, low, high = _mean(ratios), _mean(lower), _mean(upper)
    return {'cases': len(ids), 'original_correct': sum(baseline[i]['correct'] for i in ids),
            'correct': sum(treatment[i]['correct'] for i in ids), 'gains': gains, 'losses': losses,
            'caps': sum(treatment[i]['at_cap'] for i in ids),
            'generated_tokens': sum(treatment[i]['generated_tokens'] for i in ids),
            'unknown_comment_tokens': sum(treatment[i]['unknown_comment_tokens'] for i in ids),
            'mean_token_ratio': _display(ratio), 'mean_token_ratio_exact': _exact(ratio),
            'mean_comment_reduction_lower': _display(low), 'mean_comment_reduction_lower_exact': _exact(low),
            'mean_comment_reduction_upper': _display(high), 'mean_comment_reduction_upper_exact': _exact(high)}


def summarize(manifest, baseline, conditions):
    """Pair by ID, preserve all conditions and report exact rational witnesses.

    ``comment_tokens`` counts definite labels; ``unknown_comment_tokens`` covers
    every ambiguous span that could contain a comment. The producer remains
    responsible for those labels and for executing an independent program oracle.
    """
    cases = _manifest(manifest)
    base = _records(baseline, cases)
    if not isinstance(conditions, dict) or not conditions or any(not isinstance(name, str) or not name.strip() for name in conditions):
        raise ValueError('Explicit named conditions required')
    treatments = {name: _records(rows, cases) for name, rows in conditions.items()}
    coding = [i for i, c in cases.items() if c['role'] == 'coding']
    retention = [i for i, c in cases.items() if c['role'] == 'retention']
    originally_correct = [i for i in coding if base[i]['correct']]
    output = {}
    for name, treatment in treatments.items():
        families = {}
        for role, family in sorted({(c['role'], c['family']) for c in cases.values()}):
            cohort = [i for i, c in cases.items() if c['role'] == role and c['family'] == family]
            families[role + ':' + family] = _paired(cohort, base, treatment)
        output[name] = {'coding': _paired(coding, base, treatment),
                        'retention': _paired(retention, base, treatment),
                        'original_correct_coding': _paired(originally_correct, base, treatment),
                        'families': families}
    canonical = {'manifest': list(cases.values()), 'baseline': list(base.values()),
                 'conditions': {n: list(rows.values()) for n, rows in sorted(treatments.items())}}
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'input_sha256': digest, 'case_counts': {'coding': len(coding), 'retention': len(retention)},
            'baseline': {'coding': _paired(coding, base, base), 'retention': _paired(retention, base, base)},
            'conditions': output,
            'scope': 'Caller-supplied measured labels and counts, not independent grading, a model run, a semantic role or fresh confirmation.'}


def assess_n40_intent(summary):
    """Evaluate the predeclared conditional intent without choosing a winner.

    This is statistical accounting only. Source, tokenization, actual edits,
    state restoration and complete grading still require a separate frozen audit.
    """
    expected = {'selected_zero', 'randomA_zero', 'randomB_zero', 'prompt_only'}
    if summary['case_counts'] != {'coding': 32, 'retention': 16} or set(summary['conditions']) != expected:
        raise ValueError('The complete intended N40 panel must be present')
    if summary['baseline']['coding']['correct'] != 24 or summary['baseline']['retention']['correct'] != 7 or summary['baseline']['coding']['caps'] or summary['baseline']['retention']['caps']:
        raise ValueError('Frozen adequate-output original scores must be carried')
    selected = summary['conditions']['selected_zero']
    coding, retention, correct = [selected[k] for k in ['coding', 'retention', 'original_correct_coding']]
    if correct['cases'] != 24:
        raise ValueError('All originally-correct coding cases required')
    coding_families = [v for k, v in selected['families'].items() if k.startswith('coding:')]
    if len(coding_families) != 8 or any(v['cases'] != 4 for v in coding_families):
        raise ValueError('Eight dependent four-case coding families required')
    beneficial_families = sum(_fraction(v['mean_token_ratio_exact']) < 1 and _fraction(v['mean_comment_reduction_lower_exact']) > 0 for v in coding_families)
    clauses = {'preserve_original_correct_coding': not coding['losses'],
               'preserve_original_correct_retention': not retention['losses'],
               'no_new_caps': coding['caps'] + retention['caps'] == 0,
               'all32_comment_reduction_at_least_point10': _fraction(coding['mean_comment_reduction_lower_exact']) >= Fraction(1, 10),
               'all32_token_ratio_at_most_point90': _fraction(coding['mean_token_ratio_exact']) <= Fraction(9, 10),
               'correct24_comment_reduction_at_least_point05': _fraction(correct['mean_comment_reduction_lower_exact']) >= Fraction(1, 20),
               'correct24_token_ratio_at_most_point90': _fraction(correct['mean_token_ratio_exact']) <= Fraction(9, 10),
               'benefit_at_least_two_families': beneficial_families >= 2}
    for random in ['randomA_zero', 'randomB_zero']:
        for key in ['coding', 'original_correct_coding']:
            gap = _fraction(summary['conditions'][random][key]['mean_token_ratio_exact']) - _fraction(selected[key]['mean_token_ratio_exact'])
            clauses[key + '_ratio_advantage_over_' + random] = gap >= Fraction(1, 20)
    prompt = summary['conditions']['prompt_only']
    prompt_checks = {}
    for key in ['coding', 'retention']:
        prompt_checks[key + '_correct_not_lower'] = prompt[key]['correct'] >= selected[key]['correct']
        prompt_checks[key + '_baseline_losses_not_higher'] = len(prompt[key]['losses']) <= len(selected[key]['losses'])
        prompt_checks[key + '_caps_not_higher'] = prompt[key]['caps'] <= selected[key]['caps']
    for key in ['coding', 'original_correct_coding']:
        prompt_checks[key + '_token_ratio_not_higher'] = _fraction(prompt[key]['mean_token_ratio_exact']) <= _fraction(selected[key]['mean_token_ratio_exact'])
        prompt_checks[key + '_conservative_comment_reduction_not_lower'] = _fraction(prompt[key]['mean_comment_reduction_lower_exact']) >= _fraction(selected[key]['mean_comment_reduction_lower_exact'])
    return {'primary_clauses': clauses, 'primary_passed': all(clauses.values()),
            'beneficial_coding_families': beneficial_families,
            'prompt_comparison_clauses': prompt_checks,
            'simple_prompt_matches_or_improves_these_measures': all(prompt_checks.values()),
            'scope': 'Conditional intent criteria on supplied development labels. No automatic claim of accuracy gain, necessity, speedup or confirmed utility.'}
