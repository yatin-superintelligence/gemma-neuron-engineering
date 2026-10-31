"""Independent synthetic checks; no model, tokenizer, program or scorer runs."""

import copy
from fractions import Fraction
import json
import random

import pytest

from evaluation.paired_generation_metrics import summarize, assess_n40_intent


def panel():
    manifest = [{'id': f'coding_{i:02}', 'role': 'coding', 'family': f'family_{i // 4}'} for i in range(32)]
    manifest += [{'id': f'retention_{i:02}', 'role': 'retention', 'family': 'arithmetic' if i < 8 else 'json'} for i in range(16)]
    baseline = []
    for i, row in enumerate(manifest):
        baseline.append({'id': row['id'], 'correct': i < 24 if i < 32 else i - 32 < 7,
                         'at_cap': False, 'generated_tokens': 100,
                         'comment_tokens': 30 if row['role'] == 'coding' else 0, 'unknown_comment_tokens': 0})
    conditions = {name: copy.deepcopy(baseline) for name in ['selected_zero', 'randomA_zero', 'randomB_zero', 'prompt_only']}
    for name, tokens in [('selected_zero', 80), ('prompt_only', 75)]:
        for row in conditions[name][:32]:
            row.update(generated_tokens=tokens, comment_tokens=0)
    return manifest, baseline, conditions


def exact(item):
    return Fraction(item['numerator'], item['denominator'])


def test_all_conditions_and_cases_with_prompt_control_preserved():
    m, b, c = panel()
    result = summarize(m, b, c)
    assessed = assess_n40_intent(result)
    assert list(result['conditions']) == list(c)
    assert result['case_counts'] == {'coding': 32, 'retention': 16}
    assert assessed['primary_passed']
    assert assessed['simple_prompt_matches_or_improves_these_measures']
    assert result['conditions']['selected_zero']['coding']['gains'] == []
    assert result['conditions']['selected_zero']['coding']['correct'] == 24
    assert result['conditions']['selected_zero']['retention']['correct'] == 7
    assert assessed['beneficial_coding_families'] == 8


def test_mean_of_ratios_not_ratio_of_pooled_totals():
    m = [{'id': 'a', 'role': 'coding', 'family': 'one'}, {'id': 'b', 'role': 'coding', 'family': 'two'}]
    b = [{'id': i, 'correct': True, 'at_cap': False, 'generated_tokens': n, 'comment_tokens': 0, 'unknown_comment_tokens': 0} for i, n in [('a', 10), ('b', 1000)]]
    t = copy.deepcopy(b);t[0]['generated_tokens'] = 20;t[1]['generated_tokens'] = 500
    value = summarize(m, b, {'treatment': t})['conditions']['treatment']['coding']
    assert exact(value['mean_token_ratio_exact']) == Fraction(5, 4)
    assert value['mean_token_ratio'] != Fraction(520, 1010)


def test_pair_by_id_with_permutation_invariance_and_no_input_mutation():
    m, b, c = panel();before = json.dumps([m, b, c], sort_keys=True)
    expected = summarize(m, b, c)
    assert before == json.dumps([m, b, c], sort_keys=True)
    rng = random.Random(4041);rng.shuffle(m);rng.shuffle(b)
    for records in c.values():rng.shuffle(records)
    assert summarize(m, b, c) == expected


def test_only_wrong_program_shortening_cannot_pass():
    m, b, c = panel()
    for row in b[:32]:row['comment_tokens'] = 30 if row['correct'] else 90
    for name in ['randomA_zero', 'randomB_zero']:
        c[name] = copy.deepcopy(b)
    c['selected_zero'] = copy.deepcopy(b)
    for row in c['selected_zero'][:32]:
        if not row['correct']:row.update(generated_tokens=1, comment_tokens=0)
    result = assess_n40_intent(summarize(m, b, c))
    assert result['primary_clauses']['all32_token_ratio_at_most_point90']
    assert result['primary_clauses']['all32_comment_reduction_at_least_point10']
    assert not result['primary_clauses']['correct24_token_ratio_at_most_point90']
    assert not result['primary_clauses']['correct24_comment_reduction_at_least_point05']
    assert not result['primary_passed']


def test_uncertain_comment_spans_remain_conservative_bounds():
    m, b, c = panel()
    for row in c['selected_zero'][:32]:row['unknown_comment_tokens'] = 25
    summary = summarize(m, b, c)
    item = summary['conditions']['selected_zero']['coding']
    assert exact(item['mean_comment_reduction_lower_exact']) == Fraction(-1, 80)
    assert exact(item['mean_comment_reduction_upper_exact']) == Fraction(3, 10)
    assert not assess_n40_intent(summary)['primary_passed']


def test_exact_point05_threshold_has_no_float_tolerance():
    m, b, c = panel()
    for row in c['selected_zero'][:32]:row.update(generated_tokens=30, comment_tokens=0)
    for name in ['randomA_zero', 'randomB_zero']:
        for row in c[name][:32]:row.update(generated_tokens=35, comment_tokens=0)
    result = assess_n40_intent(summarize(m, b, c))
    assert result['primary_passed']
    assert result['primary_clauses']['coding_ratio_advantage_over_randomA_zero']
    c['randomA_zero'][0]['generated_tokens'] = 34
    result = assess_n40_intent(summarize(m, b, c))
    assert not result['primary_clauses']['coding_ratio_advantage_over_randomA_zero']
    assert not result['primary_passed']


def test_one_family_cannot_replace_required_coverage():
    m, b, c = panel()
    for row in b[:32]:row['comment_tokens'] = 90 if row['id'] in {f'coding_{i:02}' for i in range(4)} else 0
    for name in c:c[name] = copy.deepcopy(b)
    for row in c['selected_zero'][:4]:row.update(generated_tokens=10, comment_tokens=0)
    result = assess_n40_intent(summarize(m, b, c))
    assert result['beneficial_coding_families'] == 1
    assert not result['primary_clauses']['benefit_at_least_two_families']
    assert all(v for k, v in result['primary_clauses'].items() if k != 'benefit_at_least_two_families')
    assert not result['primary_passed']


@pytest.mark.parametrize('role,index', [('coding', 0), ('retention', 32)])
def test_any_loss_of_original_correctness_is_retained(role, index):
    m, b, c = panel();identity = c['selected_zero'][index]['id'];c['selected_zero'][index]['correct'] = False
    summary = summarize(m, b, c)
    assert summary['conditions']['selected_zero'][role]['losses'] == [identity]
    assert not assess_n40_intent(summary)['primary_passed']


def test_capped_originally_wrong_response_is_still_a_primary_failure():
    m, b, c = panel();c['selected_zero'][24]['at_cap'] = True
    result = assess_n40_intent(summarize(m, b, c))
    assert not result['primary_clauses']['no_new_caps'] and not result['primary_passed']


@pytest.mark.parametrize('kind', ['duplicate', 'missing', 'extra', 'boolean', 'zero_tokens', 'negative_comments', 'overlapping_counts', 'correct_cap', 'nontext_id', 'missing_unknown'])
def test_malformed_or_incomplete_records_refused(kind):
    m, b, c = panel();rows = c['selected_zero']
    if kind == 'duplicate':rows.append(copy.deepcopy(rows[0]))
    elif kind == 'missing':rows.pop()
    elif kind == 'extra':rows[0]['id'] = 'unknown'
    elif kind == 'boolean':rows[0]['correct'] = 1
    elif kind == 'zero_tokens':rows[0]['generated_tokens'] = 0
    elif kind == 'negative_comments':rows[0]['comment_tokens'] = -1
    elif kind == 'overlapping_counts':rows[0].update(comment_tokens=70, unknown_comment_tokens=20)
    elif kind == 'correct_cap':rows[0]['at_cap'] = True
    elif kind == 'nontext_id':rows[0]['id'] = []
    else:del rows[0]['unknown_comment_tokens']
    with pytest.raises(ValueError):summarize(m, b, c)


def test_empty_manifest_and_missing_prompt_control_refused():
    m, b, c = panel()
    with pytest.raises(ValueError):summarize([], [], c)
    del c['prompt_only']
    with pytest.raises(ValueError):assess_n40_intent(summarize(m, b, c))
