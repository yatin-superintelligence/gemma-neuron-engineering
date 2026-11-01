"""Strict all-condition first-answer scoring; no model or candidate executor."""
from collections import Counter
import math

CONDITIONS = ('original', 'remove_negative', 'random_negative_count', 'remove_positive', 'random_positive_count')


def assess(rows, records, secrets, answer_ids):
    ids = {r['id'] for r in rows}
    if len(rows) != 24 or len(ids) != 24 or set(secrets) != ids:
        raise ValueError('Exact 24 fresh contexts and labels required')
    if Counter(r['role'] for r in rows) != {'coding': 16, 'retention': 8}:
        raise ValueError('Exact role denominators required')
    if set(records) != {(c, i) for c in CONDITIONS for i in ids}:
        raise ValueError('All five complete conditions required')
    if set(answer_ids) != {'A', 'B'} or any(type(v) is not int or v < 0 for v in answer_ids.values()) or len(set(answer_ids.values())) != 2:
        raise ValueError('Strict canonical A/B IDs required')
    pairs = {}
    for r in rows:
        pairs.setdefault(r['pair_id'], []).append(r)
        if secrets[r['id']]['correct_letter'] not in ['A', 'B']:
            raise ValueError('Exact hidden literal letter required')
    if len(pairs) != 12 or any(len(v) != 2 or {r['binding'] for r in v} != {0, 1} or len({r['role'] for r in v}) != 1 or len({r['family'] for r in v}) != 1 or {secrets[r['id']]['correct_letter'] for r in v} != {'A', 'B'} for v in pairs.values()):
        raise ValueError('All twelve exact answer-order pairs required')
    correct, valid, margins = {}, {}, {}
    for c in CONDITIONS:
        for row in rows:
            m = records[c, row['id']]['metrics']; token = m['argmax']
            if type(token) is not int or token < 0:
                raise ValueError('Strict full-vocabulary argmax required')
            choice = next((k for k, v in answer_ids.items() if v == token), 'other')
            if m['choice'] != choice:
                raise ValueError('No restricted option choice or alias allowed')
            if set(m['answer_logits']) != {'A', 'B'} or set(m['answer_probabilities']) != {'A', 'B'}:
                raise ValueError('Both native option scores required')
            if any(type(v) not in [int, float] or not math.isfinite(v) for v in m['answer_logits'].values()) or any(type(v) not in [int, float] or not math.isfinite(v) or not 0 <= v <= 1 for v in m['answer_probabilities'].values()):
                raise ValueError('Finite actual option scores required')
            if abs(m['A_minus_B_margin']-(m['answer_logits']['A']-m['answer_logits']['B'])) > 1e-6:
                raise ValueError('Actual option margin differs')
            gold = secrets[row['id']]['correct_letter']
            correct[c, row['id']] = choice == gold
            valid[c, row['id']] = choice != 'other'
            margins[c, row['id']] = m['A_minus_B_margin']*(1 if gold == 'A' else -1)
    summary = {'conditions': {}, 'signed_policy_primary': {}, 'scope': 'Fresh host-authored development answer-decision panel;not full-program coding improvement or independent benchmark confirmation.'}
    for c in CONDITIONS:
        group = {'choices': dict(Counter(records[c, r['id']]['metrics']['choice'] for r in rows)), 'families': {}}
        for role in ['coding', 'retention']:
            subset = [r for r in rows if r['role'] == role]
            group[role+'_cases'] = len(subset)
            group[role+'_correct'] = sum(correct[c, r['id']] for r in subset)
            group[role+'_valid_options'] = sum(valid[c, r['id']] for r in subset)
            group[role+'_gains'] = sum(not correct['original', r['id']] and correct[c, r['id']] for r in subset)
            group[role+'_losses'] = sum(correct['original', r['id']] and not correct[c, r['id']] for r in subset)
            group[role+'_correct_margin_case_mean'] = sum(margins[c, r['id']] for r in subset)/len(subset)
        for f in sorted({r['family'] for r in rows}):
            subset = [r for r in rows if r['family'] == f]
            group['families'][f] = {'role': subset[0]['role'], 'cases': len(subset),
                'correct': sum(correct[c, r['id']] for r in subset),
                'both_orders_correct': all(correct[c, r['id']] for r in subset)}
        summary['conditions'][c] = group
    original = summary['conditions']['original']
    eligibility = {'original_coding_valid_options_at_least14': original['coding_valid_options'] >= 14,
        'original_retention_valid_options_at_least7': original['retention_valid_options'] >= 7,
        'original_coding_correct_at_least8': original['coding_correct'] >= 8,
        'original_retention_correct_at_least6': original['retention_correct'] >= 6}
    summary['baseline_eligibility'] = eligibility
    for sign in ['negative', 'positive']:
        c = 'remove_'+sign; g = summary['conditions'][c]
        old = {p: all(correct['original', r['id']] for r in rs) for p, rs in pairs.items() if rs[0]['role'] == 'coding'}
        new = {p: all(correct[c, r['id']] for r in pairs[p]) for p in old}
        gain_families = sorted({pairs[p][0]['family'] for p in old if not old[p] and new[p]})
        clauses = {**eligibility, 'preserve_all_original_correct_coding': g['coding_losses'] == 0,
            'preserve_all_original_correct_retention': g['retention_losses'] == 0,
            'coding_gains_at_least2': g['coding_gains'] >= 2,
            'both_order_program_gains_at_least2': sum(not old[p] and new[p] for p in old) >= 2,
            'gain_families_at_least2': len(gain_families) >= 2,
            'zero_both_order_program_losses': not any(old[p] and not new[p] for p in old),
            'beats_matched_random_by_at_least2': g['coding_correct'] >= summary['conditions']['random_'+sign+'_count']['coding_correct']+2}
        summary['signed_policy_primary'][c] = {'clauses': clauses, 'passed': all(clauses.values()), 'gain_families': gain_families}
    return summary
