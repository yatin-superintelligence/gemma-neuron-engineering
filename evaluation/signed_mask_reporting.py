"""Validate the fixed complete N45 reporting projection without model imports."""
import hashlib
import json
import math

CONDITIONS = ('original', 'remove_negative', 'random_negative_count', 'remove_positive', 'random_positive_count')
METRICS = {'argmax_changed', 'native_margin', 'native_winner_probability', 'delta_L2', 'delta_maxabs', 'mask_count', 'removed_h_energy_fraction'}
COUNTERS = {'calls': 2810, 'input_token_exposures': 106120, 'activation_assignments': 1405,
            'extra_FFN_evaluations': 0, 'new_generated_tokens': 0, 'manual_parameter_assignments': 0,
            'backward_calls': 0, 'optimizer_steps': 0}


def validate(data, receipt, plan):
    if (receipt.get('status') != 'complete_pass_all_frozen_guards'
            or receipt.get('all96_cases_all5conditions') is not True
            or receipt.get('public_aggregate_sha256') != hashlib.sha256(data).hexdigest()):
        raise ValueError('Complete frozen audit and exact aggregate hash required')
    a = json.loads(data)
    if (a.get('experiment') != 'N45' or a.get('status') != 'complete_host_audited'
            or type(a.get('cases')) is not int or a['cases'] != 96
            or type(a.get('sampled_decisions')) is not int or a['sampled_decisions'] != 281
            or a['conditions'] != list(CONDITIONS)
            or set(a['strata']) != set(plan['strata'])):
        raise ValueError('Every original condition, case and stratum required')
    for counters in [a['actual_counters'], receipt['actual_counters']]:
        if set(counters) != set(COUNTERS) or any(type(counters[k]) is not int or counters[k] != v for k, v in COUNTERS.items()):
            raise ValueError('Exact actual work counts required')
    for name, expected in plan['strata'].items():
        group = a['strata'][name]
        if any(type(group[k]) is not int or group[k] != expected[k] for k in ['cases', 'sampled_decisions']) or set(group['conditions']) != set(CONDITIONS):
            raise ValueError('Fixed cohort and all conditions required')
        for condition in CONDITIONS:
            metrics = group['conditions'][condition]
            if set(metrics) != METRICS: raise ValueError('Every frozen metric required')
            for key, value in metrics.items():
                if set(value) != {'case_mean', 'defined_cases', 'undefined_cases'}: raise ValueError('Exact mean and null provenance required')
                for field in ['defined_cases', 'undefined_cases']:
                    if type(value[field]) is not int or value[field] < 0: raise ValueError('Strict case counts required')
                if value['defined_cases']+value['undefined_cases'] != group['cases']: raise ValueError('Every case accounted for')
                mean = value['case_mean']
                if value['defined_cases'] == 0:
                    if mean is not None: raise ValueError('Undefined mean must stay null')
                elif type(mean) not in (float, int) or not math.isfinite(mean): raise ValueError('Finite audited mean required')
                if key != 'removed_h_energy_fraction' and value['undefined_cases']: raise ValueError('Unexpected missing finite metric')
                if mean is not None:
                    if key != 'native_margin' and mean < 0: raise ValueError('Negative nonnegative metric')
                    if key in ['argmax_changed', 'native_winner_probability', 'removed_h_energy_fraction'] and mean > 1: raise ValueError('Fraction outside0..1')
                    if key == 'mask_count' and mean > 12288: raise ValueError('Mask count exceeds native width')
                    if condition == 'original' and key in ['argmax_changed', 'delta_L2', 'delta_maxabs', 'mask_count', 'removed_h_energy_fraction'] and mean != 0:
                        raise ValueError('Original cannot differ from itself')
    return a
