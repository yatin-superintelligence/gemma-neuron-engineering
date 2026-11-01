"""Validate a complete saved normalization aggregate without recomputing science."""

from fractions import Fraction
import math

GROUPS = ['selected32', 'randomA32', 'randomB32']
CELLS = ['00', '10', '01', '11']
SITES = [24, 25, 26, 27, 29, 30, 32, 33, 34]
FAMILY_CASES = {'arithmetic': 16, 'ceil_ratio': 8, 'first_prefix': 8, 'json': 16,
    'long_tokens': 8, 'parenthesis_depth': 8, 'rotate_right': 8, 'run_lengths': 8,
    'stable_unique': 8, 'union_length': 8}
FAMILY_POINTS = {name: (48 if name == 'arithmetic' else 41 if name == 'json' else 24)
    for name in FAMILY_CASES}
CELL_METRICS = ['choice_change_rate', 'native_margin_change',
    'native_winner_probability_change', 'logit_delta_L2', 'logit_delta_max_abs']


def require(value, message):
    if not value:
        raise ValueError(message)


def mean(item, cases, *, partially_defined=False):
    require(isinstance(item, dict), 'Missing saved audited mean')
    value = item['case_mean']
    count = item['cases_with_defined_values']
    require(type(value) in (int, float) and math.isfinite(value), 'Nonfinite mean')
    require(type(count) is int and 0 < count <= cases, 'Invalid defined-case count')
    require(partially_defined or count == cases, 'Missing cases from required mean')
    fraction = item['exact_binary_float_mean']
    require(isinstance(fraction, list) and len(fraction) == 2
        and all(type(v) is int for v in fraction) and fraction[1] > 0, 'Missing exact mean fraction')
    require(float(Fraction(*fraction)) == value, 'Saved mean and exact fraction disagree')
    return value


def aggregate(item, cases, records):
    require(item['cases'] == cases and item['group_site_decision_records'] == records,
        'Incomplete fixed cohort or sampled decisions')
    require(set(item['cells']) == set(CELLS), 'Every cell, including native00, is required')
    for cell in CELLS:
        require(set(item['cells'][cell]) == set(CELL_METRICS), 'Missing fixed cell metric')
        for metric, saved in item['cells'][cell].items():
            value = mean(saved, cases)
            if metric == 'choice_change_rate':
                require(0 <= value <= 1, 'Invalid choice-change rate')
            elif metric == 'native_winner_probability_change':
                require(-1 <= value <= 1, 'Invalid probability change')
            elif metric.startswith('logit_delta'):
                require(value >= 0, 'Negative logit magnitude')
            if cell == '00':
                require(value == 0, 'Native diagonal is not exact zero')
    require(mean(item['interaction_L2'], cases) >= 0, 'Negative interaction magnitude')
    mean(item['interaction_margin'], cases)
    for key in ['zero_raw_path_records', 'zero_normalization_path_records']:
        require(type(item[key]) is int and 0 <= item[key] <= records, 'Invalid zero-path count')
    cosine = item['raw_normalization_cosine']
    if cosine is not None:
        require(-1-1e-12 <= mean(cosine, cases, partially_defined=True) <= 1+1e-12,
            'Invalid saved path cosine')


def validate(data):
    """Require the complete original audit; this is no replacement for that audit."""
    execution = data['execution']
    require(execution['status'] == 'complete_host_audited_closed'
        and execution['all_exact_guards_pass'] is True, 'No figure from incomplete or failed engineering')
    expected = {'cases': 96, 'sampled_decisions': 281, 'group_site_decision_records': 7587,
        'actual_model_forward_calls': 61645, 'actual_input_token_exposures': 185160,
        'manual_parameter_assignments': 15174, 'activation_assignments': 25290,
        'native_restorations': 7587, 'new_generated_tokens': 0, 'backward_calls': 0, 'optimizer_steps': 0}
    require(all(type(execution[k]) is int and execution[k] == v for k, v in expected.items()),
        'Original complete manifest and counters required')
    require(data['no_metric_or_cell_selected_out'] is True and data['families_and_bindings_dependent'] is True,
        'All-cell and dependency disclosures required')
    for scope in ['all_groups', 'all_group_sites', 'all_group_families', 'all_group_roles']:
        require(set(data[scope]) == set(GROUPS), 'Selected and both random groups required')
    for group in GROUPS:
        aggregate(data['all_groups'][group], 96, 2529)
        sites, families, roles = (data[name][group] for name in
            ['all_group_sites', 'all_group_families', 'all_group_roles'])
        require(set(sites) == {str(s) for s in SITES}, 'All nine fixed sites required')
        require(set(families) == set(FAMILY_CASES), 'All ten fixed families required')
        require(set(roles) == {'coding', 'retention'}, 'Both coding and retention required')
        for site in sites.values():
            aggregate(site, 96, 281)
        for name, family in families.items():
            aggregate(family, FAMILY_CASES[name], 9*FAMILY_POINTS[name])
        aggregate(roles['coding'], 64, 1728)
        aggregate(roles['retention'], 32, 801)
    return data
