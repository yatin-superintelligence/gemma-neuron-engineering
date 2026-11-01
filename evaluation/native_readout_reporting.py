"""Complete native-readout utility reporting without models or new grading."""
import copy
from statistics import median

CONDITIONS = ('original', 'subtract_native_boundary', 'add_native_boundary')
CLAUSES = ('preserve_every_original_correct_coding', 'preserve_every_original_correct_retention',
           'coding_gains_at_least4', 'gains_span_at_least2families',
           'beats_additive_control_by_at_least2')
ZERO = ('old_cell_carries', 'extra_attention_evaluations', 'extra_FFN_evaluations',
        'extra_PLE_evaluations', 'manual_parameter_assignments', 'base_buffer_assignments',
        'base_activation_hook_assignments', 'backward_calls', 'optimizer_steps')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def integer(value, lower=0, upper=None):
    return type(value) is int and value >= lower and (upper is None or value <= upper)


def expected_counters(fresh, prompt_sum):
    return {'calls': fresh+288, 'input_token_exposures': 5*prompt_sum+fresh-192,
            'fresh_choices': fresh, 'extra_native_head_evaluations': 2*fresh,
            'extra_native_final_RMS_evaluations': fresh, 'branch_input_constructions': fresh,
            'policy_support_assignments': fresh, 'native_sourceguards': 96,
            'native_restoration_forwards': 192, **dict.fromkeys(ZERO, 0)}


def length_summary(values):
    return {'responses': len(values), 'tokens_including_EOS': sum(values),
            'min': min(values), 'median': median(values), 'max': max(values),
            'ranges_are_not_confidence_intervals': True}


def panel(aggregate, rows, lengths, input_lengths):
    require(aggregate['experiment'] == 'N57' and aggregate['status'] == 'complete_host_audited',
            'Complete audited N57 aggregate required')
    for k, expected in [('records', 288), ('original_prediction_grade_carries', 96),
                        ('edited_response_records', 192), ('original_carry_new_N57_choices', 0),
                        ('old_cell_carried_choices', 0)]:
        require(integer(aggregate[k]) and aggregate[k] == expected, 'Exact response/carry accounting required')
    require(len(rows) == 96 and len({r['id'] for r in rows}) == 96
            and sum(r['role'] == 'coding' for r in rows) == 64
            and sum(r['role'] == 'retention' for r in rows) == 32
            and len({r['family'] for r in rows}) == 10, 'All original roles/families required')
    require(set(aggregate['conditions']) == set(CONDITIONS)
            and set(lengths) == {(c, r['id']) for c in CONDITIONS for r in rows}
            and set(input_lengths) == {r['id'] for r in rows}, 'All288 lengths and original prompts required')
    require(all(integer(v, 1) for v in [*lengths.values(), *input_lengths.values()]),
            'Actual complete lengths including EOS required')
    original_tokens = sum(lengths['original', r['id']] for r in rows)
    fresh = sum(v for (c, _), v in lengths.items() if c != 'original')
    require(integer(aggregate['fresh_choices']) and aggregate['fresh_choices'] == fresh
            and integer(aggregate['original_whole_response_carried_tokens'])
            and aggregate['original_whole_response_carried_tokens'] == original_tokens,
            'Exact original carried tokens and fresh choices required')
    counters = aggregate['actual_counters']
    expected = expected_counters(fresh, sum(input_lengths.values()))
    require(set(counters) == set(expected)
            and all(integer(counters[k]) and counters[k] == v for k, v in expected.items()),
            'Exact actual root/RMS/head/support/restoration accounting required')
    require(integer(aggregate['grading_calls_new_unique_candidates'], 0, 192)
            and integer(aggregate['exact_grade_carries_among192_newconditions'], 0, 192)
            and aggregate['grading_calls_new_unique_candidates']
            + aggregate['exact_grade_carries_among192_newconditions'] == 192,
            'Every edited record has one new or carried grade')
    original = aggregate['conditions']['original']
    require(original['coding_correct'] == 55 and original['retention_correct'] == 17,
            'Exact frozen original score carries required')
    families = sorted({r['family'] for r in rows})
    groups = {}
    for condition in CONDITIONS:
        g = aggregate['conditions'][condition]
        require(set(g['families']) == set(families), 'All10 families required')
        require(integer(g['coding_source_valid'], 0, 64)
                and g['coding_source_valid'] >= g['coding_correct'], 'Actual coding source-validity counts required')
        require(g['statuses'] and all(isinstance(k, str) and k and integer(v) for k, v in g['statuses'].items())
                and sum(g['statuses'].values()) == 96, 'Every grade status required')
        for role, total in [('coding', 64), ('retention', 32)]:
            correct, gains, losses = (g[role+'_'+suffix] for suffix in ['correct', 'gains', 'losses'])
            require(all(integer(v, 0, total) for v in [correct, gains, losses])
                    and gains <= total-original[role+'_correct'] and losses <= original[role+'_correct']
                    and correct == original[role+'_correct']+gains-losses, 'Actual grade conservation required')
            ff = {r['family'] for r in rows if r['role'] == role}
            for suffix in ['correct', 'gains', 'losses']:
                require(sum(g['families'][f][suffix] for f in ff) == g[role+'_'+suffix],
                        'All family grade and gain/loss totals must conserve')
        stats = {role: length_summary([lengths[condition, r['id']] for r in rows if r['role'] == role])
                 for role in ['coding', 'retention']}
        stats['families'] = {}
        for family in families:
            subset = [r for r in rows if r['family'] == family]
            f, old = g['families'][family], original['families'][family]
            require(len({r['role'] for r in subset}) == 1 and f['role'] == subset[0]['role']
                    and integer(f['cases'], 1) and f['cases'] == len(subset)
                    and all(integer(f[k], 0, len(subset)) for k in ['correct', 'gains', 'losses'])
                    and f['gains'] <= len(subset)-old['correct'] and f['losses'] <= old['correct']
                    and f['correct'] == old['correct']+f['gains']-f['losses'],
                    'Exact family roles/denominators and conservation required')
            if condition == 'original':
                require(f['gains'] == f['losses'] == 0, 'Original grades carry without reclassification')
            stats['families'][family] = length_summary([lengths[condition, r['id']] for r in subset])
        groups[condition] = {'grades': copy.deepcopy(g), 'response_lengths': stats, 'response_records': 96,
                             'fresh_N57_choices': 0 if condition == 'original' else sum(lengths[condition, r['id']] for r in rows),
                             'original_whole_response_carried_tokens': original_tokens if condition == 'original' else 0,
                             'old_cell_carried_choices': 0}
    primary = aggregate['connection_primary']
    target, control = (aggregate['conditions'][c] for c in CONDITIONS[1:])
    gain_families = sorted(f for f, v in target['families'].items() if v['role'] == 'coding' and v['gains'])
    actual_clauses = dict(zip(CLAUSES, [target['coding_losses'] == 0, target['retention_losses'] == 0,
                                      target['coding_gains'] >= 4, len(gain_families) >= 2,
                                      target['coding_correct'] >= control['coding_correct']+2]))
    require(primary['target'] == CONDITIONS[1] and primary['control'] == CONDITIONS[2]
            and primary['posthoc_additive_winner_allowed'] is False
            and primary['gain_families'] == gain_families and set(primary['clauses']) == set(CLAUSES)
            and all(type(primary['clauses'][k]) is bool and primary['clauses'][k] == v
                    for k, v in actual_clauses.items())
            and type(primary['passed']) is bool and primary['passed'] == all(actual_clauses.values()),
            'Every frozen fixed-target conjunctive primary clause must carry verbatim')
    return {'experiment': 'N57', 'status': 'complete_audited_reporting_only',
            'all_conditions': groups, 'families': families, 'connection_primary': copy.deepcopy(primary),
            'frozen_aggregate': copy.deepcopy(aggregate), 'actual_counters': copy.deepcopy(counters),
            'all_condition_records': 288, 'original_prediction_grade_carries': 96,
            'edited_response_records': 192, 'original_whole_response_carried_tokens': original_tokens,
            'fresh_N57_choices': fresh, 'old_cell_carried_choices': 0,
            'scope': 'All96/all3 reused-development complete programs. No winner, pvalues, individual meaning, mediation, speed or fresh confirmation. Original whole-response carry tokens are zero N57 choices. Min/max ranges are descriptive, not confidence intervals.'}


def render(data, destination, *, synthetic=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    require(not destination.exists(), 'Never overwrite an actual figure or preview')
    labels = ['Native carry', 'Subtract target', 'Add control']
    colors = ['#78848c', '#00766c', '#bb6a28']
    groups = data['all_conditions']
    fig, axes = plt.subplots(3, 2, figsize=(17, 16), layout='constrained')
    fig.suptitle(('SYNTHETIC SOFTWARE LAYOUT / ' if synthetic else '')
                 + 'N57: fixed native-state output connection', fontsize=18, fontweight='bold')
    x = np.arange(3)
    for ax, role, total, title in [(axes[0, 0], 'coding', 64, 'A  Coding: correct / 64'),
                                   (axes[0, 1], 'retention', 32, 'B  Retention: correct / 32')]:
        values = [groups[c]['grades'][role+'_correct'] for c in CONDITIONS]
        ax.bar(x, values, color=colors, width=.62)
        for j, value in enumerate(values):
            ax.text(j, value+total*.025, str(value), ha='center', fontsize=12, fontweight='bold')
        ticks = [label+'\n+'+str(groups[c]['grades'][role+'_gains'])+' gains / '
                 +str(groups[c]['grades'][role+'_losses'])+' losses' for label, c in zip(labels, CONDITIONS)]
        ax.set_xticks(x, ticks, fontsize=9)
        ax.set(title=title, ylim=(0, total*1.16), ylabel='Complete correct cases')
    ax = axes[1, 0]
    for j, c in enumerate(CONDITIONS[1:]):
        g = groups[c]['grades']
        ax.bar(j, g['coding_gains'], color=colors[j+1], width=.55)
        ax.bar(j, -g['coding_losses'], color=colors[j+1], width=.55, hatch='xx', alpha=.55)
        ax.text(j, g['coding_gains']+.35, '+'+str(g['coding_gains']), ha='center', fontsize=11)
        ax.text(j, -g['coding_losses']-.35, '-'+str(g['coding_losses']), ha='center', va='top', fontsize=11)
    ax.axhline(0, color='#30434e', lw=.8)
    ax.set_xticks([0, 1], labels[1:])
    ax.set(title='C  Every coding gain and loss', ylabel='Cases relative to carried native')
    ax.set_ylim(-max([groups[c]['grades']['coding_losses'] for c in CONDITIONS[1:]]+[1])-1.5,
                max([groups[c]['grades']['coding_gains'] for c in CONDITIONS[1:]]+[1])+1.5)
    ax = axes[1, 1]
    for j, role in enumerate(['coding', 'retention']):
        stats = [groups[c]['response_lengths'][role] for c in CONDITIONS]
        med = np.array([s['median'] for s in stats])
        low, high = np.array([s['min'] for s in stats]), np.array([s['max'] for s in stats])
        ax.errorbar(x+(-.1 if j == 0 else .1), med, yerr=[med-low, high-med],
                    fmt='o', capsize=4, color='#00766c' if j == 0 else '#bb6a28', label=role)
    ax.set_yscale('log')
    ax.set_xticks(x, labels, fontsize=9)
    ax.set(title='D  Length: median and min to max', ylabel='Tokens including EOS (log scale)')
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.02), ncol=2, fontsize=9)
    ax.set_title('D  Length: median and min to max', pad=38)
    ax = axes[2, 0]
    families = data['families']
    fractions = np.array([[groups[c]['grades']['families'][f]['correct']
                           / groups[c]['grades']['families'][f]['cases'] for c in CONDITIONS] for f in families])
    ax.imshow(fractions, vmin=0, vmax=1, cmap='YlGnBu', aspect='auto')
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_yticks(np.arange(10), [('Code: ' if groups['original']['grades']['families'][f]['role'] == 'coding'
                                 else 'Retain: ')+f.replace('_', ' ') for f in families], fontsize=8)
    ax.set_title('E  All families: correct / cases')
    for i, f in enumerate(families):
        for j, c in enumerate(CONDITIONS):
            v = groups[c]['grades']['families'][f]
            ax.text(j, i, str(v['correct'])+'/'+str(v['cases']), ha='center', va='center', fontsize=9,
                    color='white' if fractions[i, j] > .6 else 'black')
    ax = axes[2, 1]
    ax.set_axis_off()
    names = ['Preserve every native-correct coding case', 'Preserve every native-correct retention case',
             'At least four coding gains', 'Coding gains span at least two families',
             'At least two correct programs above add control']
    primary = data['connection_primary']
    ax.set_title('F  Subtract target: all five clauses required', pad=18)
    for i, (key, name) in enumerate(zip(CLAUSES, names)):
        yes = primary['clauses'][key]
        y = .91-i*.145
        ax.text(.01, y, name, transform=ax.transAxes, fontsize=10, va='center')
        ax.text(.99, y, 'PASS' if yes else 'FAIL', transform=ax.transAxes, fontsize=11,
                fontweight='bold', ha='right', va='center', color='#006c56' if yes else '#a53932')
    ax.text(.01, .12, 'Overall: '+('PASS' if primary['passed'] else 'FAIL'), transform=ax.transAxes,
            fontsize=13, fontweight='bold', color='#006c56' if primary['passed'] else '#a53932')
    ax.text(.01, .02, 'Addition remains the fixed control.', transform=ax.transAxes, fontsize=10)
    counters = data['actual_counters']
    fig.supxlabel('All96 cases / all3 conditions; reused development. NativeEOS required; no arbitrary response ceiling.\n'
                  '192 edited response records;96 original response/grade carries; '+str(data['fresh_N57_choices'])
                  +' fresh choices;0 old edited-cell carries; '+str(data['original_whole_response_carried_tokens'])
                  +' original carry tokens (0 N57 choices).\n'
                  'Actual '+str(counters['calls'])+' root forwards; '+str(counters['input_token_exposures'])
                  +' input exposures; '+str(counters['extra_native_head_evaluations'])+' extra native heads; '
                  +str(counters['extra_native_final_RMS_evaluations'])+' extra native RMS evaluations.\n'
                  'Ranges are descriptive, not CIs. No meaning, mediation, speed, fresh-confirmation or general coding-gain claim.',
                  fontsize=9)
    fig.savefig(destination, dpi=150)
    plt.close(fig)
