"""Render all N30 units, fixed primary components and program conditions."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

UNITS = ['target', 'random_A', 'random_B']
COLORS = ['#087f8c', '#805eaa', '#bc7640']
OPS = ['opposite', 'samepolicy', 'zero', 'conditional_cut']
OP_LABELS = ['Opposite donor', 'Same-rule donor', 'Native zero', 'Conditional cut']


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=root / 'runs/n30/version1/public_aggregate.json')
    parser.add_argument('--output', type=Path, default=root / 'research/figures/n30_summary')
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    finite = data['finite']
    assert len(data['conditions']) == 19
    assert not finite['primary_finite_bias_passed']
    plt.rcParams.update({'font.size': 10, 'svg.hashsalt': 'n30-fixed-all-results'})
    fig = plt.figure(figsize=(18, 20), layout='constrained')
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 2.25, .45])
    for panel, (cohort, title) in enumerate([
        ('all13', 'All 13 inherited role cases'),
        ('discovery8', '8 discovery cases, wording 0'),
        ('other5', '5 other development cases, wording 1')]):
        axis = fig.add_subplot(gs[0, panel * 2:panel * 2 + 2])
        for unit_index, unit in enumerate(UNITS):
            values = [finite['groups'][unit][op][cohort]['mean'] for op in OPS]
            axis.bar(np.arange(4) + (unit_index - 1) * .23, values, width=.22,
                     color=COLORS[unit_index], label=unit.replace('_', ' '))
        axis.axhline(0, color='#666666', linewidth=.8)
        axis.set_xticks(np.arange(4), OP_LABELS, rotation=26, ha='right')
        axis.set_title(title, loc='left', weight='bold', fontsize=12)
        axis.set_ylabel('Mean opposite-minus-original logit-margin change')
        axis.grid(axis='y', alpha=.15)
        axis.legend(frameon=False, fontsize=9)
    axis = fig.add_subplot(gs[1, :3])
    components = finite['all_primary_components']
    specific = finite['same_rule_specificity']
    assert len(components) + len(specific) == 27
    labels = [x['operation'].replace('conditional_cut', 'cut') + ' | ' +
              x['cohort'].replace('discovery_mapping', 'disc m').replace('_label', ' l')
              for x in components]
    labels += ['Same-rule specificity | ' + x['cohort'] for x in specific]
    passes = [x['passed'] for x in components + specific]
    assert sum(passes) == 26
    axis.barh(np.arange(27), [1] * 27,
              color=['#c5e5dc' if value else '#df8e8e' for value in passes])
    for i, value in enumerate(passes):
        axis.text(.5, i, 'PASS' if value else 'FAIL: target = control B = 0.015625',
                  va='center', ha='center', fontsize=8)
    axis.set_yticks(np.arange(27), labels, fontsize=8.8)
    axis.set_xticks([])
    axis.set_xlim(0, 1)
    axis.invert_yaxis()
    axis.set_title('Every predefined component: conjunction FAILS (26/27 pass)',
                   loc='left', weight='bold', fontsize=12, pad=12)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis = fig.add_subplot(gs[1, 3:])
    axis.axis('off')
    rows = []
    for row in data['conditions']:
        if row['retention_nonapplicable']:
            retention = 'N/A'
        elif row['retention_new_correct'] is not None:
            retention = str(row['retention_new_correct']) + '/8 new'
        else:
            retention = str(row['retention_reference_correct']) + '/8 ref'
        rows.append([row['condition'].replace('conditional_cut', 'cut'),
                     row['role_correct'], row['opposite_correct'], retention,
                     '-' if row['clean_ablation_losses'] is None else row['clean_ablation_losses'],
                     '-' if row['conditional_transfer_losses'] is None else row['conditional_transfer_losses'],
                     str(row['coding_gains']) + '/' + str(row['capped_new'])])
    table = axis.table(cellText=rows,
                       colLabels=['Policy', 'Own\n/13', 'Opp.\n/13', 'Retention',
                                  'Clean\nloss', 'Cut\nloss', 'Gain/\ncap'],
                       colWidths=[.38, .08, .08, .16, .09, .09, .09],
                       bbox=[0, .04, 1, .94], cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row_index, _), cell in table.get_celld().items():
        cell.set_edgecolor('#d2d8dc')
        cell.set_facecolor('#e4edf1' if row_index == 0 else ('#f5f8f8' if row_index % 2 else '#ffffff'))
    axis.set_title('All 19 conditions; no functional effects from any unit',
                   loc='left', weight='bold', fontsize=12, pad=12)
    axis = fig.add_subplot(gs[2, :])
    axis.axis('off')
    r = data['receipt']
    axis.text(0, .95,
        '399 records = 180 new tails + 84 exact carries + 63 guarded rescues + 72 new N/A references.\n'
        'The carries include 16 inherited donor-retention N/A cells. References are not new generations.\n'
        '390 trajectories; 805 native hashes; 46 prior scalar comparisons; 47 parent, 63 rescue, 39 conditional-self and 38 restoration checks pass.\n'
        '2638 forwards; 33314 input-token exposures; 800 new tokens; zero backwards and optimizer updates.\n'
        f"Worker {r['worker']['seconds']:.3f} s; notebook {r['notebook_seconds']:.3f} s. These durations are not charged quota.\n"
        'Reused, dependent development cases. A cut tests an engineered donor background; no native program necessity, coding gain or semantic role.\n'
        'Conditional-cut margins are full-opposite minus cut; other bars are edited minus native. All raw vector statistics remain GPU witnesses.',
        va='top', fontsize=10.5, linespacing=1.65)
    fig.suptitle('N30 | One fixed final-layer channel: measured score effects, failed specificity criterion',
                 fontsize=17, weight='bold')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ['png', 'svg']:
        fig.savefig(args.output.with_suffix('.' + suffix), dpi=140,
                    metadata={'Date': None} if suffix == 'svg' else {})
    plt.close(fig)


if __name__ == '__main__':
    main()
