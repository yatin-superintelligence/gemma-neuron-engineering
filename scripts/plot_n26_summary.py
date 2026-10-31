"""Plot every N26 condition and predetermined contrast from safe aggregates."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ORDER = ['base', 'identity_selected64_gain1', 'selected64_gain0', 'random34201_64_gain0',
         'random34202_64_gain0', 'selected64_gain2', 'random34201_64_gain2', 'random34202_64_gain2']
LABELS = ['Original (C)', 'Identity (G)', 'Selected 0', 'Random A0', 'Random B0',
          'Selected 2', 'Random A2', 'Random B2']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layout-fixture', action='store_true')
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    summary = data.get('descriptive', data)
    rows = {row['condition']: row for row in summary['conditions']}
    assert set(rows) == set(ORDER)
    assert all(row['coding_total'] == 32 and row['retention_total'] == 8 for row in rows.values())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.hashsalt': 'n26-complete-fixed-grid'})
    fig = plt.figure(figsize=(15, 11), layout='constrained')
    grid = fig.add_gridspec(2, 3, width_ratios=[1.35, .9, 1.3], height_ratios=[1, 1.1])
    a, b, c = [fig.add_subplot(grid[0, index]) for index in range(3)]
    d = fig.add_subplot(grid[1, :])
    x = np.arange(8)
    a.bar(x - .18, [rows[k]['coding_correct'] for k in ORDER], .35, color='#246e7c', label='Coding /32')
    a.bar(x + .18, [rows[k]['retention_correct'] for k in ORDER], .35, color='#c98a43', label='Retention /8')
    for index, name in enumerate(ORDER):
        for offset, key in [(-.18, 'coding_correct'), (.18, 'retention_correct')]:
            value = rows[name][key]
            a.text(index + offset, value + .35, str(value), ha='center', fontsize=8)
    a.set(xticks=x, xticklabels=LABELS, ylim=(0, 36), ylabel='Correct cases', title='A  All fixed conditions')
    a.tick_params(axis='x', labelrotation=35)
    a.legend(frameon=False, fontsize=9)
    primary_names = ORDER[2:5]
    losses = [rows[name]['loss_count'] for name in primary_names]
    b.bar(np.arange(3), losses, color=['#246e7c', '#99a9af', '#c0c9cd'])
    for index, value in enumerate(losses):
        b.text(index, value + .3, str(value), ha='center')
    b.set(xticks=np.arange(3), xticklabels=['Selected 0', 'Random A0', 'Random B0'], ylim=(0, 26),
          ylabel='Losses among 23 originally correct', title='B  Primary clean suppression')
    b.tick_params(axis='x', labelrotation=30)
    gain_values, loss_values, contrast_labels = [], [], []
    for factor, key in [(0, 'primary_suppression'), (2, 'secondary_amplification')]:
        contrasts = summary[key]['contrasts']
        for other, label in [('base', 'original'), (f'random34201_64_gain{factor}', 'random A'),
                             (f'random34202_64_gain{factor}', 'random B')]:
            gain_values.append(contrasts[other]['gain_count'])
            loss_values.append(contrasts[other]['loss_count'])
            contrast_labels.append(f'S{factor} vs {label}')
    positions = np.arange(6)
    c.barh(positions, gain_values, color='#2c7960', label='Gains')
    c.barh(positions, [-v for v in loss_values], color='#b66a56', label='Losses')
    for index, (gain, loss) in enumerate(zip(gain_values, loss_values)):
        c.text(gain + .15, index, str(gain), va='center', fontsize=9)
        c.text(-loss - .15, index, str(loss), va='center', ha='right', fontsize=9)
    bound = max(2, max(gain_values + loss_values) + 1)
    c.set(yticks=positions, yticklabels=contrast_labels, xlim=(-bound, bound),
          xlabel='Paired cases out of all 32', title='C  Suppression and secondary gain2')
    c.axvline(0, color='#7f8b91', lw=.8)
    c.axhline(2.5, color='#c2cbcf', ls=':', lw=1)
    c.invert_yaxis()
    c.legend(frameon=False, loc='lower right', fontsize=9)
    families = list(rows['base']['family_correct'])
    assert len(families) == 8
    matrix = np.array([[rows[k]['family_correct'][f] for f in families] for k in ORDER])
    assert ((matrix >= 0) & (matrix <= 4)).all()
    image = d.imshow(matrix, cmap='YlGnBu', vmin=0, vmax=4, aspect='auto')
    for (i, j), value in np.ndenumerate(matrix):
        d.text(j, i, str(value), ha='center', va='center', color='white' if value >= 3 else '#163b43')
    d.set(xticks=np.arange(8), xticklabels=[f.replace('_', '\n') for f in families], yticks=np.arange(8),
          yticklabels=LABELS, title='D  Every family and condition, correct /4')
    fig.colorbar(image, ax=d, ticks=np.arange(5), label='Correct /4', fraction=.02, pad=.02)
    primary = summary['primary_suppression']
    status = f"Primary clean screen: {primary['clean_necessity_screen_passed']}; primary utility screen: {primary['utility_screen_passed']}."
    title = 'LAYOUT FIXTURE ONLY, NOT MODEL RESULTS' if args.layout_fixture else 'N26: fixed global native-column edits, complete grid'
    fig.suptitle(title, fontsize=17, weight='bold')
    fig.supxlabel('320 records: 40 exact original carries (C), 40 guarded identity references (G), 240 new edited responses.\n'
        'Eight dependent development families; no fresh confirmation. Gain2 is secondary and cannot replace suppression.\n'
        + status + ' Returned-list identity is untested. Zero optimizer updates.', fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix('.png'), dpi=170, metadata={'Software': 'Gemma Neuron Engineering'})
    fig.savefig(args.output.with_suffix('.svg'), metadata={'Date': None, 'Creator': 'Gemma Neuron Engineering'})
    plt.close(fig)


if __name__ == '__main__':
    main()
