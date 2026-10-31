"""Render all fixed N40 conditions from audited aggregates without scoring."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORDER = ['original', 'selected_zero', 'randomA_zero', 'randomB_zero', 'prompt_only']
LABELS = ['Original', 'Selected zero', 'Random A zero', 'Random B zero', 'Instruction']
COLORS = ['#17354b', '#087e76', '#80919e', '#a9b7c1', '#ba7a1f']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'research/figures/n40_summary.png')
    args = parser.parse_args()
    data_path = args.data or (ROOT / 'results/n40_summary.json' if (ROOT / 'results/n40_summary.json').exists() else ROOT / 'runs/n40/version2/public_aggregate.json')
    data = json.loads(data_path.read_text())
    assert data['all_mandatory_guards_pass'] and not data['assessment']['primary_passed']
    assert data['all_new_responses_native_EOS']
    conditions = [data['baseline']] + [data['conditions'][n] for n in ORDER[1:]]
    pos = np.arange(5)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 12, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.labelcolor': '#263e50',
                         'text.color': '#17354b', 'axes.titleweight': 'bold'})
    fig, axes = plt.subplots(3, 2, figsize=(14, 13.2))
    fig.subplots_adjust(left=.07, right=.97, bottom=.12, top=.88, hspace=.7, wspace=.3)
    fig.suptitle('Digital Surgery: fixed native column deletion', x=.07, ha='left', y=.97, fontsize=20, weight='bold')
    fig.text(.07, .935, 'N40 output-reduction primary failed. Coding repairs are exploratory, within one reused task family.', fontsize=12)
    fig.text(.07, .91, 'All five conditions retained; zero training; no response hit its endpoint ceiling.', fontsize=11, color='#526978')
    for ax, field, title, total in [(axes[0, 0], 'coding', 'A  Complete-function coding', 32),
                                     (axes[0, 1], 'retention', 'B  Retention, weak original baseline', 16)]:
        values = [c[field]['correct'] for c in conditions]
        ax.bar(pos, values, color=COLORS, width=.65)
        for i, value in enumerate(values):
            ax.text(i, value + .35, f'{value}/{total}', ha='center', fontsize=11)
        ax.set_ylim(0, total + 3)
        ax.set_ylabel('Correct cases')
        ax.set_title(title, loc='left', pad=14)
        ax.set_xticks(pos, LABELS, rotation=20, ha='right')
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.15)
    ax = axes[1, 0]
    all_ratio = [1] + [c['coding']['mean_token_ratio'] for c in conditions[1:]]
    correct_ratio = [1] + [c['original_correct_coding']['mean_token_ratio'] for c in conditions[1:]]
    ax.plot(pos, all_ratio, 'o-', color='#087e76', label='All 32 coding cases')
    ax.plot(pos, correct_ratio, 's--', color='#b8791e', label='Original correct 24')
    ax.axhline(.9, color='#a63c39', linewidth=1, linestyle=':', label='Required ratio at most 0.90')
    ax.axhline(1, color='#74828a', linewidth=.7)
    ax.set_ylim(.6, 1.1)
    ax.set_title('C  Mean per-response token ratio', loc='left', pad=14)
    ax.set_ylabel('New tokens / original tokens')
    ax.set_xticks(pos, LABELS, rotation=20, ha='right')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(alpha=.15)
    ax = axes[1, 1]
    for offset, key, label, color, marker in [(-.08, 'coding', 'All 32', '#087e76', 'o'),
                                              (.08, 'original_correct_coding', 'Original correct 24', '#b8791e', 's')]:
        lower = np.array([c[key]['mean_comment_reduction_lower'] for c in conditions[1:]]) * 100
        upper = np.array([c[key]['mean_comment_reduction_upper'] for c in conditions[1:]]) * 100
        midpoint = (lower + upper) / 2
        ax.errorbar(np.arange(1, 5) + offset, midpoint, yerr=[midpoint - lower, upper - midpoint],
                    fmt=marker, capsize=4, color=color, label=label)
    ax.axhline(0, color='#74828a', linewidth=.8)
    ax.axhline(10, color='#087e76', linestyle=':', linewidth=1)
    ax.axhline(5, color='#b8791e', linestyle=':', linewidth=1)
    ax.set_xlim(-.25, 4.4)
    ax.text(0, .4, 'Original\nreference', ha='center', fontsize=8)
    ax.set_title('D  Conservative comment-reduction bounds', loc='left', pad=14)
    ax.set_ylabel('Percentage points; positive means reduction')
    ax.set_xticks(pos, LABELS, rotation=20, ha='right')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=.15)
    ax = axes[2, 0]
    gains = [c['coding']['gains'] for c in conditions]
    losses = [-c['coding']['losses'] for c in conditions]
    ax.bar(pos, gains, color=COLORS, width=.65, label='Gains')
    ax.bar(pos, losses, color=COLORS, width=.65, hatch='//', label='Losses')
    ax.axhline(0, color='#74828a', linewidth=.8)
    ax.set_ylim(-3, 3)
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_title('E  Paired coding gains and losses', loc='left', pad=14)
    ax.set_ylabel('Programs relative to original')
    ax.set_xticks(pos, LABELS, rotation=20, ha='right')
    ax.text(.01, .02, 'Both selected gains: one family.\nOne gain also appears under each random.', transform=ax.transAxes, fontsize=9)
    ax.grid(axis='y', alpha=.15)
    ax = axes[2, 1]
    family_names = [k for k in data['conditions']['selected_zero']['families'] if k.startswith('coding:')]
    table = []
    # Original family counts come from the same audited paired aggregates.
    for family in family_names:
        baseline = data['conditions']['selected_zero']['families'][family]['original_correct']
        table.append([baseline] + [c['families'][family]['correct'] for c in conditions[1:]])
    matrix = np.array(table)
    ax.imshow(matrix, cmap='YlGnBu', vmin=0, vmax=4, aspect='auto')
    for i in range(8):
        for j in range(5):
            ax.text(j, i, f'{matrix[i, j]}/4', ha='center', va='center', fontsize=9,
                    color='white' if matrix[i, j] >= 3 else '#17354b')
    ax.set_yticks(np.arange(8), [n.split(':')[1].replace('_', ' ') for n in family_names], fontsize=8)
    ax.set_xticks(pos, ['Original', 'Selected', 'Random A', 'Random B', 'Instruction'], rotation=20, ha='right', fontsize=9)
    ax.set_title('F  Eight dependent coding families', loc='left', pad=14)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(.07, .055, '240 records: 192 new native-EOS replies and 48 exact original carries. All mandatory source, identity, cache, scope and restoration guards pass.', fontsize=10)
    fig.text(.07, .033, 'Mean case ratios are not pooled token ratios or hardware speed. Unknown comment spans remain in conservative bounds. No fresh-confirmation claim.', fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, facecolor='white')
    plt.close(fig)
    print(json.dumps({'png': str(args.output), 'bytes': args.output.stat().st_size,
                      'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
