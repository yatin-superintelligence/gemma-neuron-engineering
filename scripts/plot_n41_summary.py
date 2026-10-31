"""Render the prospectively fixed all-condition N41 panels without scoring."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ORDER = ['original', 'selected_zero', 'randomA_zero', 'randomB_zero', 'prompt_only']
LABELS = ['Original', 'Selected zero', 'Random A zero', 'Random B zero', 'Instruction']
COLORS = ['#17354b', '#087e76', '#80919e', '#a9b7c1', '#ba7a1f']


def render(data, target, *, synthetic=False):
    execution = data['execution']
    if not execution['all_mandatory_guards_pass'] or not execution['all_new_native_EOS']:
        raise ValueError('No ability figure from failed or incomplete engineering')
    if set(data['conditions']) != set(ORDER):
        raise ValueError('All five fixed conditions required')
    conditions = [data['conditions'][name] for name in ORDER]
    families = list(conditions[0]['families'])
    if len(families) != 10 or any(set(c['families']) != set(families) for c in conditions):
        raise ValueError('All eight coding and two retention families required')
    totals = [conditions[0]['families'][name]['cases'] for name in families]
    if sorted(totals) != [8] * 8 + [16] * 2:
        raise ValueError('Exact family denominators required')
    for name, item in zip(ORDER, conditions):
        if sum(item['statuses']['coding'].values()) != 64 or sum(item['statuses']['retention'].values()) != 32:
            raise ValueError('All grading statuses required')
        if data['completion'][name] != {'native_EOS': 96, 'resource_incomplete': 0}:
            raise ValueError('Resource interruptions require separate reporting, not zero performance')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 12, 'axes.titleweight': 'bold',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'text.color': '#17354b', 'axes.labelcolor': '#263e50'})
    fig, axes = plt.subplots(3, 2, figsize=(14, 14.7))
    fig.subplots_adjust(left=.08, right=.97, bottom=.19, top=.88, hspace=.65, wspace=.36)
    primary = 'passed' if data['primary_passed'] else 'failed'
    prefix = 'SYNTHETIC LAYOUT CHECK: ' if synthetic else ''
    fig.suptitle(prefix + 'Digital Surgery: fresh fixed-edit comparison',
                 x=.08, ha='left', y=.97, fontsize=18, weight='bold')
    fig.text(.08, .935, f'N41 fixed accuracy primary {primary}. All cases, controls and losses retained.', fontsize=12)
    fig.text(.08, .91, 'Native EOS without an arbitrary response-token ceiling. Zero training. Related task families are dependent.',
             fontsize=10, color='#526978')
    pos = np.arange(5)
    for ax, role, title, total in [
            (axes[0, 0], 'coding', 'A  Complete-function correctness', 64),
            (axes[0, 1], 'retention', 'B  Retention correctness', 32)]:
        counts = [c[role + '_correct'] for c in conditions]
        ax.bar(pos, counts, color=COLORS, width=.65)
        for i, value in enumerate(counts):
            ax.text(i, value + total * .025, f'{value}/{total}', ha='center')
        ax.set_ylim(0, total * 1.17)
        ax.set_xticks(pos, LABELS, rotation=20, ha='right')
        ax.set_title(title, loc='left', pad=15)
        ax.set_ylabel('Correct cases')
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    ax = axes[1, 0]
    gains = [c['coding_gains'] for c in conditions]
    losses = [-c['coding_losses'] for c in conditions]
    ax.bar(pos, gains, color=COLORS, width=.65)
    ax.bar(pos, losses, color=COLORS, hatch='//', width=.65)
    for i, (gain, loss) in enumerate(zip(gains, losses)):
        ax.text(i, gain + .25, f'+{gain}', ha='center')
        ax.text(i, loss - .25, str(loss), ha='center', va='top')
    bound = max([2, *gains, *[-v for v in losses]]) + 1.5
    ax.set_ylim(-bound, bound)
    ax.axhline(0, color='#74828a', linewidth=.8)
    ax.set_xticks(pos, LABELS, rotation=20, ha='right')
    ax.set_title('C  Paired coding gains and losses', loc='left', pad=15)
    ax.set_ylabel('Programs relative to original')
    ax.grid(axis='y', alpha=.15)
    ax = axes[1, 1]
    matrix = np.array([[c['families'][name]['correct'] for c in conditions] for name in families])
    fractions = matrix / np.array(totals)[:, None]
    ax.imshow(fractions, cmap='YlGnBu', vmin=0, vmax=1, aspect='auto')
    for i, total in enumerate(totals):
        for j in range(5):
            ax.text(j, i, f'{matrix[i,j]}/{total}', ha='center', va='center', fontsize=8,
                    color='white' if fractions[i,j] >= .65 else '#17354b')
    ax.set_yticks(range(10), [name.replace('_', ' ') for name in families], fontsize=8)
    ax.set_xticks(pos, ['Original', 'Selected', 'Random A', 'Random B', 'Instruction'], rotation=20, ha='right', fontsize=8)
    ax.set_title('D  Every dependent family', loc='left', pad=15)
    ax = axes[2, 0]
    statuses = sorted({s for c in conditions for role in ['coding', 'retention'] for s in c['statuses'][role]})
    bottom = np.zeros(5)
    palette = plt.get_cmap('tab20')
    for j, status in enumerate(statuses):
        counts = np.array([sum(c['statuses'][role].get(status, 0) for role in ['coding', 'retention']) for c in conditions])
        ax.bar(pos, counts, bottom=bottom, color=palette(j), label=status.replace('_', ' '))
        bottom += counts
    ax.set_ylim(0, 100)
    ax.set_title('E  Every saved grading status', loc='left', pad=15)
    ax.set_ylabel('Comparison records /96')
    ax.set_xticks(pos, LABELS, rotation=20, ha='right')
    handles, status_labels = ax.get_legend_handles_labels()
    fig.legend(handles, status_labels, fontsize=8, loc='upper left',
               bbox_to_anchor=(.08, .112), ncol=3, frameon=False)
    ax = axes[2, 1]
    ax.axis('off')
    rows = [[label, '96', '0'] for label in LABELS]
    table = ax.table(cellText=rows, colLabels=['Condition', 'Native EOS', 'Interrupted'],
                     loc='upper center', cellLoc='center', bbox=[0, .36, 1, .5])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor('#d4e0e5')
        if row == 0:
            cell.set_facecolor('#edf1f3')
    ax.set_title('F  Completion of comparison records', loc='left', pad=15)
    ax.text(0, .08, '96 original carries + 32 exact retention reuses\n+ 352 new replies = 480 comparison records.\n448 unique generations across A and B.',
            transform=ax.transAxes, fontsize=9, va='bottom')
    clauses = data['clauses']
    short = [('coding preserved', 'all_originally_correct_coding_preserved'),
             ('retention preserved', 'all_originally_correct_retention_preserved'),
             ('four net gains', 'at_least4additional_correct_coding'),
             ('two gain families', 'gains_span_at_least2families'),
             ('above random A', 'at_least2more_than_randomA'),
             ('above random B', 'at_least2more_than_randomB')]
    for line, items in enumerate([short[:3], short[3:]]):
        fig.text(.08, .065 - .018 * line,
                 ' | '.join(label + ': ' + ('PASS' if clauses[key] else 'FAIL') for label, key in items), fontsize=9)
    fig.text(.08, .021, 'Saved audited grades only. Bindings and families are dependent; no p-value, neuron meaning or hardware speed claim.', fontsize=9)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150, facecolor='white')
    plt.close(fig)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Do not regenerate a completed figure without new evidence or a defect')
    digest = render(json.loads(args.data.read_text()), args.output)
    print(json.dumps({'sha256': digest, 'bytes': args.output.stat().st_size, 'scorer_calls': 0}))


if __name__ == '__main__':
    main()
