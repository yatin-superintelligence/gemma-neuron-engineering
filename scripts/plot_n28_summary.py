"""Plot every audited N28 donor-order cell without exposing task text."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


CONDITIONS = ['base', 'identity', 'opposite_same_order', 'same_same_order',
              'opposite_reversed_order', 'same_reversed_order', 'rescue']
LABELS = ['Original\ncarry', 'Identity', 'Opposite\nsame order', 'Same\nsame order',
          'Opposite\nreverse order', 'Same\nreverse order', 'Rescue']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path,
                        default=Path('runs/n28/phase_B/version1/public_aggregate.json'))
    parser.add_argument('--output', type=Path, default=Path('research/figures/n28_summary'))
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    assert data['all_records_and_mandatory_guards_pass'] and data['records'] == 168
    summary = data['descriptive_results']
    by = {r['condition']: r for r in summary['conditions']}
    assert set(by) == set(CONDITIONS)
    keys = [(f, s, d, o) for f in ['filter', 'predicate'] for s in [0, 1]
            for d in [1, -1] for o in [0, 1]]
    cells = {}
    for condition in CONDITIONS:
        records = by[condition]['all_role_cells']
        indexed = {(r['family'], r['style'], r['direction'], r['order']): r for r in records}
        assert set(indexed) == set(keys) and len(records) == 16
        assert all(not (r['correct'] and r['opposite_correct']) for r in records)
        cells[condition] = indexed
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'svg.fonttype': 'none', 'axes.titleweight': 'bold'})
    fig = plt.figure(figsize=(16, 13), facecolor='white')
    grid = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], height_ratios=[1.15, 1],
                           hspace=.42, wspace=.60)
    fig.subplots_adjust(top=.86, bottom=.18, left=.11, right=.97)
    ax = fig.add_subplot(grid[0, 0])
    matrix = np.array([[2 if cells[c][key]['correct'] else
                        1 if cells[c][key]['opposite_correct'] else 0
                        for c in CONDITIONS] for key in keys])
    ax.imshow(matrix, cmap=ListedColormap(['#d7dce0', '#c27b4b', '#438b7f']),
              vmin=0, vmax=2, aspect='auto')
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, ['x', 'A', 'O'][value], ha='center', va='center',
                color='white' if value else '#263746', weight='bold')
    ax.set_xticks(range(7), LABELS, fontsize=8)
    ax.set_yticks(range(16), [f'{f[:4]} W{s} {"higher" if d == 1 else "lower"} R{o}'
                             for f, s, d, o in keys], fontsize=8)
    ax.axhline(7.5, color='white', lw=2)
    ax.set_title('A. Every role cell and condition', loc='left', pad=12)
    ax.set_xlabel('O: original request correct   A: opposite request correct   x: neither', labelpad=10)
    ax = fig.add_subplot(grid[0, 1]); ax.axis('off')
    rows = []
    for c in CONDITIONS:
        item = by[c]
        rows.append([LABELS[CONDITIONS.index(c)].replace('\n', ' '), item['coding_correct'],
                     sum(r['opposite_correct'] for r in item['all_role_cells']),
                     f"{item['coding_gains']} / {item['coding_losses']}",
                     item['retention_correct'], item['at_cap']])
    table = ax.table(cellText=rows,
                     colLabels=['Policy', 'Own\n/16', 'Alt\n/16', 'Gain /\nloss', 'Ret\n/8', 'Cap\n/24'],
                     colWidths=[.42, .11, .11, .16, .1, .1], loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1.1, 2.15)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#d9e0e4')
        if r == 0: cell.set_facecolor('#edf2f4')
    role = summary['role']
    ax.set_title('B. Complete outcomes and fixed primary', loc='left', pad=12)
    ax.text(0, .04, f"Complete pairs: {role['successful_complete_pairs']}/8 (requires 6)\n"
            f"Primary passed: {role['primary_passed']}", transform=ax.transAxes, fontsize=10)
    ax = fig.add_subplot(grid[1, 0])
    pairs = [(f, s, d) for f in ['filter', 'predicate'] for s in [0, 1] for d in [1, -1]]
    required = [('opposite_same_order', 0), ('opposite_same_order', 1),
                ('opposite_reversed_order', 0), ('opposite_reversed_order', 1)]
    checks = np.array([[int(cells[c][(*key, o)]['opposite_correct']) for c, o in required]
                       for key in pairs])
    ax.imshow(checks, cmap=ListedColormap(['#d7dce0', '#438b7f']), vmin=0, vmax=1, aspect='auto')
    for (i, j), value in np.ndenumerate(checks):
        ax.text(j, i, str(value), ha='center', va='center')
    ax.set_yticks(range(8), [f'{f[:4]} W{s} {"higher" if d == 1 else "lower"}' for f, s, d in pairs])
    ax.set_xticks(range(4), ['Same donor\norder R0', 'Same donor\norder R1',
                            'Reverse donor\norder R0', 'Reverse donor\norder R1'])
    ax.set_title('C. All four checks required for each complete pair', loc='left', pad=12)
    ax.set_xlabel('1: alternative program correct. All four must pass; no best-arm score.', labelpad=10)
    ax = fig.add_subplot(grid[1, 1])
    foreign = CONDITIONS[2:6]
    diag = summary['foreign_operator_diagnostics']
    copies = [diag[c]['copies_actual_donor_operator'] for c in foreign]
    valid = [diag[c]['valid_single_comparison'] for c in foreign]
    ax.barh(range(4), copies, color='#557789')
    for i, (copied, n) in enumerate(zip(copies, valid)):
        label = f'{copied}/{n} valid' if n else 'N/A (0 valid)'
        ax.text(copied-.3 if copied else .2, i, label, va='center', fontsize=9,
                ha='right' if copied else 'left', color='white' if copied else '#263746')
    ax.set_yticks(range(4), [LABELS[CONDITIONS.index(c)].replace('\n', ' ') for c in foreign])
    ax.invert_yaxis(); ax.set_xlim(0, 18); ax.set_xticks([0, 4, 8, 12, 16])
    ax.set_xlabel('Copies actual donor operator, among valid source programs')
    ax.set_title('D. Literal-operator diagnostic, all foreign arms', loc='left', pad=12)
    ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('N28: direct relation and donor operand order', x=.08, ha='left', y=.97,
                 fontsize=18, weight='bold')
    fig.text(.08, .93, '168 audited records: 24 exact original carries and 144 new continuations. Zero optimizer updates.', fontsize=10)
    fig.text(.08, .035, 'Two code families with dependent wordings and directions. Conditional completion, not whole-program improvement.\n'
             'Foreign retention donors repeat the same paired paraphrase. Invalid source is not evidence for or against operator copying.\n'
             'Identity/rescue generations, native replays, past-cache paths, scope, counters and restorations pass.', fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix('.png'), dpi=180, bbox_inches='tight', metadata={'Software': 'Matplotlib'})
    fig.savefig(args.output.with_suffix('.svg'), bbox_inches='tight', metadata={'Date': None})
    plt.close(fig)


if __name__ == '__main__':
    main()
