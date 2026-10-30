"""Render every fixed N24 group and operation from audited aggregates."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

GROUPS = ['selected64', 'random34201_64', 'random34202_64', 'selected16', 'random34201_16', 'random34202_16']
LABELS = ['Selected 64', 'Random A 64', 'Random B 64', 'Selected 16', 'Random A 16', 'Random B 16']
OPS = ['opposite', 'samepolicy', 'zero', 'conditional_cut']
COLORS = ['#00796b', '#4169a1', '#9a5e22', '#49a99d', '#8eacd2', '#cca36f']


def range_panel(axis, values, title, color):
    y = np.arange(len(values))
    means = np.array([row['mean'] for row in values])
    low = np.array([row['minimum'] for row in values])
    high = np.array([row['maximum'] for row in values])
    axis.errorbar(means, y, xerr=np.stack([means - low, high - means]), fmt='none', ecolor=color, capsize=4, linewidth=1.3)
    axis.scatter(means, y, c=COLORS, s=48, zorder=3)
    axis.axvline(0, color='#737b83', linewidth=0.8)
    axis.set_yticks(y, LABELS)
    axis.invert_yaxis()
    axis.set_title(title, loc='left', fontweight='bold')
    axis.set_xlabel('Logit-margin change')
    axis.grid(axis='x', alpha=0.17)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=root / 'runs/n24/version1/descriptive_results.json')
    parser.add_argument('--output', type=Path, default=root / 'research/figures/n24_summary')
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    data = data.get('descriptive', data)
    assert data['prior_program_null_unchanged']
    groups = data['groups']
    assert list(groups) == GROUPS
    matrix = np.array([[groups[group][op]['all13']['mean'] for op in OPS] for group in GROUPS])
    assert np.isfinite(matrix).all()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11.5), layout='constrained')
    a, b, c, d = axes.flat
    span = float(np.abs(matrix).max())
    visual_span = span if span else 1
    heat = a.imshow(matrix, cmap='RdBu_r', vmin=-visual_span, vmax=visual_span, aspect='auto')
    a.set_xticks(range(4), ['Opposite', 'Same policy', 'Zero', 'Conditional cut'], rotation=20, ha='right')
    a.set_yticks(range(6), LABELS)
    a.set_title('A  Every group and operation', loc='left', fontweight='bold')
    for i in range(6):
        for j in range(4):
            a.text(j, i, format(matrix[i, j], '.3g'), ha='center', va='center',
                   color='white' if abs(matrix[i, j]) > visual_span * .55 else '#172634')
    fig.colorbar(heat, ax=a, label='Mean margin shift from native', fraction=.045)
    range_panel(b, [groups[g]['opposite']['all13'] for g in GROUPS],
                'B  Opposite donor: mean and case range', '#5f7484')
    markers = ['o', 's', '^', 'D']
    all_values = []
    for index, group in enumerate(GROUPS):
        rows = data['gradient_finite_calibration'][group]
        assert [row['stratum'] for row in rows] == list(range(4))
        for row in rows:
            x, y = row['archived_first_order_sum'], row['actual_finite_mean']
            c.scatter(x, y, color=COLORS[index], marker=markers[row['stratum']], s=44,
                      label=LABELS[index] if row['stratum'] == 0 else None)
            all_values.extend([x, y])
    lo, hi = min(all_values), max(all_values)
    pad = (hi - lo) * .08 if hi != lo else .05
    c.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle=':', color='#697681', linewidth=1)
    c.set_xlim(lo - pad, hi + pad)
    c.set_ylim(lo - pad, hi + pad)
    c.set_xlabel('Archived first-order sum')
    c.set_ylabel('Measured finite margin shift')
    c.set_title('C  All four discovery strata per group', loc='left', fontweight='bold')
    c.legend(ncol=2, fontsize=8, loc='best')
    c.grid(alpha=.17)
    range_panel(d, [groups[g]['conditional_cut']['conditional_margin_loss_all13'] for g in GROUPS],
                'D  Conditional cut: loss from full-opposite margin', '#5f7484')
    scale = data.get('full_state_control')
    if scale is None:
        values = [r['full_opposite_scale'] for r in data['private_case_rows'] if r['group'] == 'selected64' and r['operation'] == 'opposite']
        scale = {'mean': float(np.mean(values)), 'minimum': min(values), 'maximum': max(values)}
    fig.suptitle('N24: finite effects of fixed native channel groups', fontsize=17, fontweight='bold')
    fig.supxlabel('13 reused development decisions. Ranges are case minima/maxima, not confidence intervals. Nested 16 is descriptive.\n'
                  'Panel C uses the same eight discovery cases; the dotted line is equality, with no fitted correction.\n'
                  'Markers circle/square/triangle/diamond identify discovery strata 0/1/2/3.\n'
                  f"Full-state opposite scale: mean {scale['mean']:.3f}, range {scale['minimum']:.3f} to {scale['maximum']:.3f}. This is not a mediation estimate.\n"
                  'Zero new completions or training. Prior program-level nulls remain unchanged; no semantic role or coding gain is established.',
                  fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix('.png'), dpi=170)
    fig.savefig(args.output.with_suffix('.svg'))
    plt.close(fig)


if __name__ == '__main__':
    main()
