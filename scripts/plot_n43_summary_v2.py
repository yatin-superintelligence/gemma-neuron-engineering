"""Plot every fixed normalization control from a complete saved auditor aggregate."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from evaluation.normalization_figure_contract import GROUPS, CELLS, SITES, FAMILY_CASES, validate

LABELS = ['Selected', 'Random A', 'Random B']
COLORS = ['#087e76', '#657f95', '#b77836']
CELL_LABELS = ['00 native/native', '10 removed/native', '01 native/removed', '11 removed/removed']


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.spines.top': False,
        'axes.spines.right': False, 'text.color': '#17354b', 'axes.labelcolor': '#263e50'})


def heading(fig, text, synthetic):
    prefix = 'SYNTHETIC LAYOUT CHECK: ' if synthetic else ''
    fig.suptitle(prefix + text, x=.07, ha='left', y=.975, fontsize=16, weight='bold')
    fig.text(.07, .945, 'N43 layer-local native-state controls. All groups and fixed cells retained. Zero training.', fontsize=10)
    fig.text(.07, .925, 'Saved audited equal-case means; dependent positions and families. No new replies or coding-utility test.', fontsize=9)


def save(fig, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150, facecolor='white')
    plt.close(fig)
    return {'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'bytes': target.stat().st_size}


def overview(data, target, synthetic):
    fig, axes = plt.subplots(3, 2, figsize=(14, 12.8))
    fig.subplots_adjust(left=.08, right=.97, bottom=.16, top=.87, hspace=.75, wspace=.30)
    heading(fig, 'Digital Surgery: normalization-factor contrasts', synthetic)
    panels = [('logit_delta_L2', 'A  Full-vocabulary logit shift', 'L2 from native00', 1),
        ('choice_change_rate', 'B  Native next-choice changes', 'Equal-case rate (%)', 100),
        ('native_margin_change', 'C  Native winner versus fixed other token', 'Logit-margin change', 1),
        ('native_winner_probability_change', 'D  Native winner probability', 'Probability change (percentage points)', 100),
        ('logit_delta_max_abs', 'E  Largest vocabulary-coordinate shift', 'Maximum absolute logit change', 1)]
    positions = np.arange(4)
    for ax, (metric, title, ylabel, scale) in zip(axes.flat, panels):
        for i, group in enumerate(GROUPS):
            values = [data['all_groups'][group]['cells'][cell][metric]['case_mean']*scale for cell in CELLS]
            ax.bar(positions+(i-1)*.24, values, width=.23, color=COLORS[i], label=LABELS[i])
        ax.set_xticks(positions, CELL_LABELS, rotation=18, ha='right', fontsize=8)
        ax.set_title(title, loc='left', pad=12)
        ax.set_ylabel(ylabel)
        ax.axhline(0, color='#70838d', linewidth=.8)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
        low, high = ax.get_ylim()
        ax.set_ylim(min(0, low), max(0, high))
    ax = axes[2, 1]
    ax.axis('off')
    ax.set_title('F  Interaction and defined path comparisons', loc='left', pad=12)
    rows = []
    for label, group in zip(LABELS, GROUPS):
        item = data['all_groups'][group]
        cosine = item['raw_normalization_cosine']
        cosine_text = 'undefined' if cosine is None else f"{cosine['case_mean']:.3g}\n({cosine['cases_with_defined_values']}/96 cases)"
        rows.append([label, f"{item['interaction_L2']['case_mean']:.3g}",
            f"{item['interaction_margin']['case_mean']:.3g}", cosine_text])
    table = ax.table(cellText=rows, colLabels=['Group', 'Interaction L2', 'Margin interaction', 'Path cosine'],
        cellLoc='center', bbox=[0, .37, 1, .55], colWidths=[.19, .24, .29, .28])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor('#d4e0e5')
        if row == 0:
            cell.set_facecolor('#edf1f3')
    zero_lines = ['Zero raw / normalization paths (group-site-decision records):']
    for label, group in zip(LABELS, GROUPS):
        item = data['all_groups'][group]
        zero_lines.append(f"{label}: {item['zero_raw_path_records']:,} / {item['zero_normalization_path_records']:,} of 2,529")
    ax.text(0, .08, '\n'.join(zero_lines), transform=ax.transAxes, va='bottom', fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(.07, .066), ncol=3, frameon=False)
    fig.text(.08, .045, '96 cases | 281 sampled decisions | 7,587 group/site rows | 61,645 forwards | all exact guards pass', fontsize=9)
    fig.text(.08, .024, '00: native/native. 10: removed raw/native RMS. 01: native raw/removed RMS. 11: removed/removed.', fontsize=9)
    fig.text(.08, .008, 'Finite contrasts are not additive mediation, individual neuron meaning or improvement over the original model.', fontsize=8)
    return save(fig, target)


def strata(data, target, synthetic):
    fig, axes = plt.subplots(3, 2, figsize=(14, 12.8))
    fig.subplots_adjust(left=.09, right=.91, bottom=.19, top=.86, hspace=.76, wspace=.28)
    heading(fig, 'Digital Surgery: all fixed normalization strata', synthetic)
    families = list(FAMILY_CASES)
    matrices = {}
    for cell in CELLS[1:]:
        for scope, keys in [('all_group_sites', list(map(str, SITES))), ('all_group_families', families)]:
            matrices[cell, scope] = np.array([[data[scope][group][key]['cells'][cell]['logit_delta_L2']['case_mean']
                for key in keys] for group in GROUPS])
    largest = max(float(m.max()) for m in matrices.values())
    image = None
    for i, cell in enumerate(CELLS[1:]):
        for j, scope in enumerate(['all_group_sites', 'all_group_families']):
            ax = axes[i, j]
            values = matrices[cell, scope]
            if largest:
                image = ax.imshow(values, cmap='YlGnBu', vmin=0, vmax=largest, aspect='auto')
            else:
                ax.imshow(np.zeros(values.shape), cmap=matplotlib.colors.ListedColormap(['#ffffff']), aspect='auto')
            for row in range(3):
                for col in range(values.shape[1]):
                    ax.text(col, row, f'{values[row,col]:.2g}', ha='center', va='center', fontsize=7,
                        color='white' if largest and values[row,col]>.65*largest else '#17354b')
            labels = [str(s) for s in SITES] if j == 0 else [f"{name.replace('_', ' ')} ({FAMILY_CASES[name]})" for name in families]
            ax.set_xticks(range(len(labels)), labels, rotation=0 if j == 0 else 37,
                ha='center' if j == 0 else 'right', fontsize=7)
            ax.set_yticks(range(3), LABELS, fontsize=8)
            ax.set_title(f"{chr(65+2*i+j)}  {CELL_LABELS[CELLS.index(cell)]}: {'every site' if j==0 else 'every family'}", loc='left', pad=10)
            ax.set_xlabel('Native layer (96 cases each)' if j == 0 else 'Family (case denominator)', fontsize=8)
    if image is not None:
        bar = fig.colorbar(image, cax=fig.add_axes([.94, .25, .012, .53]))
        bar.set_label('Equal-case full-vocabulary logit-delta L2', fontsize=9)
    else:
        fig.text(.09, .89, 'Every plotted L2 mean is exactly zero; no color range is assigned.', fontsize=9)
    fig.text(.09, .096, 'Shared linear scale across all six matrices. 00 exact native control is retained in the overview and complete JSON.', fontsize=9)
    fig.text(.09, .072, 'All other metrics and both coding/retention roles remain in the audited JSON. No site or family selected by effect.', fontsize=9)
    fig.text(.09, .048, 'Arithmetic: 16 cases/48 sampled decisions. JSON: 16/41. Each coding family: 8/24. Families and bindings are dependent.', fontsize=9)
    fig.text(.09, .024, 'Layer-local finite contrasts do not establish utility, a global explanation of N41 regressions or individual meaning.', fontsize=9)
    return save(fig, target)


def render(data, directory, *, synthetic=False):
    validate(data)
    targets = [directory/'n43_normalization_overview.png', directory/'n43_normalization_strata.png']
    if any(path.exists() for path in targets):
        raise FileExistsError('Do not rerender completed figures without new evidence or a recorded defect')
    style()
    return {str(targets[0].name): overview(data, targets[0], synthetic),
        str(targets[1].name): strata(data, targets[1], synthetic)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aggregate', type=Path, required=True)
    parser.add_argument('--audit-receipt', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.aggregate.read_text())
    if data['execution'] != json.loads(args.audit_receipt.read_text()):
        raise ValueError('Passing original audit receipt must match the aggregate execution exactly')
    result = render(data, args.output_directory)
    result['aggregate_sha256'] = hashlib.sha256(args.aggregate.read_bytes()).hexdigest()
    result['plot_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['model_or_scorer_calls'] = 0
    print(json.dumps(result))


if __name__ == '__main__':
    main()
