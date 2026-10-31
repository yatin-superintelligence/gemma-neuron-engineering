"""Render the complete original-only N32 baseline stop from reviewed aggregates."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=root / 'runs/n32/phase_A/version2/public_aggregate.json')
    parser.add_argument('--output', type=Path, default=root / 'research/figures/n32_summary')
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    assert data['status'] == 'closed_planned_original_only_baseline_stop'
    assert data['interventions'] == 0 and len(data['all8blocks']) == 8
    plt.rcParams.update({'font.size': 11, 'svg.hashsalt': 'n32-complete-baseline-stop'})
    fig = plt.figure(figsize=(14, 11), layout='constrained')
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1.25, .8])
    colors = {'passed': '#087f8c', 'wrong_answer': '#cf8665', 'source_contract_error': '#795c92'}
    labels = {'passed': 'Correct under frozen contract', 'wrong_answer': 'Executable wrong answer',
              'source_contract_error': 'Source-contract failure'}
    axis = fig.add_subplot(grid[0, 0])
    bottom = np.zeros(3)
    families = data['family_outcomes']
    for key, color in colors.items():
        counts = [row['statuses'].get(key, 0) for row in families]
        axis.bar(range(3), counts, bottom=bottom, color=color, label=labels[key])
        for i, value in enumerate(counts):
            if value:
                axis.text(i, bottom[i] + value / 2, str(value), ha='center', va='center', color='white', weight='bold')
        bottom += counts
    axis.set_xticks(range(3), ['Predicate /16', 'Filter /16', 'Retention /8'])
    axis.set_ylabel('Original programs')
    axis.set_ylim(0, 18)
    axis.set_title('All 40 original grades', loc='left', weight='bold')
    axis.legend(frameon=False, fontsize=9, loc='upper center', bbox_to_anchor=(.5, -.12))
    axis = fig.add_subplot(grid[0, 1])
    axis.axis('off')
    gate = data['baseline_gate']
    axis.text(0, .96, 'Predefined gate: STOP', va='top', weight='bold', fontsize=15)
    axis.text(0, .78,
        f"Role {gate['role_correct']}/32; required 32/32\n"
        f"Retention {gate['retention_correct']}/8; required 8/8\n"
        f"Common valid prefixes {data['valid_prefix_pairs']}/16; required 16/16\n"
        f"Complete blocks {data['complete_blocks']}/8\n"
        'No capped output; no intervention run\n'
        'B and C were never built or launched', va='top', linespacing=1.8)
    axis = fig.add_subplot(grid[1, :])
    blocks = data['all8blocks']
    left = np.zeros(8)
    for key, color in colors.items():
        values = [row['statuses'].get(key, 0) for row in blocks]
        axis.barh(range(8), values, left=left, color=color)
        for i, value in enumerate(values):
            if value:
                axis.text(left[i] + value / 2, i, str(value), ha='center', va='center', color='white', weight='bold')
        left += values
    axis.set_yticks(range(8), [f"{row['family'].title()}, wording {row['style']}, name variant {row['name_index']}" for row in blocks])
    axis.invert_yaxis()
    axis.set_xticks(range(5))
    axis.set_xlim(0, 4)
    axis.set_xlabel('All four relation/order cases in each dependent block')
    axis.set_title('Every declared block remains reported; none is selected for further testing', loc='left', weight='bold')
    axis = fig.add_subplot(grid[2, :])
    axis.axis('off')
    receipt = data['phase_A']
    axis.text(0, .96,
        '10 source failures: 6 declared-list/loop-name failures and 4 requested-operand-order failures.\n'
        'The other 3 role errors are executable wrong answers. All original grades remain unchanged.\n'
        '40 new complete-function records and 40 trajectories; 1065 forwards, 5104 input-token exposures and 1065 generated tokens.\n'
        f"Worker {receipt['worker_seconds']:.3f} s; notebook {receipt['notebook_seconds']:.3f} s. Durations do not measure charged quota.\n"
        'All token, cache, parameter, hook, source and counter guards pass. Zero backwards and optimizer updates.\n'
        'Version 1 is a separate preserved metadata-schema startup failure with zero model forwards or predictions.\n'
        'This assay does not test a group effect, semantic-vs-token pattern, neuron meaning or coding gain.',
        va='top', linespacing=1.7, fontsize=10.3)
    fig.suptitle('N32 | Complete-function baseline fails before any neuron intervention', weight='bold', fontsize=17)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ['png', 'svg']:
        fig.savefig(args.output.with_suffix('.' + suffix), dpi=140,
                    metadata={'Date': None} if suffix == 'svg' else {})
    plt.close(fig)


if __name__ == '__main__':
    main()
