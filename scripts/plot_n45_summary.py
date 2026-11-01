"""Fixed all-condition N45 report, with the code-fence context explicit."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from evaluation.signed_mask_reporting import CONDITIONS, validate


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(a, plan, destination, *, synthetic=False):
    destination = Path(destination)
    if destination.exists(): raise ValueError('Never overwrite a reporting artifact')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    destination.mkdir(parents=True)
    labels = ['Original', 'Neg zero', 'Rand neg', 'Pos zero', 'Rand pos']
    fig, axs = plt.subplots(2, 3, figsize=(18, 11))
    panels = [('argmax_changed', 'Case mean: native choice changed', True),
              ('delta_L2', 'Case mean: full-logit delta L2', False),
              ('native_winner_probability', 'Case mean: native winner probability', True),
              ('native_margin', 'Case mean: native winner vs fixed other', False),
              ('removed_h_energy_fraction', 'Case mean: removed native h energy', True)]
    for ax, (key, title, fraction) in zip(list(axs.flat)[:5], panels):
        x = np.arange(5); observed = []
        for offset, role, color in [(-.18, 'coding', '#2563eb'), (.18, 'retention', '#d97706')]:
            group = a['strata']['role:'+role]
            records = [group['conditions'][c][key] for c in CONDITIONS]
            values = [r['case_mean'] for r in records]
            observed += [v for v in values if v is not None]
            bars = ax.bar(x+offset, [0 if v is None else v for v in values], width=.35, color=color, label=role.title())
            for i, (bar, value, record) in enumerate(zip(bars, values, records)):
                if value is None:
                    bar.set_hatch('//'); text = 'undefined'
                else: text = f'{value:.0f}' if key == 'delta_L2' else f'{100*value:.3g}%' if key == 'argmax_changed' else f'{value:.3g}'
                if value is None: text = 'n/a'
                if value is not None and fraction and value > .9:
                    y = value-.035; va = 'top'
                else:
                    y = value or 0; va = 'bottom'
                ax.annotate(text, (i+offset, y), xytext=(0, 4 if va == 'bottom' else -2), textcoords='offset points', ha='center', va=va, fontsize=8)
        if fraction: ax.set_ylim(0, 1)
        else:
            lower = min([0, *observed]); upper = max([0, *observed])
            if lower == upper: ax.set_ylim(0, 1); ax.text(.5, .5, 'All values zero', transform=ax.transAxes, ha='center')
            else: ax.set_ylim(lower*1.15 if lower < 0 else 0, upper*1.16 if upper > 0 else 0)
        ax.set_xticks(x, labels, rotation=20, ha='right', fontsize=9)
        if key == 'removed_h_energy_fraction':
            counts = []
            for label, condition in zip(labels, CONDITIONS):
                code = a['strata']['role:coding']['conditions'][condition][key]['defined_cases']
                retain = a['strata']['role:retention']['conditions'][condition][key]['defined_cases']
                counts.append(label+'\nC'+str(code)+'/64 R'+str(retain)+'/32')
            ax.set_xticks(x, counts, rotation=20, ha='right', fontsize=7)
        ax.set_title(title, fontsize=12); ax.axhline(0, color='#475569', linewidth=.6)
        ax.spines[['top', 'right']].set_visible(False)
    ax = axs[1, 2]
    families = [name for name in plan['strata'] if name.startswith('family:')]
    matrix = np.array([[a['strata'][name]['conditions'][c]['delta_L2']['case_mean'] for c in CONDITIONS] for name in families])
    maximum = float(matrix.max())
    if maximum > 0:
        image = ax.imshow(matrix, vmin=0, vmax=maximum, aspect='auto', cmap='Blues')
        fig.colorbar(image, ax=ax, fraction=.035, pad=.03, label='Mean full-logit delta L2')
    else:
        ax.set_xlim(-.5, 4.5); ax.set_ylim(len(families)-.5, -.5); ax.set_facecolor('white')
        ax.text(.5, -.23, 'All matrix values zero; no color range', transform=ax.transAxes, ha='center', fontsize=8)
    ax.set_xticks(range(5), labels, rotation=20, ha='right', fontsize=9)
    ax.set_yticks(range(len(families)), [name.split(':')[1].replace('_', ' ') for name in families], fontsize=9)
    for i in range(len(families)):
        for j in range(5): ax.text(j, i, f'{matrix[i,j]:.0f}', ha='center', va='center', fontsize=8, color='white' if maximum and matrix[i,j] > .55*maximum else '#172554')
    ax.set_title('Every family and condition: full-logit delta L2', fontsize=12)
    prefix = 'SYNTHETIC SOFTWARE FIXTURE ONLY | ' if synthetic else ''
    fig.suptitle(prefix+'N45: signed final-MLP contributions at the native prefix', fontsize=18, y=.98)
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .945), ncol=2)
    footer = ('Synthetic counts are not model outcomes.' if synthetic else 'All 192 coding-task sample positions are the opening Python code fence, not function-body decisions.')
    fig.text(.5, .035, footer+'\nAll 96 cases and five conditions retained. Random controls match cardinality, not energy. No new pretrained tokens or optimizer updates.', ha='center', fontsize=11)
    fig.subplots_adjust(left=.06, right=.95, bottom=.15, top=.87, wspace=.42, hspace=.43)
    path = destination/('synthetic_n45_summary.png' if synthetic else 'n45_summary.png')
    fig.savefig(path, dpi=180, facecolor='white'); plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser()
    for name in ['aggregate', 'audit-receipt', 'reporting-plan', 'source-freeze', 'output-directory']: p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args(); freeze = json.loads(args.source_freeze.read_text())
    for name, h in freeze['sources'].items():
        if sha(name) != h: raise ValueError('Reporting source changed: '+name)
    for key, path in [('aggregate', args.aggregate), ('audit_receipt', args.audit_receipt), ('reporting_plan', args.reporting_plan)]:
        if sha(path) != freeze['input_sha256'][key]: raise ValueError('Frozen report input changed')
    receipt = json.loads(args.audit_receipt.read_text()); plan = json.loads(args.reporting_plan.read_text())
    if receipt.get('fixture'): raise ValueError('Synthetic fixture cannot be presented as actual evidence')
    a = validate(args.aggregate.read_bytes(), receipt, plan)
    figure = render(a, plan, args.output_directory)
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'actual_complete_audited_figure_rendered_once',
        'input_sha256': freeze['input_sha256'], 'sources': freeze['sources'], 'figure_sha256': sha(figure),
        'all_conditions_roles_families_retained': True, 'model_tokenizer_executor_scorer_calls': 0, 'visual_QA': 'pending'}
    (args.output_directory/'figure_receipt.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__': main()
