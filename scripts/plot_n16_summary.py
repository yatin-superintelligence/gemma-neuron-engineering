"""Show all thirty-one native layer-subset controls, including closed parents."""
from pathlib import Path
import datetime, hashlib, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'runs/n16/version1/descriptive_results.json'; private = source.exists()
    if not private: source = ROOT / 'results/n16_layer_subset_summary.json'
    data = json.loads(source.read_text()); gates = data['estimands']['complete_31_subset_grid']
    assert len(gates) == 31
    gates = sorted(gates, key=lambda g: (len(g['layers']), tuple(g['layers'])))
    labels = [', '.join(map(str, g['layers'])) for g in gates]
    selected = [5, 6, 9]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 13), sharey=True,
        gridspec_kw={'left': .20, 'right': .93, 'top': .84, 'bottom': .17, 'wspace': .17})
    for ax, key, title, threshold in zip(axes, ['opposite_alternative_correct', 'same_rule_preserved'],
        ['A. Opposite-rule transfer / 16', 'B. Same-rule preservation / 16'], [8, 14]):
        colors = ['#b85b21' if g['layers'] == selected else '#2d6573' if g['eligible'] else '#b9c7cc' for g in gates]
        values = [g[key] for g in gates]
        ax.barh(range(31), values, color=colors, height=.72)
        ax.axvline(threshold, color='#982e29', ls=':', lw=1.4)
        ax.set_xlim(0, 17.5); ax.set_xticks([0, 4, 8, 12, 16]); ax.set_title(title, loc='left', weight='bold', pad=16)
        ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
        for i, value in enumerate(values): ax.text(value + .18, i, str(value), va='center', fontsize=9)
        for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
    axes[0].set_yticks(range(31), labels); axes[0].invert_yaxis(); axes[0].set_ylabel('Native layer subset', labelpad=16)
    for i in [4.5, 14.5, 24.5, 29.5]:
        for ax in axes: ax.axhline(i, color='#76858b', lw=.6)
    fig.text(.035, .955, 'N16 | A three-layer development site passes the fixed gate', fontsize=17, weight='bold', color='#173640')
    fig.text(.035, .916, 'All 31 nonempty subsets of layers 5 to 9 are retained. No singleton or pair passes; five triples do.', fontsize=10.5)
    fig.text(.035, .883, 'Orange: selected layers 5, 6, 9 by fewest layers then lexicographic order. Selection does not use the highest score.', fontsize=10)
    fig.text(.035, .12, 'Dotted cutoffs: transfer at least 8; preservation at least 14. Coverage and all eight retention cases also required.', fontsize=9)
    fig.text(.035, .087, 'N16: 1,880 audited records, 1,680 new and 200 carried. Parent single-layer, adjacent-pair and full-block results are closed.', fontsize=9)
    fig.text(.035, .054, 'Zero training. Reused development; conditional donor sufficiency, not individual neuron meaning, formal synergy or coding improvement.', fontsize=9)
    out = ROOT / ('research/figures' if private else 'docs/figures'); out.mkdir(parents=True, exist_ok=True); paths = []
    for suffix in ['png', 'svg']:
        path = out / ('n16_summary.' + suffix); fig.savefig(path, dpi=150, facecolor='white')
        if suffix == 'svg': path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
        paths.append(path)
    plt.close(fig)
    if private:
        receipt = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'all31site_gates_represented': True, 'new_model_forwards': 0, 'optimizer_steps': 0,
            'files': {str(p.relative_to(ROOT)): {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size} for p in paths},
            'visual_QA': 'pending'}
        (ROOT / 'runs/n16/version1/figure_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')

if __name__ == '__main__': main()
