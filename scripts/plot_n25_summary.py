"""Plot the complete audited N25 public aggregates, without private inputs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ORDER = ["base", "identity_selected64_gain1", "selected64_gain2",
         "random34201_64_gain2", "random34202_64_gain2"]
LABELS = ["Original", "Identity", "Selected 64", "Random A64", "Random B64"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    summary = data.get("descriptive", data)
    rows = {row["condition"]: row for row in summary["conditions"]}
    assert set(rows) == set(ORDER)
    assert all(row["coding_total"] == 32 and row["retention_total"] == 8 for row in rows.values())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.hashsalt": "n25-complete-fixed-grid"})
    fig = plt.figure(figsize=(13.2, 9.6), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.08])
    a, b, c = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])
    x = np.arange(5)
    a.bar(x - .19, [rows[name]["coding_correct"] for name in ORDER], .36,
          label="Coding /32", color="#246e7c")
    a.bar(x + .19, [rows[name]["retention_correct"] for name in ORDER], .36,
          label="Retention /8", color="#c98a43")
    for index, name in enumerate(ORDER):
        for offset, key in [(-.19, "coding_correct"), (.19, "retention_correct")]:
            value = rows[name][key]
            a.text(index + offset, value + .35, str(value), ha="center", fontsize=9)
    a.set(xticks=x, xticklabels=LABELS, ylim=(0, 35), ylabel="Correct cases",
          title="A  Every fixed condition")
    a.tick_params(axis="x", labelrotation=18)
    a.legend(loc="upper right", frameon=False)
    comparators = ["base", "random34201_64_gain2", "random34202_64_gain2"]
    contrasts = summary["contrasts"]
    gains = [contrasts[name]["gain_count"] for name in comparators]
    losses = [contrasts[name]["loss_count"] for name in comparators]
    positions = np.arange(3)
    b.barh(positions, gains, color="#2c7960", label="Selected gains")
    b.barh(positions, [-value for value in losses], color="#b66a56", label="Selected losses")
    for index, (gain, loss) in enumerate(zip(gains, losses)):
        b.text(gain + .15, index, str(gain), va="center")
        b.text(-loss - .15, index, str(loss), va="center", ha="right")
    bound = max(2, max(gains + losses) + 1)
    b.set(yticks=positions, yticklabels=["vs original", "vs random A64", "vs random B64"],
          xlim=(-bound, bound), xlabel="Paired cases out of 32", title="B  All selected gains and losses")
    b.axvline(0, color="#7f8b91", lw=.8)
    b.invert_yaxis()
    b.legend(loc="lower right", frameon=False)
    families = list(rows["base"]["family_correct"])
    assert len(families) == 8
    matrix = np.array([[rows[name]["family_correct"][family] for family in families] for name in ORDER])
    image = c.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=4, aspect="auto")
    for (i, j), value in np.ndenumerate(matrix):
        c.text(j, i, str(value), ha="center", va="center", color="white" if value >= 3 else "#163b43")
    c.set(xticks=np.arange(8), xticklabels=[name.replace("_", "\n") for name in families],
          yticks=np.arange(5), yticklabels=LABELS, title="C  Complete family grid, correct /4")
    fig.colorbar(image, ax=c, ticks=np.arange(5), label="Correct /4", fraction=.024, pad=.02)
    passed = summary["development_signal_gate_passed"]
    fig.suptitle("N25: runtime coding controller, complete fixed grid", fontsize=17, weight="bold")
    gate = "passed" if passed else "failed"
    fig.supxlabel("200 records: 40 exact original carries and 160 new B outputs. Eight dependent development families; not confirmation.\n"
                  f"Identity is an exact guarded control. Fixed development signal gate {gate}. Zero optimizer updates.", fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=170, metadata={"Software": "Gemma Neuron Engineering"})
    fig.savefig(args.output.with_suffix(".svg"), metadata={"Date": None, "Creator": "Gemma Neuron Engineering"})
    plt.close(fig)


if __name__ == "__main__":
    main()
