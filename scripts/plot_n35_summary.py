"""Complete N35 unmodified-checkpoint results from a reviewed aggregate."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=root / "runs/n35/version3/public_aggregate.json"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "research/figures/n35_summary"
    )
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    assert data["records"] == 96 and not data["models_edited"]
    families = data["all_eight_families"]
    assert len(families) == 8 and all(f["cases"] == 4 for f in families)
    plt.rcParams.update(
        {"font.size": 11, "svg.hashsalt": "n35-all-unmodified-model-cases"}
    )
    fig, axes = plt.subplots(
        3, 1, figsize=(12, 11), height_ratios=[2, 0.8, 1.2], layout="constrained"
    )
    colors = {"recipient": "#087f8c", "donor": "#c77b42"}
    a = axes[0]
    y = np.arange(8)
    for condition, shift in [("recipient", -0.18), ("donor", 0.18)]:
        values = [f[condition]["correct"] for f in families]
        a.barh(
            y + shift,
            values,
            height=0.32,
            color=colors[condition],
            label=condition.title(),
        )
        for i, v in enumerate(values):
            a.text(v + 0.06, i + shift, str(v), va="center")
    a.set_yticks(y, [f["family"].replace("_", " ").title() for f in families])
    a.invert_yaxis()
    a.set_xlim(0, 4.7)
    a.set_xticks(range(5))
    a.set_xlabel("Correct of four dependent wording/name variants")
    a.set_title("All eight coding families", loc="left", weight="bold")
    a.legend(frameon=False, ncols=2, loc="lower right")
    a = axes[1]
    a.axis("off")
    lines = []
    for role, denominator in [("coding", 32), ("retention", 16)]:
        t = data["totals"][role]
        lines.append(
            f"{role.title()} /{denominator}: recipient {t['recipient']['correct']}, donor {t['donor']['correct']}; paired gains {t['gains']}, losses {t['losses']}."
        )
        lines.append(
            f"Caps: recipient {t['recipient']['caps']}, donor {t['donor']['caps']}. All cases remain in the denominator."
        )
    a.text(0, 0.98, "\n".join(lines), va="top", linespacing=1.7)
    a = axes[2]
    a.axis("off")
    clauses = list(data["adequacy_clauses"].items()) + list(
        data["quality_clauses"].items()
    )
    text = "\n".join(
        ("PASS" if value else "FAIL") + "  " + name.replace("_", " ")
        for name, value in clauses
    )
    a.text(0, 0.98, text, va="top", linespacing=1.5, fontsize=10)
    verdict = "PASS" if data["N36_prerequisite_passed"] else "FAIL"
    a.text(
        0.56,
        0.98,
        "Transplant prerequisite: " + verdict + "\n\n"
        "No recipient weights are edited in N35.\n"
        "48 original responses are preserved, not regenerated.\n"
        "48 new donor responses complete the paired panel.\n"
        "Both engineering stops remain recorded.\n"
        "Zero backwards, optimizer updates or weight assignments.\n\n"
        "Development donor selection, not fresh confirmation.",
        va="top",
        linespacing=1.5,
        fontsize=10,
    )
    fig.suptitle(
        "N35 | Unmodified recipient and donor coding quality",
        fontsize=16,
        weight="bold",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "svg"]:
        fig.savefig(
            args.output.with_suffix("." + suffix),
            dpi=140,
            metadata={"Date": None} if suffix == "svg" else {},
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
