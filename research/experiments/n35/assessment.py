"""Fixed donor adequacy and quality clauses, separate from execution witnesses."""


def assess(rows, grades):
    coding = [r for r in rows if r["role"] == "coding"]
    retention = [r for r in rows if r["role"] == "retention"]
    assert len(coding) == 32 and len(retention) == 16
    scores = {
        c: {
            role: sum(grades[c, r["id"]]["correct"] for r in values)
            for role, values in [("coding", coding), ("retention", retention)]
        }
        for c in ["recipient", "donor"]
    }
    gains = [
        r["id"]
        for r in coding
        if not grades["recipient", r["id"]]["correct"]
        and grades["donor", r["id"]]["correct"]
    ]
    losses = [
        r["id"]
        for r in coding
        if grades["recipient", r["id"]]["correct"]
        and not grades["donor", r["id"]]["correct"]
    ]
    families = sorted({r["family"] for r in coding if r["id"] in gains})
    retention_losses = [
        r["id"]
        for r in retention
        if grades["recipient", r["id"]]["correct"]
        and not grades["donor", r["id"]]["correct"]
    ]
    adequacy = {
        "coding_at_least_16": scores["recipient"]["coding"] >= 16,
        "coding_at_least_4_errors": 32 - scores["recipient"]["coding"] >= 4,
        "retention_at_least_12": scores["recipient"]["retention"] >= 12,
    }
    quality = {
        "net_gain_at_least_4": len(gains) - len(losses) >= 4,
        "gain_families_at_least_2": len(families) >= 2,
        "new_retention_losses_at_most_1": len(retention_losses) <= 1,
        "retention_score_within_1": scores["donor"]["retention"]
        >= scores["recipient"]["retention"] - 1,
    }
    return {
        "scores": scores,
        "gains": gains,
        "losses": losses,
        "gain_families": families,
        "retention_losses": retention_losses,
        "adequacy_clauses": adequacy,
        "quality_clauses": quality,
        "adequacy_passed": all(adequacy.values()),
        "donor_quality_passed": all(quality.values()),
        "N36_prerequisite_passed": all(adequacy.values()) and all(quality.values()),
        "interpretation": "A paired unmodified-donor comparison does not improve the recipient. Eight algorithm families have dependent wording/name variants. This is development selection, not transplant confirmation.",
    }
