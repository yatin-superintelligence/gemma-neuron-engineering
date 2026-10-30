"""Compact finite decision metrics, computed from actual inference logits."""
import numpy as np


def vector(value):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size < 2 or not np.isfinite(result).all():
        raise ValueError("A finite vocabulary vector is required")
    return result


def log_partition(value):
    highest = value.max()
    return float(highest + np.log(np.exp(value - highest).sum(dtype=np.float64)))


def summarize(logits, own, opposite, native):
    """Return scalar metrics only; callers retain no full output vectors.

    FP64 arithmetic here does not change the model inference precision.
    KL is reported as computed, including tiny floating point roundoff.
    """
    values, baseline = vector(logits), vector(native)
    if values.shape != baseline.shape:
        raise ValueError("Native and edited vocabulary sizes must agree")
    if own == opposite or not 0 <= own < values.size or not 0 <= opposite < values.size:
        raise ValueError("Two distinct valid native token IDs are required")
    normalizer, base_normalizer = log_partition(values), log_partition(baseline)
    logp, base_logp = values - normalizer, baseline - base_normalizer
    delta = values - baseline
    centered = delta - delta.mean(dtype=np.float64)
    rms = float(np.sqrt(np.mean(delta * delta, dtype=np.float64)))
    margin = float(values[opposite] - values[own])
    delta_margin = float(margin - (baseline[opposite] - baseline[own]))
    order = np.lexsort((np.arange(values.size), -values))[:8]
    return {
        "vocabulary_size": values.size,
        "own_token_id": own, "opposite_token_id": opposite,
        "own_logit": float(values[own]), "opposite_logit": float(values[opposite]),
        "operator_margin": margin, "margin_shift_from_native": delta_margin,
        "logsumexp": normalizer,
        "own_log_probability": float(logp[own]),
        "opposite_log_probability": float(logp[opposite]),
        "top1_token_id": int(order[0]),
        "top8": [{"token_id": int(i), "logit": float(values[i])} for i in order],
        "max_abs_change": float(np.abs(delta).max()),
        "L2_change": float(np.linalg.norm(delta)),
        "centered_L2_change": float(np.linalg.norm(centered)),
        "vocabulary_RMS_change": rms,
        "KL_native_to_edited": float(np.sum(np.exp(base_logp) * (base_logp - logp), dtype=np.float64)),
        "own_logit_shift": float(delta[own]),
        "opposite_logit_shift": float(delta[opposite]),
        "margin_shift_over_vocabulary_RMS": delta_margin / rms if rms else None,
        "own_shift_over_vocabulary_RMS": float(delta[own]) / rms if rms else None,
        "opposite_shift_over_vocabulary_RMS": float(delta[opposite]) / rms if rms else None,
        "zero_vocabulary_change": rms == 0,
    }
