"""Finite native-state contrasts without fitted coefficients or causal fractions."""

import torch


def contrast(logits, original, original_winner, original_other):
    if logits.ndim != 1 or logits.shape != original.shape or not torch.isfinite(logits).all():
        raise ValueError('Aligned finite full-vocabulary logits required')
    if original_winner == original_other:
        raise ValueError('Distinct native contrast tokens required')
    value, base = logits.float(), original.float()
    delta = value.double() - base.double()
    return {'argmax': int(value.argmax()),
        'native_margin': float(value[original_winner].double() - value[original_other].double()),
        'native_winner_probability': float(value.softmax(-1)[original_winner]),
        'delta_L2': float(delta.square().sum().sqrt()), 'delta_max_abs': float(delta.abs().max())}


def effects(cells, winner, other):
    if set(cells) != {'00', '10', '01', '11'}:
        raise ValueError('All four prescribed native-state cells required')
    base = cells['00']
    summary = {key: contrast(value, base, winner, other) for key, value in cells.items()}
    raw = cells['10'].double() - base.double()
    norm = cells['01'].double() - base.double()
    joint = cells['11'].double() - base.double()
    raw_norm, norm_norm = raw.square().sum().sqrt(), norm.square().sum().sqrt()
    interaction = joint - raw - norm
    return {'cells': summary, 'interaction_L2': float(interaction.square().sum().sqrt()),
        'interaction_margin': float(interaction[winner] - interaction[other]),
        'raw_normalization_cosine': None if not raw_norm or not norm_norm else float((raw * norm).sum() / (raw_norm * norm_norm))}
