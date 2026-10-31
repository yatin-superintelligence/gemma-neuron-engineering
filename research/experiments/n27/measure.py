"""Compact first-logit and complete-cache witnesses for existing native paths."""
import torch

from interpretability.prefix_decode import cache_fingerprint
from interpretability.scoped_channel import digest


def snapshot(output, slots=15):
    logits = output.logits[0, -1].detach().float().cpu()
    assert torch.isfinite(logits).all()
    return {'full_logits_sha256': digest(logits), 'top1': int(logits.argmax()),
            'cache': cache_fingerprint(output.past_key_values, slots)}


def forward(model, encoded, slots=15):
    with torch.no_grad():
        output = model(**encoded, use_cache=True, logits_to_keep=1)
    return output, snapshot(output, slots)


def trace(model, encoded, reference_tokens, slots=15):
    """Teacher tokens are lineage guards, never the free-generation prefix."""
    output, first = forward(model, encoded, slots)
    states = [first]
    for token in reference_tokens:
        ids = torch.tensor([[token]], dtype=torch.long, device=encoded['input_ids'].device)
        with torch.no_grad():
            output = model(input_ids=ids, past_key_values=output.past_key_values,
                           use_cache=True, logits_to_keep=1)
        states.append(snapshot(output, slots))
    return states
