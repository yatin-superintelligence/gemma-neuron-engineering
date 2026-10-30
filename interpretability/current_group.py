"""One current-token channel edit with compact exact target witnesses."""
from contextlib import contextmanager
import math

import torch

from interpretability.decode_step import frozen_versions
from interpretability.scoped_channel import bit_equal, digest


@contextmanager
def current_group(model, coordinates, operation, donor=None, gain=1):
    """Edit scalar channels once; retain targets, never full activations.

    The caller must prepare and pass a real cache. Sequence length one alone
    cannot establish cache presence. Hooks remove before subsequent decoding.
    """
    if operation not in {"gain", "swap", "rescue"} or not math.isfinite(gain):
        raise ValueError("A prescribed finite gain or actual donor is required")
    versions = frozen_versions(model)
    layers = model.model.language_model.layers
    grouped = {}
    seen = set()
    for coordinate in coordinates:
        layer, channel = coordinate["layer"], coordinate["channel"]
        if (layer, channel) in seen or not 0 <= layer < len(layers):
            raise ValueError("Unique native coordinates required")
        if not 0 <= channel < layers[layer].mlp.down_proj.in_features:
            raise ValueError("Channel outside native layer")
        seen.add((layer, channel))
        grouped.setdefault(layer, []).append(coordinate)
    if not grouped:
        raise ValueError("Empty native group")
    receipt = {"operation": operation, "gain": gain, "coordinates": len(seen),
               "hits": {layer: 0 for layer in grouped}, "scope": [],
               "optimizer_steps": 0}
    handles = []
    success = False
    try:
        for layer, group in sorted(grouped.items()):
            channels = [c["channel"] for c in group]
            values = None
            if operation != "gain":
                if donor is None:
                    raise ValueError("Actual donor values required")
                values = [float(donor[c["flat"]]) for c in group]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("Nonfinite donor value")

            def change(module, arguments, layer=layer, channels=channels, values=values):
                before = arguments[0]
                if before.ndim != 3 or before.shape[:2] != (1, 1):
                    raise ValueError("One current cached token required")
                if receipt["hits"][layer]:
                    raise ValueError("Only one forward per current-token context")
                receipt["hits"][layer] += 1
                prior = before[:, :, channels].detach().clone()
                assert torch.isfinite(prior).all()
                expected = (prior * gain if operation == "gain" else
                            torch.tensor(values, device=before.device,
                                         dtype=before.dtype).view(1, 1, -1))
                assert torch.isfinite(expected).all()
                after = before.clone()
                if operation == "rescue":
                    after[:, :, channels] = 0
                after[:, :, channels] = expected
                outside = after.clone()
                outside[:, :, channels] = prior
                assert bit_equal(outside, before)
                assert bit_equal(after[:, :, channels], expected)
                receipt["scope"].append({
                    "layer": layer, "channels": channels,
                    "other_values_bit_exact": True,
                    "before_target_sha256": digest(prior),
                    "after_target_sha256": digest(after[:, :, channels]),
                    "expected_target_sha256": digest(expected),
                    "target_dtype": str(before.dtype),
                    "target_before_values": prior.float().cpu().flatten().tolist(),
                    "target_after_values": expected.float().cpu().flatten().tolist(),
                })
                return (after, *arguments[1:])

            handles.append(layers[layer].mlp.down_proj.register_forward_pre_hook(change))
        yield receipt
        success = True
    finally:
        for handle in handles:
            handle.remove()
        assert versions == {name: p._version for name, p in model.named_parameters()}
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        if success:
            assert all(hits == 1 for hits in receipt["hits"].values())
        receipt["native_parameter_versions_unchanged"] = True
