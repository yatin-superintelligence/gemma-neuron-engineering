"""Audit compact target scalars for prescribed current-token channel edits."""
import numpy as np
import torch
from interpretability.scoped_channel import digest


def check_group_receipt(receipt, coordinates, operation, source_values, opposite_values=None):
    """Check all saved BF16 scalar assignments, including dynamic gain inputs."""
    assert receipt["optimizer_steps"] == 0 and receipt["native_parameter_versions_unchanged"]
    expected_op = "gain" if operation in {"zero", "gain2"} else "rescue" if operation == "rescue" else "swap"
    assert receipt["operation"] == expected_op
    assert receipt["gain"] == (0 if operation == "zero" else 2 if operation == "gain2" else 1)
    grouped = {}
    for index, coordinate in enumerate(coordinates):
        grouped.setdefault(coordinate["layer"], []).append((index, coordinate["channel"]))
    assert receipt["coordinates"] == len(coordinates)
    assert receipt["hits"] == {str(layer): 1 for layer in grouped}
    assert [scope["layer"] for scope in receipt["scope"]] == sorted(grouped)
    for scope in receipt["scope"]:
        indices, channels = zip(*grouped[scope["layer"]])
        assert scope["channels"] == list(channels)
        assert scope["target_dtype"] == "torch.bfloat16" and scope["other_values_bit_exact"]
        before_values, after_values = scope["target_before_values"], scope["target_after_values"]
        assert len(before_values) == len(after_values) == len(channels)
        assert np.isfinite(before_values).all() and np.isfinite(after_values).all()
        before = torch.tensor(before_values, dtype=torch.bfloat16).view(1, 1, -1)
        after = torch.tensor(after_values, dtype=torch.bfloat16).view(1, 1, -1)
        assert before.float().flatten().tolist() == before_values
        assert after.float().flatten().tolist() == after_values
        assert digest(before) == scope["before_target_sha256"]
        assert digest(after) == scope["after_target_sha256"] == scope["expected_target_sha256"]
        expected = (before * receipt["gain"] if expected_op == "gain" else
                    torch.tensor([source_values[i] for i in indices], dtype=torch.bfloat16).view(1, 1, -1))
        assert digest(after) == digest(expected)
        if operation in {"rescue", "conditional_self"}:
            assert digest(before) == digest(after)
        if opposite_values is not None:
            expected_before = torch.tensor([opposite_values[i] for i in indices], dtype=torch.bfloat16).view(1, 1, -1)
            assert digest(before) == digest(expected_before)
