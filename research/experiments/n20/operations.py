"""Compose actual current-state donors without changing native parameters."""
from contextlib import contextmanager

from interpretability.current_group import current_group
from interpretability.decode_step import edit_decode_step


@contextmanager
def policy(model, operation, coordinates, recipient, opposite, samepolicy, full_coordinates):
    """Apply a full parent or a partial native/conditional intervention.

    Donors are actual unmodified state values. Registration order is explicit:
    a full opposite hook runs first, then the conditional group hook.
    """
    receipt = {"background": None, "group": None}
    if coordinates is None:
        source = opposite if operation == "opposite" else samepolicy if operation == "samepolicy" else recipient
        with edit_decode_step(model, full_coordinates, operation="swap", donor=source) as full:
            receipt["background"] = full
            yield receipt
        return
    if operation in {"conditional_cut", "conditional_self"}:
        with edit_decode_step(model, full_coordinates, operation="swap", donor=opposite) as full:
            with current_group(model, coordinates, "swap",
                               donor=recipient if operation == "conditional_cut" else opposite) as group:
                receipt.update(background=full, group=group)
                yield receipt
        return
    op = "gain" if operation in {"zero", "gain2"} else "rescue" if operation == "rescue" else "swap"
    source = opposite if operation == "opposite" else samepolicy if operation == "samepolicy" else recipient
    with current_group(model, coordinates, op, donor=source,
                       gain=0 if operation == "zero" else 2 if operation == "gain2" else 1) as group:
        receipt["group"] = group
        yield receipt
