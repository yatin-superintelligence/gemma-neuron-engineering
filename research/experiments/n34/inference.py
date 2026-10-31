"""Native block patches and isolated teacher/free paths for the frozen N34 study."""

from contextlib import ExitStack, contextmanager

import torch
from transformers import StopStringCriteria
from transformers.cache_utils import DynamicCache

from interpretability.cache_proofs import check_cache
from interpretability.mlp_composition import compose_mlp_donors
from interpretability.native_capture import capture_native_mlp
from research.experiments.n19.generation_contract import resolve_generation_contract
from research.experiments.n19.trajectory import cache_witness
from research.experiments.n28.measure import snapshot as legacy_snapshot
from research.experiments.n35.core import snapshot, stable, tensor_sha

FOREIGN = [
    "opposite_same_order",
    "same_same_order",
    "opposite_reversed_order",
    "same_reversed_order",
]


def state(model):
    return {
        **snapshot(model),
        "device_map_sha256": stable(getattr(model, "hf_device_map", None)),
    }


def assert_state(model, before):
    assert state(model) == before, "Native state changed outside an activation scope"


def encode(tokenizer, row, device):
    text = (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        + row["assistant_prefix"]
    )
    value = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    ids = value["input_ids"][0].tolist()
    assert 1 < len(ids) < 512
    return value.to(device), {
        "position": len(ids) - 1,
        "input_tokens": len(ids),
        "last_token": ids[-1],
        "template_sha256": stable(text),
        "input_ids_sha256": stable(ids),
    }


def witness(output, slots=15):
    # The legacy path checks every actual cache tensor for shape and finiteness.
    legacy = legacy_snapshot(output, slots)
    raw = cache_witness(output.past_key_values)
    dtype = str(output.past_key_values.layers[0].keys.dtype)
    check_cache(raw, slots, dtype)
    past = []
    for i, layer in enumerate(output.past_key_values.layers):
        row = {"layer": i}
        for name in ["keys", "values"]:
            value = getattr(layer, name)[:, :, :-1]
            assert value.shape[-2] > 0
            row[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": tensor_sha(value),
            }
        past.append(row)
    return {
        "legacy": legacy,
        "raw_cache": raw,
        "raw_past_cache": {"layers": past, "sha256": stable(past)},
    }


def path_guard(actual, native, layers):
    assert actual["raw_past_cache"] == native["raw_past_cache"]
    assert actual["raw_cache"]["layers"][0] == native["raw_cache"]["layers"][0]
    if min(layers) >= 15:
        assert actual["raw_cache"] == native["raw_cache"]


def fresh(model, encoded):
    cache = DynamicCache(config=model.config.text_config)
    with torch.no_grad():
        return model(**encoded, past_key_values=cache, use_cache=True, logits_to_keep=1)


def raw_cache(cache, dtype, slots=15):
    assert isinstance(cache, DynamicCache) and len(cache.layers) == slots
    for layer in cache.layers:
        for name in ["keys", "values"]:
            value = getattr(layer, name, None)
            assert torch.is_tensor(value) and value.ndim == 4 and value.shape[0] == 1
            assert all(size > 0 for size in value.shape) and str(value.dtype) == dtype
            assert torch.isfinite(value).all()
    result = cache_witness(cache)
    check_cache(result, slots, dtype)
    return result


@contextmanager
def assignment(model, position, layers, operation, row, sources):
    if (
        len(layers) != len(set(layers))
        or not layers
        or any(type(i) is not int or not 0 <= i < 35 for i in layers)
    ):
        raise ValueError("Exact nonempty native layer scope required")
    if operation == "self":
        names = [row["id"]]
    elif operation in FOREIGN:
        names = [row[operation]]
    elif operation == "rescue":
        names = [row["opposite_same_order"], row["id"]]
    elif operation == "zero":
        names = ["zero"]
    else:
        raise ValueError("Undeclared activation operation")
    proofs = []
    with ExitStack() as stack:
        for name in names:
            donor = (
                {
                    "MLP": {
                        i: torch.zeros_like(sources[row["id"]]["MLP"][i])
                        for i in layers
                    }
                }
                if name == "zero"
                else sources[name]
            )
            if any(i not in donor["MLP"] for i in layers):
                raise ValueError("Incomplete actual donor scope")
            # Overlapping ordered assignments use separate nested contexts.
            proof = stack.enter_context(
                compose_mlp_donors(
                    model, position, [{"layers": layers, "donor": donor}]
                )
            )
            proofs.append(proof)
        yield {
            "operation": operation,
            "layers": list(layers),
            "sources": names,
            "contexts": proofs,
            "direct_KV_writes": False,
            "optimizer_steps": 0,
        }


def check_scope(proof):
    expected_contexts = 2 if proof["operation"] == "rescue" else 1
    assert len(proof["contexts"]) == expected_contexts
    for context in proof["contexts"]:
        assert context["optimizer_steps"] == 0 and len(context["parts"]) == 1
        part = context["parts"][0]
        assert part["layers"] == proof["layers"]
        receipt = part["receipt"]
        assert receipt["kind"] == "MLP" and receipt["MLP_prefill_hits"] == len(
            proof["layers"]
        )
        assert (
            receipt["MLP_decode_skips"]
            == receipt["KV_calls"]
            == receipt["optimizer_steps"]
            == 0
        )
        assert receipt["native_parameter_versions_unchanged"]
    # Legacy KV receipt flags mean no direct KV assignment, not no downstream effect.


def stopping(tokenizer, cap=32):
    assert cap > 0
    return StopStringCriteria(tokenizer, stop_strings=["\n"])


def stop_reason(history, token, count, criterion, eos, cap):
    if token in eos:
        return "eos"
    if criterion is not None and bool(criterion(history, None)[0]):
        return "newline"
    return "cap" if count == cap else None


def current_trace(output, step):
    logits = output.logits[0, -1].detach()
    assert torch.isfinite(logits).all()
    return {
        "step": step,
        "argmax": int(logits.argmax()),
        "full_logits_sha256": tensor_sha(logits.float()),
        "raw_cache_sha256": raw_cache(output.past_key_values, str(output.logits.dtype))[
            "sha256"
        ],
    }


def free_tail(model, tokenizer, encoded, output, criterion, eos, cap=32):
    """Receives only the fresh edited state. No teacher token or teacher cache argument."""
    history = encoded["input_ids"].clone()
    ids, trace = [], []
    reason = None
    while len(ids) < cap:
        item = current_trace(output, len(ids))
        token = item["argmax"]
        ids.append(token)
        trace.append(item)
        new = torch.tensor([[token]], dtype=torch.long, device=history.device)
        history = torch.cat([history, new], dim=1)
        reason = stop_reason(history, token, len(ids), criterion, eos, cap)
        if reason:
            break
        with torch.no_grad():
            output = model(
                input_ids=new,
                past_key_values=output.past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
    assert reason and 0 < len(ids) <= cap
    return {
        "generated_ids": ids,
        "text": tokenizer.decode(ids, skip_special_tokens=True),
        "at_cap": len(ids) == cap,
        "termination": reason,
        "generated_tokens": len(ids),
        "trace": trace,
    }


def native_source_teacher(model, encoded, reference_ids, criterion, eos, cap=32):
    """Teacher tokens are checked and consumed only inside this discarded native cache."""
    assert 0 < len(reference_ids) <= cap
    position = encoded["input_ids"].shape[1] - 1
    with capture_native_mlp(model, position, list(range(35))) as source:
        output = fresh(model, encoded)
    first = witness(output)
    logits = output.logits[0, -1].detach().float().cpu()
    history = encoded["input_ids"].clone()
    steps = []
    for i, expected in enumerate(reference_ids):
        item = current_trace(output, i)
        assert item["argmax"] == expected, (
            "Original teacher choice changed before any edit"
        )
        steps.append(item)
        new = torch.tensor([[expected]], dtype=torch.long, device=history.device)
        history = torch.cat([history, new], dim=1)
        reason = stop_reason(history, expected, i + 1, criterion, eos, cap)
        assert bool(reason) == (i == len(reference_ids) - 1), (
            "Original termination changed"
        )
        if reason:
            break
        with torch.no_grad():
            output = model(
                input_ids=new,
                past_key_values=output.past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
    assert len(source["MLP"]) == 35
    assert all(torch.isfinite(value).all() for value in source["MLP"].values())
    return (
        source,
        first,
        logits,
        {"reference_steps": steps, "termination": reason, "new_generated_tokens": 0},
    )


def resolved_contract(model):
    contract = resolve_generation_contract(model)
    config, leftover = model._prepare_generation_config(
        None, do_sample=False, use_cache=True, max_new_tokens=32
    )
    actual = config.to_dict()
    assert not leftover and actual["max_new_tokens"] == 32
    assert {k: v for k, v in actual.items() if k != "max_new_tokens"} == {
        k: v for k, v in contract.items() if k != "max_new_tokens"
    }
    return {**actual, "stop_strings": ["\n"], "manual_full_vocabulary": True}
