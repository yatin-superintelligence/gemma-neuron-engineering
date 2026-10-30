"""Inspect prescribed gains around Gemma's post-MLP normalization on CPU.

This uses random module weights and constructed vectors, never a pretrained
checkpoint, text task, fitted gain or optimizer. Results are engineering facts
about these fixtures, not coding performance or a neuron interpretation.
"""
from contextlib import contextmanager
from pathlib import Path
import copy
import datetime
import hashlib
import inspect
import json
import time
import traceback

import torch
import transformers
from transformers import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm, Gemma4TextMLP, Gemma4TextDecoderLayer


def tensor_sha(value):
    return hashlib.sha256(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def difference(before, after):
    a, b = before.detach().double(), after.detach().double()
    return {"byte_equal": tensor_sha(before) == tensor_sha(after),
            "maxabs": float((b - a).abs().max()),
            "relative_L2": float((b - a).norm() / a.norm().clamp_min(1e-30)),
            "L2_norm_ratio": float(b.norm() / a.norm().clamp_min(1e-30)),
            "before_sha256": tensor_sha(before), "after_sha256": tensor_sha(after)}


@contextmanager
def columns(mlp, selected, factor):
    weight = mlp.down_proj.weight
    original = weight.detach().clone()
    version = weight._version
    expected = original[:, selected] * factor
    assert torch.isfinite(expected).all() and not weight.requires_grad
    receipt = {"selected": selected, "factor": factor,
               "original_matrix_sha256": tensor_sha(original), "optimizer_steps": 0}
    try:
        with torch.no_grad():
            weight[:, selected] = expected
        assert tensor_sha(weight[:, selected]) == tensor_sha(expected)
        outside = weight.detach().clone()
        outside[:, selected] = original[:, selected]
        assert tensor_sha(outside) == tensor_sha(original)
        assert weight._version == version + 1
        receipt.update(other_columns_byte_exact=True, edited_matrix_sha256=tensor_sha(weight))
        yield receipt
    finally:
        with torch.no_grad():
            weight[:, selected] = original[:, selected]
        assert tensor_sha(weight) == tensor_sha(original) and weight._version == version + 2
        receipt.update(original_matrix_restored_exact=True, declared_parameter_version_increments=2)


@contextmanager
def activity(mlp, selected, factor):
    receipt = {"hits": 0, "selected": selected, "factor": factor, "optimizer_steps": 0}

    def change(module, arguments):
        before = arguments[0]
        receipt["hits"] += 1
        assert receipt["hits"] == 1
        after = before.clone()
        after[..., selected] = before[..., selected] * factor
        outside = after.clone()
        outside[..., selected] = before[..., selected]
        assert tensor_sha(outside) == tensor_sha(before)
        receipt["other_input_channels_byte_exact"] = True
        return (after, *arguments[1:])

    handle = mlp.down_proj.register_forward_pre_hook(change)
    try:
        yield receipt
        assert receipt["hits"] == 1
    finally:
        handle.remove()


def run():
    started = time.monotonic()
    root = Path(__file__).resolve().parents[3]
    protocol_path = root / "research/experiments/n21/postnorm_host_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    assert transformers.__version__ == "5.8.0"
    for name, expected in protocol["installed_class_source_sha256"].items():
        cls = {"Gemma4TextMLP": Gemma4TextMLP, "Gemma4TextDecoderLayer": Gemma4TextDecoderLayer,
               "Gemma4RMSNorm": Gemma4RMSNorm}[name]
        assert hashlib.sha256(inspect.getsource(cls).encode()).hexdigest() == expected
    torch.set_num_threads(1)
    torch.manual_seed(protocol["seed"])
    epsilon = protocol["rms_epsilon"]
    metrics = {"MLP_calls": 0, "MLP_synthetic_positions": 0, "norm_calls": 0, "norm_synthetic_positions": 0}

    def meter(kind):
        def count(module, arguments):
            value = arguments[0]
            metrics[kind + "_calls"] += 1
            metrics[kind + "_synthetic_positions"] += value.numel() // value.shape[-1]
        return count

    amplitudes = []
    raw_parent = torch.randn(1, 1, 16, dtype=torch.float64)
    for amplitude in protocol["analytical_probe"]["input_amplitudes"]:
        raw = raw_parent * amplitude
        m = raw.square().mean(-1, keepdim=True)
        normalized = raw / (m + epsilon).sqrt()
        doubled = (2 * raw) / (4 * m + epsilon).sqrt()
        ratio = ((m + epsilon) / (m + epsilon / 4)).sqrt()
        error = float((doubled - normalized * ratio).norm() / doubled.norm())
        assert error <= 1e-12
        for dtype in (torch.float32, torch.bfloat16):
            norm = Gemma4RMSNorm(16, eps=epsilon).to(dtype).eval().requires_grad_(False)
            handle = norm.register_forward_pre_hook(meter("norm"))
            with torch.no_grad():
                before, after = norm(raw.to(dtype)), norm((raw * 2).to(dtype))
            handle.remove()
            amplitudes.append({"amplitude": amplitude, "input_dtype": str(dtype),
                               "FP64_raw_mean_square": float(m.item()), "epsilon": epsilon,
                               "FP64_exact_uniform_gain_ratio": float(ratio.item()),
                               "FP64_theorem_relative_error": error,
                               "actual_RMSNorm_output": difference(before, after)})
    coupling_norm = Gemma4RMSNorm(4, eps=epsilon).eval().requires_grad_(False)
    handle = coupling_norm.register_forward_pre_hook(meter("norm"))
    raw_before = torch.tensor(protocol["coupling_probe"]["raw_before"], dtype=torch.float32).view(1, 1, -1)
    raw_after = torch.tensor(protocol["coupling_probe"]["raw_after"], dtype=torch.float32).view(1, 1, -1)
    with torch.no_grad():
        normalized_before, normalized_after = coupling_norm(raw_before), coupling_norm(raw_after)
    handle.remove()
    assert torch.equal(raw_before[..., 1:], raw_after[..., 1:])
    assert torch.count_nonzero(normalized_before[..., 1:] != normalized_after[..., 1:]).item() == 3
    coupling = {"raw_unchanged_coordinates": [1, 2, 3], "normalized_changed_coordinates": [1, 2, 3],
                "normalized_difference": difference(normalized_before, normalized_after),
                "constructed_arithmetic_example": True}

    setting = protocol["module_probe"]
    config = Gemma4TextConfig(hidden_size=setting["hidden_size"], intermediate_size=setting["intermediate_size"],
                             num_hidden_layers=6, num_kv_shared_layers=2, hidden_activation="gelu_pytorch_tanh")
    parent = Gemma4TextMLP(config, layer_idx=0).eval().requires_grad_(False)
    inputs = torch.randn(1, setting["synthetic_positions"], setting["hidden_size"])
    modules, comparisons = [], []
    for dtype in (torch.float32, torch.bfloat16):
        mlp = copy.deepcopy(parent).to(dtype)
        norm = Gemma4RMSNorm(setting["hidden_size"], eps=epsilon).to(dtype).eval().requires_grad_(False)
        x = inputs.to(dtype)
        parameters = {"MLP." + n: p for n, p in mlp.named_parameters()}
        parameters.update({"norm." + n: p for n, p in norm.named_parameters()})
        versions = {n: p._version for n, p in parameters.items()}
        hashes = {n: tensor_sha(p) for n, p in parameters.items()}
        flags = {n: p.requires_grad for n, p in parameters.items()}
        hook_counts = {n: (len(m._forward_pre_hooks), len(m._forward_hooks)) for n, m in mlp.named_modules()}
        mlp_meter = mlp.register_forward_pre_hook(meter("MLP"))
        norm_meter = norm.register_forward_pre_hook(meter("norm"))
        outcomes = {}

        def observe(name, receipt=None):
            with torch.no_grad():
                raw = mlp(x)
                normalized = norm(raw)
            outcomes[name] = (raw.clone(), normalized.clone())
            modules.append({"input_dtype": str(dtype), "condition": name,
                            "raw_output_sha256": tensor_sha(raw), "normalized_output_sha256": tensor_sha(normalized),
                            "raw_mean_square": float(raw.double().square().mean()),
                            "raw_L2": float(raw.double().norm()), "normalized_L2": float(normalized.double().norm()),
                            "scope_receipt": receipt})

        observe("base")
        with columns(mlp, list(range(setting["intermediate_size"])), 2) as receipt:
            observe("uniform_all_columns2", receipt)
        for factor in (0, 2):
            with columns(mlp, setting["group"], factor) as receipt:
                observe("group_stored" + str(factor), receipt)
            with activity(mlp, setting["group"], factor) as receipt:
                observe("group_activity" + str(factor), receipt)
            comparisons.append({"input_dtype": str(dtype), "factor": factor,
                                "raw_stored_vs_activity": difference(outcomes["group_stored" + str(factor)][0], outcomes["group_activity" + str(factor)][0]),
                                "normalized_stored_vs_activity": difference(outcomes["group_stored" + str(factor)][1], outcomes["group_activity" + str(factor)][1])})
        comparisons.append({"input_dtype": str(dtype), "uniform_all_columns2_vs_base": {
            "raw": difference(outcomes["base"][0], outcomes["uniform_all_columns2"][0]),
            "normalized": difference(outcomes["base"][1], outcomes["uniform_all_columns2"][1])}})
        assert [r["condition"] for r in modules if r["input_dtype"] == str(dtype)] == setting["conditions"]
        assert parameters["MLP.down_proj.weight"]._version == versions["MLP.down_proj.weight"] + 6
        assert all(p._version == versions[n] for n, p in parameters.items() if n != "MLP.down_proj.weight")
        assert hashes == {n: tensor_sha(p) for n, p in parameters.items()}
        assert flags == {n: p.requires_grad for n, p in parameters.items()}
        assert all(not p.requires_grad and p.grad is None for p in parameters.values())
        mlp_meter.remove()
        norm_meter.remove()
        assert hook_counts == {n: (len(m._forward_pre_hooks), len(m._forward_hooks)) for n, m in mlp.named_modules()}
    assert metrics == {"MLP_calls": 12, "MLP_synthetic_positions": 36,
                       "norm_calls": 26, "norm_synthetic_positions": 50}
    return {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "status": "host_math_and_scope_probe_complete",
            "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "seed": protocol["seed"],
            "official_selected_class_AST_exact": True, "uniform_scale_probes": amplitudes,
            "coupling_probe": coupling, "module_conditions": modules, "paired_comparisons": comparisons,
            "actual_counters": metrics, "actual_decoder_model_forwards": 0, "pretrained_model_forwards": 0,
            "text_input_tokens": 0, "generated_tokens": 0, "backward_calls": 0, "optimizer_steps": 0,
            "all_scope_parameter_and_restoration_guards_pass": True, "seconds": time.monotonic() - started,
            "environment": {"torch": torch.__version__, "transformers": transformers.__version__, "device": "CPU"},
            "scope": "Synthetic vectors and random-weight modules only. No semantic role, pretrained numerical equivalence, universal error floor, efficiency or coding improvement."}


def main():
    root = Path(__file__).resolve().parents[3]
    out = root / "runs/n21/host_postnorm"
    out.mkdir(parents=True, exist_ok=True)
    try:
        result = run()
    except Exception as error:
        name = "failure_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S") + ".json"
        (out / name).write_text(json.dumps({"error": repr(error), "traceback": traceback.format_exc(),
                                          "pretrained_model_forwards": 0, "optimizer_steps": 0}, indent=2) + "\n")
        raise
    assert not (out / "result.json").exists(), "Never overwrite the first completed probe"
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"uniform_scale_probes", "module_conditions", "paired_comparisons"}}))


if __name__ == "__main__":
    main()
