"""Actual random Gemma block paths before any N34 pretrained forward."""

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Gemma4Config,
    Gemma4ForConditionalGeneration,
    Gemma4TextConfig,
)
from research.experiments.n34.inference import (
    assignment,
    assert_state,
    check_scope,
    free_tail,
    fresh,
    native_source_teacher,
    path_guard,
    raw_cache,
    resolved_contract,
    state,
    stopping,
    witness,
)
from research.experiments.n35.core import Counter, configure

ROOT = Path(__file__).resolve().parents[3]


class Tokens:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


def tiny(dtype, schedule):
    torch.manual_seed(4802)
    text = Gemma4TextConfig(
        vocab_size=32,
        vocab_size_per_layer_input=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=35,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=8,
        num_global_key_value_heads=2,
        num_kv_shared_layers=20,
        hidden_size_per_layer_input=4,
        use_double_wide_mlp=True,
        sliding_window=16,
        max_position_embeddings=128,
        layer_types=schedule,
        rope_parameters={
            k: {"rope_type": "default", "rope_theta": 10000.0}
            for k in ["sliding_attention", "full_attention"]
        },
    )
    config = Gemma4Config(text_config=text.to_dict())
    config.vision_config = config.audio_config = None
    model = (
        Gemma4ForConditionalGeneration(config).to(dtype).eval().requires_grad_(False)
    )
    model.generation_config.eos_token_id = [1, 106, 50]
    model.generation_config.bos_token_id = 2
    model.generation_config.pad_token_id = 0
    return model


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    out = ROOT / "runs/n34/model_host/version1"
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    files = [
        "research/experiments/n34/host_path_protocol.json",
        "research/experiments/n34/model_host.py",
        "research/experiments/n34/inference.py",
        "interpretability/mlp_composition.py",
        "interpretability/native_capture.py",
        "interpretability/selector_site.py",
        "research/experiments/n35/core.py",
    ]
    save(
        out / "source_freeze.json",
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "sources": {
                n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in files
            },
        },
    )
    outcomes, counters, refuses, newline_cases = [], [], [], []
    meter = None
    random_tokens = 0
    try:
        configure()
        config = AutoConfig.from_pretrained(
            "google/gemma-4-E2B-it",
            revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
            cache_dir="/tmp/neuron_n03_tokenizer",
            local_files_only=True,
        )
        blocks = json.loads(
            (ROOT / "research/experiments/n34/protocol.json").read_text()
        )["blocks"]
        for dtype in [torch.float32, torch.bfloat16]:
            model = tiny(dtype, config.text_config.layer_types)
            meter = Counter(model)
            before = state(model)
            assert resolved_contract(model)["max_new_tokens"] == 32
            encoded = {"input_ids": torch.tensor([[3, 4, 5, 6, 7]])}
            foreign = {"input_ids": torch.tensor([[3, 4, 8, 9, 10, 11]])}
            lengths = []

            def measured(label, operation, expected_lengths):
                previous = meter.value()
                result = operation()
                delta = {k: meter.value()[k] - previous[k] for k in previous}
                assert delta == {
                    "calls": len(expected_lengths),
                    "input_token_exposures": sum(expected_lengths),
                }, (label, delta)
                lengths.extend(expected_lengths)
                assert_state(model, before)
                return result

            first_value = measured("own_free", lambda: fresh(model, encoded), [5])
            ordinary = measured(
                "own_cap",
                lambda: free_tail(
                    model, Tokens(), encoded, first_value, None, {-1}, cap=3
                ),
                [1, 1],
            )
            assert ordinary["generated_tokens"] == 3
            random_tokens += 3
            own, native, _, teacher = measured(
                "own_teacher",
                lambda: native_source_teacher(
                    model, encoded, ordinary["generated_ids"], None, {-1}, cap=3
                ),
                [5, 1, 1],
            )
            assert teacher["new_generated_tokens"] == 0
            other_first = measured("foreign_first", lambda: fresh(model, foreign), [6])
            other_generation = measured(
                "foreign_cap",
                lambda: free_tail(
                    model, Tokens(), foreign, other_first, None, {-1}, cap=3
                ),
                [1, 1],
            )
            random_tokens += 3
            other, _, _, _ = measured(
                "foreign_teacher",
                lambda: native_source_teacher(
                    model, foreign, other_generation["generated_ids"], None, {-1}, cap=3
                ),
                [6, 1, 1],
            )
            sources = {"own": own, "other": other}
            row = {"id": "own", "opposite_same_order": "other"}
            for block in blocks:
                scope = block["layers"]
                for operation in ["self", "opposite_same_order", "zero", "rescue"]:

                    def run(operation=operation):
                        with assignment(
                            model, 4, scope, operation, row, sources
                        ) as proof:
                            value = fresh(model, encoded)
                        check_scope(proof)
                        current = witness(value)
                        path_guard(current, native, scope)
                        if operation in ["self", "rescue"]:
                            assert current == native
                        result = {
                            "dtype": str(dtype),
                            "block": block["id"],
                            "operation": operation,
                            "path_and_scope_passed": True,
                            "proof": proof,
                        }
                        json.loads(json.dumps(result))
                        with (out / "outcomes.jsonl").open("a") as f:
                            f.write(json.dumps(result) + "\n")
                        outcomes.append(result)

                    measured("block_path", run, [5])

                def cached_self():
                    native_value = fresh(model, encoded)
                    with assignment(model, 4, scope, "self", row, sources) as proof:
                        patched = fresh(model, encoded)
                    check_scope(proof)
                    assert witness(native_value) == witness(patched)
                    for _ in range(2):
                        assert all(
                            getattr(a, n).untyped_storage().data_ptr()
                            != getattr(b, n).untyped_storage().data_ptr()
                            for a, b in zip(
                                native_value.past_key_values.layers,
                                patched.past_key_values.layers,
                            )
                            for n in ["keys", "values"]
                        )
                        token = int(native_value.logits[0, -1].argmax())
                        ids = torch.tensor([[token]])
                        with torch.no_grad():
                            native_value = model(
                                input_ids=ids,
                                past_key_values=native_value.past_key_values,
                                use_cache=True,
                                logits_to_keep=1,
                            )
                            patched = model(
                                input_ids=ids,
                                past_key_values=patched.past_key_values,
                                use_cache=True,
                                logits_to_keep=1,
                            )
                        assert witness(native_value) == witness(patched)

                measured("paired_cached_self", cached_self, [5, 5, 1, 1, 1, 1])
                restored = measured(
                    "restoration", lambda: witness(fresh(model, encoded)), [5]
                )
                assert restored == native
            ended_value = measured("EOS_first", lambda: fresh(model, encoded), [5])
            ended = measured(
                "EOS_no_more_calls",
                lambda: free_tail(
                    model,
                    Tokens(),
                    encoded,
                    ended_value,
                    None,
                    {ordinary["generated_ids"][0]},
                    cap=3,
                ),
                [],
            )
            assert ended["generated_tokens"] == 1 and ended["termination"] == "eos"
            random_tokens += 1
            for label, scope, donor in [
                ("empty_scope", [], sources),
                ("duplicate_scope", [1, 1], sources),
                ("out_of_range", [35], sources),
                ("missing_donor", [4], {"own": own, "other": {"MLP": {}}}),
            ]:
                try:
                    with assignment(model, 4, scope, "opposite_same_order", row, donor):
                        raise RuntimeError("Malformed scope accepted")
                except ValueError:
                    refuses.append(str(dtype) + ":" + label)
            actual = measured(
                "cache_refusals_source", lambda: fresh(model, encoded), [5]
            )
            for label in ["truncated", "nonfinite", "uninitialized"]:
                cache = copy.deepcopy(actual.past_key_values)
                if label == "truncated":
                    cache.layers.pop()
                elif label == "nonfinite":
                    cache.layers[0].keys[:, :, 0, 0] = float("nan")
                else:
                    cache.layers[0].keys = None
                try:
                    raw_cache(cache, str(dtype))
                except (AssertionError, ValueError):
                    refuses.append(str(dtype) + ":" + label)
                else:
                    raise AssertionError("Malformed cache accepted")
            assert_state(model, before)
            assert meter.value() == {
                "calls": len(lengths),
                "input_token_exposures": sum(lengths),
            }
            counters.append(meter.value())
            meter.remove()
            meter = None
        tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-4-E2B-it",
            revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
            cache_dir="/tmp/neuron_n03_tokenizer",
            local_files_only=True,
        )
        criterion = stopping(tokenizer)
        for text in ["alpha\n    beta", "alpha\n", "alpha\n\tbeta"]:
            ids = tokenizer.encode(text, add_special_tokens=False)
            seen = []
            for i in range(1, len(ids) + 1):
                history = torch.tensor([ids[:i]])
                stop = bool(criterion(history, None)[0])
                seen.append(stop)
                if stop:
                    break
            assert seen[-1] and "\n" in tokenizer.decode(
                ids[: len(seen)], skip_special_tokens=False
            )
            newline_cases.append(
                {
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "first_stop_token": len(seen),
                }
            )
        history = torch.tensor(
            [tokenizer.encode("def fn(a,b):\n    return a >", add_special_tokens=False)]
        )
        assert not bool(criterion(history, None)[0])
        save(
            out / "result.json",
            {
                "utc": datetime.now(timezone.utc).isoformat(),
                "status": "passed",
                "block_path_cases": len(outcomes),
                "paired_cached_self_blocks": 14,
                "restoration_blocks": 14,
                "scope_and_cache_refusals": refuses,
                "pinned_newline_cases": newline_cases,
                "old_prompt_newline_not_new_stop": True,
                "all_receipts_JSON_roundtrip": True,
                "actual_random_model_forward_calls": sum(c["calls"] for c in counters),
                "actual_input_token_exposures": sum(
                    c["input_token_exposures"] for c in counters
                ),
                "random_generated_tokens": random_tokens,
                "pretrained_model_forwards": 0,
                "backward_calls": 0,
                "optimizer_steps": 0,
                "seconds": time.monotonic() - start,
            },
        )
    except BaseException:
        save(
            out / "failure.json",
            {
                "utc": datetime.now(timezone.utc).isoformat(),
                "traceback": traceback.format_exc(),
                "completed_cases": len(outcomes),
                "closed_dtype_counters": counters,
                "current_counter": meter.value() if meter else None,
                "pretrained_model_forwards": 0,
                "optimizer_steps": 0,
            },
        )
        raise


if __name__ == "__main__":
    main()
