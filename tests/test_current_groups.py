"""Actual Gemma current-token scope, conditional identity and cache proofs."""
import copy
import json
import unittest

import numpy as np
import torch
from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig
from transformers.cache_utils import DynamicCache

from interpretability.cache_proofs import check_cache
from interpretability.decode_step import capture_decode_step
from interpretability.scoped_channel import bit_equal
from research.experiments.n19.trajectory import cache_witness, continue_tail, current, prepare
from interpretability.current_group_audit import check_group_receipt
from research.experiments.n20.operations import policy


class CurrentGroupTests(unittest.TestCase):
    observations = []
    audited_calls = []

    @staticmethod
    def model(dtype):
        torch.manual_seed(3420)
        torch.set_num_threads(1)
        text = Gemma4TextConfig(
            vocab_size=32, vocab_size_per_layer_input=32, hidden_size=16,
            intermediate_size=24, num_hidden_layers=6, num_attention_heads=2,
            num_key_value_heads=2, head_dim=8, global_head_dim=8,
            num_global_key_value_heads=2, num_kv_shared_layers=2,
            hidden_size_per_layer_input=4, use_double_wide_mlp=True,
            sliding_window=16, max_position_embeddings=64,
            layer_types=["sliding_attention", "full_attention"] * 3,
            rope_parameters={k: {"rope_type": "default", "rope_theta": 10000.0}
                             for k in ["sliding_attention", "full_attention"]})
        config = Gemma4Config(text_config=text.to_dict())
        config.vision_config = config.audio_config = None
        model = Gemma4ForConditionalGeneration(config).to(dtype).eval().requires_grad_(False)
        def count(module, arguments, kwargs):
            ids = kwargs.get("input_ids", arguments[0] if arguments else None)
            CurrentGroupTests.audited_calls.append(ids.numel())
        model.register_forward_pre_hook(count, with_kwargs=True)
        return model

    def test_actual_current_native_and_conditional_paths(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = self.model(dtype)
                widths = [layer.mlp.down_proj.in_features for layer in model.model.language_model.layers]
                offsets = np.cumsum([0] + widths).tolist()
                full = [{"flat": offsets[i] + j, "layer": i, "channel": j}
                        for i, width in enumerate(widths) for j in range(width)]
                group = [{"flat": offsets[i] + j, "layer": i, "channel": j}
                         for i, j in ((4, 1), (4, 3), (5, 2), (5, 4))]
                sources = [{"input_ids": torch.tensor([ids])} for ids in
                           ([2, 3, 4, 5], [2, 7, 4, 5], [2, 3, 7, 5])]
                prefix = [9, 8]
                versions = {n: p._version for n, p in model.named_parameters()}
                hooks = {n: (len(m._forward_pre_hooks), len(m._forward_hooks)) for n, m in model.named_modules()}
                donors, logits, caches = [], [], []
                forwards = []
                handle = model.register_forward_pre_hook(lambda m, a, k: forwards.append(k["input_ids"].numel()), with_kwargs=True)
                for source in sources:
                    cache = prepare(model, source, prefix)
                    with capture_decode_step(model, range(6)) as observed:
                        with torch.no_grad():
                            value = current(model, source, prefix, cache)
                    donors.append(np.concatenate([observed["values"][i].float().flatten().numpy() for i in range(6)]))
                    logits.append(value.logits.clone())
                    caches.append(cache_witness(value.past_key_values))
                self.assertEqual(len(DynamicCache(config=model.config.text_config).layers), 4)
                for cache in caches:
                    check_cache(cache, 4, str(dtype))
                parent = {}
                cases = [("self", None), ("opposite", None), ("samepolicy", None),
                         ("opposite", group), ("samepolicy", group), ("zero", group),
                         ("gain2", group), ("rescue", group), ("conditional_self", group),
                         ("conditional_cut", group), ("opposite", group[:1]), ("rescue", group[:1])]
                for operation, coords in cases:
                    cache = prepare(model, sources[0], prefix)
                    with policy(model, operation, coords, *donors, full) as receipt:
                        with torch.no_grad():
                            value = current(model, sources[0], prefix, cache)
                    actual_cache = cache_witness(value.past_key_values)
                    if coords is None:
                        parent[operation] = (value.logits.clone(), actual_cache)
                        if operation == "self":
                            self.assertTrue(bit_equal(value.logits, logits[0]))
                            self.assertEqual(actual_cache, caches[0])
                    else:
                        expected_cache = parent["opposite"][1] if operation.startswith("conditional") else caches[0]
                        self.assertEqual(actual_cache, expected_cache)
                        if operation == "conditional_self":
                            self.assertTrue(bit_equal(value.logits, parent["opposite"][0]))
                        if operation == "rescue":
                            self.assertTrue(bit_equal(value.logits, logits[0]))
                        if dtype == torch.bfloat16:
                            source = donors[1] if operation in {"opposite", "conditional_self"} else donors[2] if operation == "samepolicy" else donors[0]
                            actual_receipt = json.loads(json.dumps(receipt["group"]))
                            check_group_receipt(actual_receipt, coords, operation,
                                                [float(source[c["flat"]]) for c in coords],
                                                [float(donors[1][c["flat"]]) for c in coords] if operation.startswith("conditional") else None)
                            bad = copy.deepcopy(actual_receipt)
                            bad["scope"][0]["target_after_values"][0] += 1
                            with self.assertRaises(AssertionError):
                                check_group_receipt(bad, coords, operation, [float(source[c["flat"]]) for c in coords])
                    with torch.no_grad():
                        restored = current(model, sources[0], prefix, prepare(model, sources[0], prefix))
                    self.assertTrue(bit_equal(restored.logits, logits[0]))
                    self.assertEqual(cache_witness(restored.past_key_values), caches[0])
                    self.observations.append({"dtype": str(dtype), "operation": operation,
                                              "coordinates": len(full) if coords is None else len(coords),
                                              "restoration_exact": True, "cache_guard_passed": True})
                for misuse in ("prefill", "second_call"):
                    cache = prepare(model, sources[0], prefix)
                    with self.assertRaises(ValueError):
                        with policy(model, "zero", group, *donors, full):
                            with torch.no_grad():
                                if misuse == "prefill":
                                    model(**sources[0], use_cache=True)
                                else:
                                    value = current(model, sources[0], prefix, cache)
                                    current(model, sources[0], prefix, value.past_key_values)
                handle.remove()
                self.assertEqual(versions, {n: p._version for n, p in model.named_parameters()})
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
                self.assertEqual(hooks, {n: (len(m._forward_pre_hooks), len(m._forward_hooks)) for n, m in model.named_modules()})
                self.observations.append({"dtype": str(dtype), "forward_calls": len(forwards),
                                          "input_token_exposures": sum(forwards), "backward_calls": 0,
                                          "optimizer_steps": 0, "all_parameters_and_hooks_restore": True})

    def test_malformed_cache_proofs_are_rejected(self):
        model = self.model(torch.bfloat16)
        with torch.no_grad():
            value = model(input_ids=torch.tensor([[2, 3, 4]]), use_cache=True)
        witness = cache_witness(value.past_key_values)
        check_cache(witness, 4)
        variants = []
        short = copy.deepcopy(witness)
        short["layers"] = short["layers"][:-1]
        variants.append(short)
        extra = copy.deepcopy(witness)
        extra["layers"].append(copy.deepcopy(extra["layers"][-1]))
        variants.append(extra)
        dtype = copy.deepcopy(witness)
        dtype["layers"][0]["keys"]["dtype"] = "torch.float32"
        variants.append(dtype)
        shape = copy.deepcopy(witness)
        shape["layers"][0]["keys"]["shape"][0] = 0
        variants.append(shape)
        missing = copy.deepcopy(witness)
        missing["layers"][0]["values"] = None
        variants.append(missing)
        import hashlib
        for altered in variants:
            # Recompute the outer checksum so semantic guards must reject it.
            altered["sha256"] = hashlib.sha256(json.dumps(altered["layers"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            with self.assertRaises(AssertionError):
                check_cache(altered, 4)

    def test_cached_tail_counters_and_hook_removal(self):
        model = self.model(torch.bfloat16)
        encoded = {"input_ids": torch.tensor([[2, 3, 4, 5]])}
        prefix = [9, 8]
        widths = [layer.mlp.down_proj.in_features for layer in model.model.language_model.layers]
        offsets = np.cumsum([0] + widths).tolist()
        group = [{"flat": offsets[4] + 1, "layer": 4, "channel": 1}]
        full = [{"flat": offsets[i] + j, "layer": i, "channel": j}
                for i, width in enumerate(widths) for j in range(width)]
        actual = []
        handle = model.register_forward_pre_hook(lambda m, a, k: actual.append(k["input_ids"].numel()), with_kwargs=True)
        for end_at_first in (False, True):
            actual.clear()
            cache = prepare(model, encoded, prefix)
            with policy(model, "zero", group, None, None, None, full):
                with torch.no_grad():
                    value = current(model, encoded, prefix, cache)
            # No edit hook remains during the following autoregressive steps.
            hooks = sum(len(layer.mlp.down_proj._forward_pre_hooks) for layer in model.model.language_model.layers)
            self.assertEqual(hooks, 0)
            eos = {int(value.logits[0, -1].argmax())} if end_at_first else set()
            tail = continue_tail(model, encoded, prefix, value, eos, cap=9)
            self.assertEqual(len(tail), 1 if end_at_first else 7)
            self.assertEqual(len(actual), len(prefix) + len(tail))
            self.assertEqual(sum(actual), 4 + len(prefix) + len(tail) - 1)
            self.observations.append({"counted_trajectory": "first_EOS" if end_at_first else "cap",
                                      "forward_calls": len(actual), "input_token_exposures": sum(actual),
                                      "tail_tokens": len(tail), "late_hooks_removed_before_tail": True,
                                      "optimizer_steps": 0})
        handle.remove()


if __name__ == "__main__":
    unittest.main()
