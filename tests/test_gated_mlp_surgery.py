"""Engineering invariants on the actual Gemma MLP class; no model weights/GPU."""
import unittest
import torch
from transformers import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextMLP
from interpretability.gated_mlp_surgery import (
    steer_channels, delete_channels, duplicate_channels, add_channels, edit_output_vectors,
)


class SurgeryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(196)
        config = Gemma4TextConfig(hidden_size=8, intermediate_size=12, num_hidden_layers=2,
                                 num_kv_shared_layers=0, use_double_wide_mlp=False,
                                 layer_types=['full_attention', 'full_attention'])
        self.mlp = Gemma4TextMLP(config, 0).double().eval()
        self.x = torch.randn(2, 5, 8, dtype=torch.float64)
        self.reference = self.mlp(self.x).detach()
        self.original = {k: v.detach().clone() for k, v in self.mlp.state_dict().items()}

    def contribution(self, j):
        gate = self.x @ self.mlp.gate_proj.weight[j]
        up = self.x @ self.mlp.up_proj.weight[j]
        return self.mlp.act_fn(gate).mul(up).unsqueeze(-1) * self.mlp.down_proj.weight[:, j]

    def close(self, a, b):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)

    def tearDown(self):
        for key, value in self.mlp.state_dict().items():
            self.assertTrue(torch.equal(value, self.original[key]))
        self.assertTrue(torch.equal(self.mlp(self.x), self.reference))

    def test_gain_and_exception_restore(self):
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with steer_channels(self.mlp, [3], gain=2) as audit:
                self.close(self.mlp(self.x), self.reference + self.contribution(3))
                self.assertGreater(audit['calls'], 0)
                raise RuntimeError('fixture')

    def test_identity_is_bitwise(self):
        with steer_channels(self.mlp, [1, 7]) as audit:
            self.assertTrue(torch.equal(self.mlp(self.x), self.reference))
            self.assertTrue(audit['identity'])
            self.assertEqual(audit['calls'], 1)

    def test_physical_delete_matches_independent_channel_subtraction(self):
        edited = delete_channels(self.mlp, [2, 5])
        self.assertEqual(edited.gate_proj.out_features, 10)
        self.assertEqual(edited.down_proj.in_features, 10)
        self.close(edited(self.x), self.reference - self.contribution(2) - self.contribution(5))

    def test_split_duplication_preserves_function(self):
        edited = duplicate_channels(self.mlp, [1, 1, 8], preserve_function=True)
        self.assertEqual(edited.intermediate_size, 15)
        self.close(edited(self.x), self.reference)

    def test_unsplit_duplication_is_amplification(self):
        edited = duplicate_channels(self.mlp, [4], preserve_function=False)
        self.close(edited(self.x), self.reference + self.contribution(4))

    def test_new_zero_readout_channels_preserve_function(self):
        edited = add_channels(self.mlp, torch.randn(3, 8), torch.randn(3, 8), torch.zeros(8, 3))
        self.assertEqual(edited.intermediate_size, 15)
        self.close(edited(self.x), self.reference)

    def test_output_vector_edit(self):
        delta = torch.arange(8, dtype=torch.float64).reshape(8, 1) / 100
        edited = edit_output_vectors(self.mlp, [6], delta)
        h = self.mlp.act_fn(self.x @ self.mlp.gate_proj.weight[6]) * (self.x @ self.mlp.up_proj.weight[6])
        self.close(edited(self.x), self.reference + h.unsqueeze(-1) * delta[:, 0])

    def test_last_token_scope_and_offset(self):
        with steer_channels(self.mlp, [7], offset=.25, token_scope='last'):
            expected = self.reference.clone()
            expected[:, -1] += .25 * self.mlp.down_proj.weight[:, 7]
            self.close(self.mlp(self.x), expected)

    def test_invalid_structures_refused(self):
        for ids in [[], [12], [-1], [1, 1]]:
            with self.assertRaises(ValueError): delete_channels(self.mlp, ids)
        with self.assertRaises(ValueError): delete_channels(self.mlp, list(range(12)))
        with self.assertRaises(ValueError): add_channels(self.mlp, torch.ones(2, 8), torch.ones(1, 8), torch.ones(8, 2))
        with self.assertRaises(ValueError): edit_output_vectors(self.mlp, [1], torch.full((8, 1), float('nan')))

    def test_packed_integer_weight_storage_refused(self):
        packed = duplicate_channels(self.mlp, [0])
        packed.gate_proj.weight = torch.nn.Parameter(torch.zeros_like(packed.gate_proj.weight, dtype=torch.uint8),
                                                     requires_grad=False)
        with self.assertRaisesRegex(ValueError, 'unpacked floating-point'):
            delete_channels(packed, [0])


if __name__ == '__main__': unittest.main()
