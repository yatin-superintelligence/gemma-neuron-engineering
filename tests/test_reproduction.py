"""Public bundle integrity and independent edit identities on small fixtures."""
from pathlib import Path
import json
import tempfile
import unittest
import torch
from torch import nn
from reproduce.model_edits import inspect_bundle, apply_physical_edit, file_sha256, projection_hashes
from interpretability.gated_mlp_surgery import DenseGatedMLP


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(196)
        self.mlp = DenseGatedMLP(torch.randn(7, 5, dtype=torch.float64),
             torch.randn(7, 5, dtype=torch.float64), torch.randn(5, 7, dtype=torch.float64), nn.GELU())
        self.x = torch.randn(2, 3, 5, dtype=torch.float64)
        self.basis = {'gate_target': self.mlp.gate_proj.weight[2].detach().clone(),
                      'up_partner': self.mlp.up_proj.weight[4].detach().clone(),
                      'down_target': self.mlp.down_proj.weight[:, 2].detach().clone()}

    def test_released_vectors_define_the_new_computation(self):
        model = apply_physical_edit(self.mlp, 'dense_recombined', 2, 4, self.basis)
        new_h = torch.nn.functional.gelu(self.x @ self.basis['gate_target']) * (self.x @ self.basis['up_partner'])
        reference = self.mlp(self.x) + new_h.unsqueeze(-1) * self.basis['down_target'] / 2
        torch.testing.assert_close(model(self.x), reference, atol=1e-12, rtol=1e-12)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_delete_scale_clone_and_zero_match_independent_identities(self):
        before = projection_hashes(self.mlp)
        h = self.mlp.act_fn(self.x @ self.mlp.gate_proj.weight[2]) * (self.x @ self.mlp.up_proj.weight[2])
        contribution = h.unsqueeze(-1) * self.mlp.down_proj.weight[:, 2]
        expected = {'dense_delete': self.mlp(self.x) - contribution,
                    'dense_scale2': self.mlp(self.x) + contribution,
                    'dense_clone_unsplit': self.mlp(self.x) + contribution,
                    'dense_clone_split': self.mlp(self.x), 'dense_new_zero': self.mlp(self.x)}
        for edit, reference in expected.items():
            actual = apply_physical_edit(self.mlp, edit, 2, 4, self.basis)
            torch.testing.assert_close(actual(self.x), reference, atol=1e-12, rtol=1e-12)
        self.assertEqual(before, projection_hashes(self.mlp))

    def test_mismatched_basis_or_unknown_edit_is_refused(self):
        bad = dict(self.basis)
        bad['down_target'] = bad['down_target'] + 1
        with self.assertRaises(ValueError):
            apply_physical_edit(self.mlp, 'dense_scale2', 2, 4, bad)
        with self.assertRaises(ValueError):
            apply_physical_edit(self.mlp, 'unknown', 2, 4, self.basis)

    def test_bundle_file_hashes_and_path_scope(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'data.json').write_text('{}')
            manifest = {'schema_version': 1, 'optimizer_steps': 0,
                        'files': {'data.json': file_sha256(root / 'data.json')}}
            (root / 'bundle.json').write_text(json.dumps(manifest))
            self.assertEqual(inspect_bundle(root), manifest)
            (root / 'data.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                inspect_bundle(root)
            manifest['files'] = {'../outside': '0' * 64}
            (root / 'bundle.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                inspect_bundle(root)


if __name__ == '__main__':
    unittest.main()
