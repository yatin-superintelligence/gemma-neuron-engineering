"""Small actual-architecture checks for coordinate and sign reversals."""
import unittest
import torch
from tests.test_selector_site import SelectorSiteTests
from interpretability.mlp_reversal import reverse_native_mlps
from interpretability.stored_neuron_gain import tensor_sha


class ReversalTests(unittest.TestCase):
    def test_aligned_permutation_and_exact_restoration(self):
        fixture = SelectorSiteTests(); fixture.setUp(); model = fixture.model
        before = {n: tensor_sha(p) for n, p in model.named_parameters()}
        with torch.no_grad(): original = model(fixture.ids, use_cache=True).logits
        for operation in ['aligned_channel_reverse', 'uncompensated_channel_reverse', 'sign_flip']:
            with reverse_native_mlps(model, operation) as receipt:
                with torch.no_grad(): changed = model(fixture.ids, use_cache=True).logits
                difference = float((changed - original).abs().max())
                if operation == 'aligned_channel_reverse': self.assertLessEqual(difference, 1e-10)
                else: self.assertGreater(difference, 0)
            self.assertTrue(receipt['all_target_hashes_restored_exact'])
            for target in receipt['targets']:
                self.assertEqual(target['source_sha256'], target['restored_sha256'])
                self.assertEqual(target['edited_version'], target['source_version'] + 1)
                self.assertEqual(target['restored_version'], target['source_version'] + 2)
            self.assertEqual(before, {n: tensor_sha(p) for n, p in model.named_parameters()})

    def test_exception_restoration_and_validation(self):
        fixture = SelectorSiteTests(); fixture.setUp(); model = fixture.model
        before = {n: tensor_sha(p) for n, p in model.named_parameters()}
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with reverse_native_mlps(model, 'sign_flip'): raise RuntimeError('fixture')
        self.assertEqual(before, {n: tensor_sha(p) for n, p in model.named_parameters()})
        with self.assertRaises(ValueError):
            with reverse_native_mlps(model, 'reverse_knowledge'): pass
        model.requires_grad_(True)
        with self.assertRaises(ValueError):
            with reverse_native_mlps(model, 'sign_flip'): pass

if __name__ == '__main__': unittest.main()
