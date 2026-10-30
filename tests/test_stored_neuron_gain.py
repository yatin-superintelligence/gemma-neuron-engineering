"""Actual architecture tests of explicit column assignments and restoration."""
import unittest
import torch
from tests import test_selector_site as fixtures
from interpretability.stored_neuron_gain import stored_neuron_gain,tensor_sha
from interpretability.anchor_intervention import anchor_intervention


class StoredGainTests(unittest.TestCase):
    def test_exact_edit_scope_restoration_and_identity(self):
        fixture=fixtures.SelectorSiteTests();fixture.setUp();model=fixture.model
        target=model.model.language_model.layers[1].mlp.down_proj.weight;original=target.detach().clone()
        with torch.no_grad():base=model(fixture.ids,use_cache=True).logits
        for factor in [0,1,2,'zero_then_original']:
            before=target._version
            with stored_neuron_gain(model,1,3,factor)as receipt:
                expected=original[:,3]if factor=='zero_then_original'else original[:,3]*factor
                self.assertTrue(torch.equal(target[:,3],expected))
                with torch.no_grad():changed=model(fixture.ids,use_cache=True).logits
                self.assertEqual(torch.equal(base,changed),factor in[1,'zero_then_original'])
            self.assertTrue(torch.equal(target,original));self.assertTrue(receipt['original_matrix_restored_exact'])
            self.assertEqual(target._version-before,3 if factor=='zero_then_original'else 2)
            self.assertEqual(receipt['source_matrix_sha256'],receipt['restored_matrix_sha256'])

    def test_global_hook_comparison_measured(self):
        fixture=fixtures.SelectorSiteTests();fixture.setUp();model=fixture.model;coord=[{'flat':27,'layer':1,'channel':3}]
        for factor in[0,2]:
            with stored_neuron_gain(model,1,3,factor):
                with torch.no_grad():weight_result=model(fixture.ids,use_cache=True).logits
            with anchor_intervention(model,coord,2,gain=factor,scope='all'):
                with torch.no_grad():hook_result=model(fixture.ids,use_cache=True).logits
            self.assertTrue(torch.equal(weight_result,hook_result))
        with self.assertRaises(ValueError):
            with stored_neuron_gain(model,1,3,1.5):pass


if __name__=='__main__':unittest.main()
