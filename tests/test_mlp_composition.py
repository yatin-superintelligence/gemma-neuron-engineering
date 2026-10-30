"""Actual small Gemma checks of disjoint background and rescue hooks."""
import unittest
import torch
from tests import test_selector_site as fixtures
from interpretability.selector_site import capture_selector
from interpretability.mlp_composition import compose_mlp_donors


class MLPCompositionTests(unittest.TestCase):
    def setUp(self):
        fixture=fixtures.SelectorSiteTests();fixture.setUp()
        self.model=fixture.model;self.ids=fixture.ids;self.donors=fixture.donors
        self.background=[{'layers':[0],'donor':self.donors[1]}]
        with torch.inference_mode():self.base=self.model(self.ids,use_cache=True).logits
        with compose_mlp_donors(self.model,2,self.background):
            with capture_selector(self.model,2,range(3),3)as donor:
                with torch.inference_mode():self.changed=self.model(self.ids,use_cache=True).logits
        self.background_donor=donor
        self.assertFalse(torch.equal(self.base,self.changed))

    def test_conditional_self_and_restoration(self):
        assignments=self.background+[{'layers':[1,2],'donor':self.background_donor}]
        with compose_mlp_donors(self.model,2,assignments)as receipt:
            with torch.inference_mode():actual=self.model(self.ids,use_cache=True).logits
        self.assertTrue(torch.equal(actual,self.changed))
        self.assertEqual([p['receipt']['MLP_prefill_hits']for p in receipt['parts']],[1,2])
        with torch.inference_mode():self.assertTrue(torch.equal(self.base,self.model(self.ids,use_cache=True).logits))

    def test_cached_composition(self):
        assignments=self.background+[{'layers':[1,2],'donor':self.donors[0]}]
        with compose_mlp_donors(self.model,2,assignments)as receipt:
            with torch.inference_mode():self.model.generate(self.ids,max_new_tokens=3,do_sample=False,use_cache=True,eos_token_id=None,pad_token_id=0)
        self.assertEqual([p['receipt']['MLP_prefill_hits']for p in receipt['parts']],[1,2])
        self.assertEqual([p['receipt']['MLP_decode_skips']for p in receipt['parts']],[2,4])
        self.assertTrue(all(p['receipt']['native_parameter_versions_unchanged']for p in receipt['parts']))

    def test_overlap_rejected_before_execution(self):
        with self.assertRaises(ValueError):
            with compose_mlp_donors(self.model,2,self.background+[{'layers':[0,1],'donor':self.donors[0]}]):pass
        with torch.inference_mode():self.assertTrue(torch.equal(self.base,self.model(self.ids,use_cache=True).logits))


if __name__=='__main__':unittest.main()
