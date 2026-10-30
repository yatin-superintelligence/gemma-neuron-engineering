"""Fixture checks for the published donor-composition interface."""
import unittest
import torch
from tests import test_selector_site as fixtures
from interpretability.selector_site import capture_selector
from interpretability.native_group_policy import native_group_policy


class NativeGroupPolicyTests(unittest.TestCase):
    def test_context_restoration_and_self_rescue(self):
        fixture=fixtures.SelectorSiteTests();fixture.setUp()
        model=fixture.model;ids=fixture.ids
        with torch.no_grad(),capture_selector(model,2,range(3),3)as captured:base=model(input_ids=ids,use_cache=True).logits.clone()
        donors={k:captured for k in['recipient','samepolicy','opposite','background_self']}
        coordinates=[{'flat':2,'layer':2,'channel':2}]
        versions={n:p._version for n,p in model.named_parameters()}
        for operation in['none','background_self','samepolicy','opposite','recipient','opposite_then_same']:
            condition={'background':True,'full_donor':'samepolicy','operation':operation}
            with torch.no_grad(),native_group_policy(model,2,condition,donors,coordinates,block=[1,2]):out=model(input_ids=ids,use_cache=True).logits
            self.assertTrue(torch.equal(out,base),operation)
            with torch.no_grad():self.assertTrue(torch.equal(model(input_ids=ids,use_cache=True).logits,base))
        self.assertEqual(versions,{n:p._version for n,p in model.named_parameters()})
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__=='__main__':unittest.main()
