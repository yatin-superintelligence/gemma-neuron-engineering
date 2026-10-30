"""Multi-layer column scope must restore even when inference raises."""
import unittest
import torch
from tests import test_selector_site as fixtures
from interpretability.stored_group_gain import stored_group_gain,tensor_sha


class StoredGroupGainTests(unittest.TestCase):
    def test_exception_restores_every_target_and_other_parameter(self):
        fixture=fixtures.SelectorSiteTests();fixture.setUp();model=fixture.model
        params=dict(model.named_parameters());before={n:tensor_sha(p)for n,p in params.items()};versions={n:p._version for n,p in params.items()}
        coords=[(0,1),(1,2),(1,3)];targets={id(model.model.language_model.layers[i].mlp.down_proj.weight)for i,c in coords}
        with torch.no_grad():base=model(fixture.ids,use_cache=True).logits
        with self.assertRaisesRegex(RuntimeError,'inference failed'):
            with stored_group_gain(model,coords,2)as receipt:
                self.assertTrue(all(r['other_columns_exact']for r in receipt['layers']))
                raise RuntimeError('inference failed')
        self.assertEqual(before,{n:tensor_sha(p)for n,p in params.items()})
        self.assertTrue(receipt['all_original_matrices_restored'])
        for n,p in params.items():self.assertEqual(p._version,versions[n]+(2 if id(p)in targets else 0))
        with torch.no_grad():restored=model(fixture.ids,use_cache=True).logits
        self.assertTrue(torch.equal(base,restored))

    def test_malformed_groups_stop_before_any_assignment(self):
        fixture=fixtures.SelectorSiteTests();fixture.setUp();model=fixture.model
        params=dict(model.named_parameters());before={n:tensor_sha(p)for n,p in params.items()};versions={n:p._version for n,p in params.items()}
        for coords in [[(0,1),(0,1)],[(0,1),(100,0)],[(0,1),(1,10000)]]:
            with self.assertRaises(ValueError):
                with stored_group_gain(model,coords,2):self.fail('Malformed coordinates accepted')
            self.assertEqual(versions,{n:p._version for n,p in params.items()})
            self.assertEqual(before,{n:tensor_sha(p)for n,p in params.items()})


if __name__=='__main__':unittest.main()
