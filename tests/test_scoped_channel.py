"""Actual small Gemma scalar-scope, cache and restoration fixtures."""
import unittest
import torch
from tests import test_selector_site as fixture_module
from interpretability.native_capture import capture_native_mlp
from interpretability.mlp_composition import compose_mlp_donors
from interpretability.scoped_channel import scoped_channel,bit_equal


class ScopedChannelTests(unittest.TestCase):
    def setUp(self):
        fixture=fixture_module.SelectorSiteTests();fixture.setUp();self.model=fixture.model;self.ids=fixture.ids
        self.layers=[0,2,4];self.coords=[{'flat':i*24+1,'layer':i,'channel':1}for i in self.layers]
        self.donors=[]
        other=self.ids.clone();other[0,2]=8
        for ids in[self.ids,other]:
            with capture_native_mlp(self.model,2,self.layers)as donor:
                with torch.no_grad():self.model(ids,use_cache=True)
            self.donors.append(donor)

    def values(self,index):return{c['flat']:float(self.donors[index]['MLP'][c['layer']][0,c['channel']])for c in self.coords}

    def path(self):
        with torch.no_grad():value=self.model(self.ids,use_cache=True)
        values=[value.logits.clone()];cache=value.past_key_values
        for token in[7,8]:
            with torch.no_grad():value=self.model(torch.tensor([[token]]),past_key_values=cache,use_cache=True)
            values.append(value.logits.clone());cache=value.past_key_values
        return values

    def test_scalar_self_and_foreign_cache_scope(self):
        base=self.path()
        for operation,gain,source in[('gain',0,0),('gain',2,0),('swap',1,0),('swap',1,1),('rescue',1,0)]:
            with scoped_channel(self.model,self.coords,2,operation=operation,gain=gain,donor=self.values(source))as receipt:changed=self.path()
            self.assertEqual(all(bit_equal(a,b)for a,b in zip(base,changed)),operation=='rescue'or(operation=='swap'and source==0))
            self.assertEqual(receipt['scope']['prefill_checks'],3);self.assertEqual(receipt['scope']['cached_checks'],6)
            self.assertTrue(all(bit_equal(a,b)for a,b in zip(base,self.path())))

    def test_live_background_and_exception_restoration(self):
        base=self.path();versions={n:p._version for n,p in self.model.named_parameters()}
        with compose_mlp_donors(self.model,2,[{'layers':self.layers,'donor':self.donors[1]}]):
            with scoped_channel(self.model,self.coords,2,operation='swap',donor=self.values(0))as receipt:self.path()
        self.assertTrue(receipt['scope']['all_other_values_bit_exact'])
        with self.assertRaises(RuntimeError):
            with scoped_channel(self.model,self.coords,2,operation='gain',gain=2):raise RuntimeError('deliberate fixture exit')
        self.assertTrue(all(bit_equal(a,b)for a,b in zip(base,self.path())))
        self.assertEqual(versions,{n:p._version for n,p in self.model.named_parameters()})


if __name__=='__main__':unittest.main()
