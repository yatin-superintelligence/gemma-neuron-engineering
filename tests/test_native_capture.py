"""Check native-only capture against an existing actual-Gemma fixture."""
import unittest
import torch
from tests import test_selector_site as fixtures
from interpretability.native_capture import capture_native_mlp


class NativeCaptureTests(unittest.TestCase):
    def test_values_and_attention_hook_scope(self):
        fixture=fixtures.SelectorSiteTests();fixture.setUp();model=fixture.model
        layers=model.model.language_model.layers
        attention_counts=[len(layer.self_attn._forward_hooks)for layer in layers]
        versions={n:p._version for n,p in model.named_parameters()}
        with torch.no_grad():base=model(fixture.ids,use_cache=True).logits
        with capture_native_mlp(model,2,range(3))as capture:
            self.assertEqual(attention_counts,[len(layer.self_attn._forward_hooks)for layer in layers])
            with torch.no_grad():actual=model(fixture.ids,use_cache=True).logits
        self.assertTrue(torch.equal(actual,base));self.assertEqual(set(capture),{'MLP'})
        for index,value in capture['MLP'].items():self.assertTrue(torch.equal(value,fixture.donors[0]['MLP'][index]))
        self.assertTrue(all(not layer.mlp.down_proj._forward_pre_hooks for layer in layers))
        self.assertEqual(versions,{n:p._version for n,p in model.named_parameters()})

if __name__=='__main__':unittest.main()
