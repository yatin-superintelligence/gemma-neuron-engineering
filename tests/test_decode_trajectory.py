"""Greedy-prefix lineage, EOS and current-step cache invariants."""
import unittest
import torch
from tests import test_selector_site as fixture_module
from interpretability.decode_step import capture_decode_step,edit_decode_step
from interpretability.scoped_channel import bit_equal
from research.experiments.n19.trajectory import prepare,current,continue_tail,cache_witness


class DecodeTrajectoryTests(unittest.TestCase):
    def setUp(self):
        f=fixture_module.SelectorSiteTests();f.setUp();self.model=f.model.to(torch.bfloat16);self.encoded={'input_ids':f.ids[:,:3]}

    def test_native_greedy_prefix_and_EOS(self):
        model=self.model;encoded=self.encoded
        with torch.no_grad():native=model.generate(**encoded,max_new_tokens=8,eos_token_id=None,pad_token_id=0,do_sample=False)[0,3:].tolist()
        for k in[1,2,4]:
            cache=prepare(model,encoded,native[:k])
            with torch.no_grad():value=current(model,encoded,native[:k],cache)
            self.assertEqual(native[:k]+continue_tail(model,encoded,native[:k],value,set(),cap=8),native)
        cache=prepare(model,encoded,native[:1])
        with torch.no_grad():value=current(model,encoded,native[:1],cache)
        eos=int(value.logits[0,-1].argmax())
        self.assertEqual(continue_tail(model,encoded,native[:1],value,{eos},cap=8),[eos])

    def test_full_self_and_measurement_preserve_entire_cache(self):
        model=self.model;encoded=self.encoded;prefix=[5,6];indices=list(range(6))
        cache=prepare(model,encoded,prefix)
        with capture_decode_step(model,indices)as native:
            with torch.no_grad():original=current(model,encoded,prefix,cache)
        logits=original.logits.detach().clone();cache_hash=cache_witness(original.past_key_values)
        coords=[{'flat':layer*10000+j,'layer':layer,'channel':j}for layer in indices for j in range(model.model.language_model.layers[layer].mlp.down_proj.in_features)]
        donor={c['flat']:float(native['values'][c['layer']][0,0,c['channel']])for c in coords};cache=prepare(model,encoded,prefix)
        with edit_decode_step(model,coords,operation='swap',donor=donor):
            with torch.no_grad():value=current(model,encoded,prefix,cache)
        self.assertTrue(bit_equal(value.logits,logits));self.assertEqual(cache_witness(value.past_key_values),cache_hash)
        cache=prepare(model,encoded,prefix)
        with capture_decode_step(model,indices,measurement_gradients=True)as captured:
            with torch.enable_grad():value=current(model,encoded,prefix,cache)
            torch.autograd.grad(value.logits[0,-1,3]-value.logits[0,-1,7],[captured['graph'][i]for i in indices])
            self.assertTrue(bit_equal(value.logits,logits));self.assertEqual(cache_witness(value.past_key_values),cache_hash)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))


if __name__=='__main__':unittest.main()
