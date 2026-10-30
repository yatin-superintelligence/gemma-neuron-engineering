"""Small real-architecture fixtures for shared-read interventions."""
import unittest
import torch
from transformers import Gemma4Config,Gemma4TextConfig,Gemma4ForConditionalGeneration
from interpretability.selector_site import capture_selector,selector_site


class SelectorSiteTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1606);torch.set_num_threads(1)
        text=Gemma4TextConfig(vocab_size=32,vocab_size_per_layer_input=32,hidden_size=16,
            intermediate_size=24,num_hidden_layers=6,num_attention_heads=2,num_key_value_heads=2,
            head_dim=8,global_head_dim=8,num_global_key_value_heads=2,num_kv_shared_layers=2,
            hidden_size_per_layer_input=4,use_double_wide_mlp=True,sliding_window=16,max_position_embeddings=64,
            layer_types=['sliding_attention','full_attention']*3,
            rope_parameters={k:{'rope_type':'default','rope_theta':10000.0}for k in ['sliding_attention','full_attention']})
        config=Gemma4Config(text_config=text.to_dict());config.vision_config=None;config.audio_config=None
        self.model=Gemma4ForConditionalGeneration(config).double().eval().requires_grad_(False)
        self.ids=torch.tensor([[2,3,4,5,6]]);other=self.ids.clone();other[0,2]=8
        self.donors=[]
        for ids in [self.ids,other]:
            with capture_selector(self.model,2,range(3),3)as donor:
                with torch.inference_mode():self.model(ids,use_cache=True)
            self.donors.append(donor)

    def test_self_donor_native_cache_and_restoration(self):
        with torch.inference_mode():base=self.model(self.ids,use_cache=True)
        for kind in ['K','V','KV','MLP','both']:
            for identity in [True,False]:
                with self.subTest(kind=kind,identity=identity):
                    with selector_site(self.model,2,self.donors[0 if identity else 1],kind,range(3),3):
                        with capture_selector(self.model,2,range(3),3)as measured:
                            with torch.inference_mode():changed=self.model(self.ids,use_cache=True)
                    self.assertEqual(torch.equal(base.logits,changed.logits),identity)
                    if kind in ['K','V','KV']:
                        self.assertTrue(torch.equal(measured['producer_output'],self.donors[0]['producer_output']))
                        for attr in ['keys','values']:
                            self.assertTrue(torch.equal(getattr(base.past_key_values.layers[3],attr),getattr(changed.past_key_values.layers[3],attr)))
                    with torch.inference_mode():self.assertTrue(torch.equal(base.logits,self.model(self.ids,use_cache=True).logits))

    def test_cached_reapplication(self):
        for kind in ['K','V','KV','MLP','both']:
            with selector_site(self.model,2,self.donors[1],kind,range(3),3)as receipt:
                with torch.inference_mode():self.model.generate(self.ids,max_new_tokens=3,do_sample=False,use_cache=True,eos_token_id=None,pad_token_id=0)
            if kind!='MLP':self.assertEqual(receipt['KV_calls'],3)
            if kind in ['MLP','both']:
                self.assertEqual(receipt['MLP_prefill_hits'],3);self.assertEqual(receipt['MLP_decode_skips'],6)

    def test_single_layer_sites(self):
        with torch.inference_mode():base=self.model(self.ids,use_cache=True).logits
        for layer in range(3):
            for identity in [True,False]:
                with selector_site(self.model,2,self.donors[0 if identity else 1],'MLP',mlp_layers=[layer],producer=3):
                    with torch.inference_mode():changed=self.model(self.ids,use_cache=True).logits
                self.assertEqual(torch.equal(base,changed),identity)
                with torch.inference_mode():self.assertTrue(torch.equal(base,self.model(self.ids,use_cache=True).logits))


if __name__=='__main__':unittest.main()
