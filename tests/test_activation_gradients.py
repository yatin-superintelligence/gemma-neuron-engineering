"""Independent finite differences for frozen-model activation derivatives."""
import unittest
import torch
from tests import test_selector_site as fixtures
from interpretability.activation_gradients import native_gradient_tap


class ActivationGradientTests(unittest.TestCase):
    def test_derivatives_and_parameter_invariance(self):
        fixture=fixtures.SelectorSiteTests();fixture.setUp();model=fixture.model;ids=fixture.ids
        versions={n:p._version for n,p in model.named_parameters()}
        with torch.no_grad():baseline=model(ids,use_cache=False).logits
        with native_gradient_tap(model,[1,2])as activations:
            logits=model(ids,use_cache=False).logits
            self.assertTrue(torch.equal(baseline,logits))
            objective=logits[0,-1,7]-logits[0,-1,11];objective.backward()
        for layer in [1,2]:
            gradient=activations[layer].grad[0,2].detach();channel=int(gradient.abs().argmax());expected=float(gradient[channel])
            self.assertNotEqual(expected,0)
            values=[];eps=1e-4
            for sign in [-1,1]:
                def shift(module,args):
                    changed=args[0].clone();changed[0,2,channel]+=sign*eps;return(changed,*args[1:])
                handle=model.model.language_model.layers[layer].mlp.down_proj.register_forward_pre_hook(shift)
                try:
                    with torch.no_grad():output=model(ids,use_cache=False).logits;values.append(float(output[0,-1,7]-output[0,-1,11]))
                finally:handle.remove()
            self.assertAlmostEqual((values[1]-values[0])/(2*eps),expected,delta=max(1e-8,abs(expected)*1e-4))
        self.assertEqual(versions,{n:p._version for n,p in model.named_parameters()})
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))

    def test_trainable_parameters_refused(self):
        fixture=fixtures.SelectorSiteTests();fixture.setUp();model=fixture.model
        next(model.parameters()).requires_grad_(True)
        with self.assertRaises(ValueError):
            with native_gradient_tap(model,[1,2]):pass

    def test_full_program_alignment_and_frozen_backward(self):
        fixture=fixtures.SelectorSiteTests();fixture.setUp();model=fixture.model
        program=torch.tensor([[7,11,10]])
        ids=torch.cat([fixture.ids,program[:,:-1]],dim=1)
        with torch.no_grad():ordinary=model(ids,use_cache=False).logits[:,-3:]
        with native_gradient_tap(model,[1,2])as activations:
            measured=model(ids,use_cache=False,logits_to_keep=3).logits
            self.assertTrue(torch.equal(ordinary,measured))
            lp=measured.float().log_softmax(-1).gather(-1,program[:,:,None]).mean()
            independent=sum(measured[0,i].float().log_softmax(-1)[program[0,i]]for i in range(3))/3
            self.assertTrue(torch.equal(lp,independent))
            lp.backward()
        self.assertTrue(all(torch.isfinite(v.grad).all()for v in activations.values()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))


if __name__=='__main__':unittest.main()
