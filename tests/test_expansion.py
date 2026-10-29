"""Independent invariants for additions, gradients, serialization and restoration."""
import io
import unittest
import torch
from torch import nn
from interpretability.expansion import NewGatedNeurons,insert_branch


class ExpansionTests(unittest.TestCase):
    def test_zero_insertion_and_exception_restore(self):
        base=nn.Linear(7,7,bias=False).to(dtype=torch.bfloat16)
        x=torch.randn(1,4,7,dtype=torch.bfloat16);y=base(x).detach().clone()
        for placement in ['parallel','residual']:
            branch=NewGatedNeurons(7,3,normalize=placement=='residual')
            with self.assertRaisesRegex(RuntimeError,'fixture'):
                with insert_branch(base,branch,placement)as audit:
                    self.assertTrue(torch.equal(base(x),y));self.assertEqual(audit['calls'],1)
                    raise RuntimeError('fixture')
            self.assertEqual(len(base._forward_hooks),0)
            self.assertTrue(torch.equal(base(x),y))

    def test_nonzero_branch_matches_independent_formula(self):
        branch=NewGatedNeurons(5,2,seed=12)
        with torch.no_grad():branch.down.copy_(torch.arange(10).reshape(5,2)/50)
        x=torch.randn(2,3,5)
        g=x@branch.gate.T;u=x@branch.up.T
        activation=.5*g*(1+torch.tanh((2/torch.pi)**.5*(g+.044715*g**3)))
        expected=(activation*u)@branch.down.T
        torch.testing.assert_close(branch(x),expected,rtol=2e-6,atol=1e-6)

    def test_measurement_gradients_manual_edit_and_roundtrip(self):
        base=nn.Linear(5,5,bias=False).requires_grad_(False)
        old=base.weight.detach().clone();branch=NewGatedNeurons(5,2,normalize=True)
        x=torch.randn(1,4,5)
        with insert_branch(base,branch,'residual'):
            loss=(base(x)-torch.ones_like(x)).square().mean();loss.backward()
        self.assertTrue(torch.isfinite(branch.down.grad).all())
        self.assertGreater(branch.down.grad.norm().item(),0)
        self.assertIsNone(base.weight.grad)
        # A prescribed edit, independent of gradients; no parameter learning.
        with torch.no_grad():branch.down.copy_(torch.arange(10,dtype=torch.float32).reshape(5,2)/1000)
        self.assertTrue(torch.equal(base.weight,old))
        buffer=io.BytesIO();torch.save(branch.state_dict(),buffer);buffer.seek(0)
        loaded=NewGatedNeurons(5,2,normalize=True);loaded.load_state_dict(torch.load(buffer,weights_only=True))
        self.assertTrue(torch.equal(branch(x),loaded(x)))

    def test_shape_and_placement_fail_closed(self):
        with self.assertRaises(ValueError):NewGatedNeurons(0,2)
        with self.assertRaises(ValueError):NewGatedNeurons(5,2)(torch.zeros(1,3,6))
        with self.assertRaises(ValueError):
            with insert_branch(nn.Identity(),NewGatedNeurons(5,2),'attention'):pass


if __name__=='__main__':unittest.main()
