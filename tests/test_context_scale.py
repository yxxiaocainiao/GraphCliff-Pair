import unittest
import torch
from experiments.mechanism_retry.context_scale import ContextScaleSwap, contextual_difference
from experiments.mechanism_retry.model import LayerSwap
from graphcliff_pair.model import parameters
from graphcliff_pair.train import state_hash
from test_mechanism_retry import graphs


class ContextContracts(unittest.TestCase):
    def test_same_parameters_identity_isolation_gradient_and_reload(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        old = LayerSwap("retry_centered", 32, 2)
        torch.manual_seed(42)
        model = ContextScaleSwap(32, 2).eval()
        self.assertEqual(parameters(model), parameters(old))
        self.assertEqual(state_hash(model.state_dict()), state_hash(old.state_dict()))
        q, r = graphs(["CCO", "c1ccccc1O"]), graphs(["N", "CCCC"])
        result = model(q, r)
        torch.testing.assert_close(result, -model(r, q), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(model(q, q), torch.zeros_like(result), atol=0, rtol=0)
        torch.testing.assert_close(result[:1], model(graphs(["CCO"]), graphs(["N"])), atol=1e-6, rtol=1e-5)
        result.square().mean().backward()
        self.assertGreater(model.interactions[0].attention.in_proj_weight.grad.abs().sum(), 0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        restored = ContextScaleSwap(32, 2).eval()
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored(q, r), result.detach(), atol=0, rtol=0)
        long = model.encoder.layers[0].long
        u, v = torch.randn(3, 32, requires_grad=True), torch.randn(3, 32, requires_grad=True)
        contextual_difference(long, u, v).square().mean().backward()
        self.assertTrue(torch.isfinite(u.grad).all() and torch.isfinite(v.grad).all())
        self.assertGreater(u.grad.abs().sum(), 0)


if __name__ == "__main__":
    unittest.main()
