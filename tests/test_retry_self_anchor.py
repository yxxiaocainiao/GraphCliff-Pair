"""One contract check: exact self endpoint, learnable gain, pair boundaries."""
import unittest
from types import SimpleNamespace
import torch
import graphcliff_pair.model as shared
from graphcliff_pair import train
from experiments.mechanism_retry.model import LayerSwap
from experiments.mechanism_retry.self_anchor import SelfAnchorSwap, combine
from experiments.mechanism_retry.repro import stable_max_pool
from tests.test_mechanism_retry import graphs


class AnchorContract(unittest.TestCase):
    def test_endpoint_gradient_and_boundaries(self):
        torch.set_num_threads(2)
        original = shared.global_max_pool
        deterministic, warn_only = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
        shared.global_max_pool = stable_max_pool
        try:
            devices = [("cpu", 32, 2)]
            if torch.cuda.is_available():
                devices += [("cuda", 32, 2), ("cuda", 256, 3)]
            for device, hidden, layers in devices:
                train.set_seed(SimpleNamespace(seed=42))
                torch.use_deterministic_algorithms(True, warn_only=False)
                baseline = LayerSwap("retry_self", hidden, layers).to(device).eval()
                model = SelfAnchorSwap(hidden, layers).to(device).eval()
                missing = model.load_state_dict(baseline.state_dict(), strict=False)
                self.assertEqual(missing.missing_keys, ["innovation_gain"])
                self.assertFalse(missing.unexpected_keys)
                q, r = graphs(["CCO", "c1ccccc1O"]).to(device), graphs(["N", "CCCC"]).to(device)
                torch.testing.assert_close(model(q, r), baseline(q, r), atol=0, rtol=0)
                count = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in baseline.parameters())
                self.assertEqual(count, layers)
                # Training dropout consumes the same RNG despite the extra zero-dropout attention calls.
                baseline.train()
                model.train()
                cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                expected = baseline(q, r)
                torch.set_rng_state(cpu)
                if cuda:
                    torch.cuda.set_rng_state_all(cuda)
                actual = model(q, r)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                (actual + 1.25).square().mean().backward()
                self.assertTrue(torch.isfinite(model.innovation_gain.grad).all())
                self.assertGreater(float(model.innovation_gain.grad.abs().sum()), 0)
                model.eval()
                with torch.no_grad():
                    model.innovation_gain.fill_(.3)
                result = model(q, r)
                torch.testing.assert_close(result, -model(r, q), atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(model(q, q), torch.zeros_like(result), atol=1e-6, rtol=0)
                restored = SelfAnchorSwap(hidden, layers).to(device).eval()
                restored.load_state_dict(model.state_dict())
                torch.testing.assert_close(result, restored(q, r), atol=0, rtol=0)
                # Change only the other pair at the same padded shapes: GPU TF32 kernels
                # may round differently when an entire batch is resized.
                changed_q = graphs(["CCO", "c1ccccc1N"]).to(device)
                changed_r = graphs(["N", "CCCN"]).to(device)
                torch.testing.assert_close(result[:1], model(changed_q, changed_r)[:1], atol=1e-6, rtol=1e-5)
                if device == "cpu":
                    torch.testing.assert_close(result[:1], model(graphs(["CCO"]), graphs(["N"])), atol=1e-6, rtol=1e-5)
                with self.assertRaisesRegex(ValueError, "unequal"):
                    model(q, graphs(["N"]).to(device))
                u, v = torch.randn(5, hidden, device=device), torch.randn(5, hidden, device=device)
                for gain in (-3., 0., 3.):
                    coefficient = torch.tensor(gain, device=device, requires_grad=True)
                    correction = combine(u, v, coefficient) - u
                    self.assertLessEqual(float(correction.norm()), float((v-u).norm()) + 1e-5)
                    torch.testing.assert_close(combine(u, u, coefficient), u, atol=0, rtol=0)
                print(f"PASS anchor contracts {device} H={hidden} layers={layers} parameters={sum(p.numel() for p in model.parameters())}", flush=True)
        finally:
            shared.global_max_pool = original
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


if __name__ == "__main__":
    unittest.main()
