import unittest
import torch
from torch_geometric.data import Batch, Data
from graphcliff_pair.vendor.dataset_utils import smiles_to_graph
from graphcliff_pair.fingerprint import membership
from graphcliff_pair.vendor.model import GraphCliffRegressor
from graphcliff_pair.model import PairRegressor, parameters
from experiments.mechanism_retry.model import ResidualFP, LayerSwap, LayerInteraction, SingleBaseline


def graphs(smiles):
    values = [smiles_to_graph(s, 0) for s in smiles]
    return Batch.from_data_list([Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr,
                                     atom_fp=membership(g)) for g in values])


class RetryContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.base = GraphCliffRegressor(38, 13, hidden_size=32, num_layers=2, dropout=0).state_dict()
        self.q, self.r = graphs(["CCO", "c1ccccc1O"]), graphs(["N", "CCCC"])

    def test_fp_exact_initial_baseline_freeze_and_training(self):
        baseline = PairRegressor(hidden_size=32, num_layers=2).eval()
        baseline.load_shared(self.base)
        expected = baseline.head(baseline.pool(baseline.encode(self.q), self.q))
        for space in ("feature", "output"):
            model = ResidualFP(32, 2, space, frozen=True).eval()
            model.load_base(self.base)
            torch.testing.assert_close(model(self.q), expected, atol=0, rtol=0)
            original = {k:v.clone() for k,v in model.state_dict().items() if not k.startswith(("fp.", "correction.", "alpha"))}
            model.train()
            self.assertFalse(model.encoder.training)
            optim = torch.optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=.01)
            optim.zero_grad()
            (model(self.q) - 2).square().mean().backward()
            p = model.alpha if space == "feature" else model.correction.weight
            self.assertGreater(p.grad.abs().sum(), 0)
            optim.step()
            for key, value in original.items():
                torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
            optim.zero_grad()
            model(self.q).square().mean().backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.fp.parameters()))
            restored = ResidualFP(32, 2, space, frozen=True).eval()
            restored.load_state_dict(model.state_dict())
            torch.testing.assert_close(model.eval()(self.q), restored(self.q), atol=0, rtol=0)

    def test_direct_baseline_and_empty_graph_guard(self):
        model = SingleBaseline(38, 13, hidden_size=32, num_layers=2, dropout=0).eval()
        model.load_state_dict(self.base)
        source = GraphCliffRegressor(38, 13, hidden_size=32, num_layers=2, dropout=0).eval()
        source.load_state_dict(self.base)
        def call(m, g):
            return m(g.x, g.edge_index, g.edge_attr, g.batch)
        torch.testing.assert_close(call(model, self.q), call(source, self.q), atol=0, rtol=0)
        mixed, single = graphs(["N", "CCCC"]), graphs(["N"])
        torch.testing.assert_close(call(model, mixed)[:1], call(model, single), atol=1e-6, rtol=1e-5)

    def test_attention_centering_and_permutation(self):
        module = LayerInteraction(32, 4).eval()
        q, r = torch.randn(3, 32), torch.randn(5, 32)
        qb, rb = torch.zeros(3, dtype=torch.long), torch.zeros(5, dtype=torch.long)
        a, b = module(q, q, qb, qb, "centered")
        torch.testing.assert_close(a, torch.zeros_like(a), atol=0, rtol=0)
        torch.testing.assert_close(b, torch.zeros_like(b), atol=0, rtol=0)
        a, b = module(q, r, qb, rb, "centered")
        perm = torch.tensor([2, 0, 1])
        ap, bp = module(q[perm], r, qb, rb, "centered")
        torch.testing.assert_close(ap, a[perm], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(bp, b, atol=1e-6, rtol=1e-5)

    def test_pair_controls_identity_isolation_gradient_reload(self):
        counts = []
        for mode in ("full", "self", "cross", "centered", "full_fp", "centered_fp"):
            torch.manual_seed(42)
            model = LayerSwap("retry_" + mode, 32, 2).eval()
            model.load_shared(self.base)
            if mode in ("self", "cross", "centered"):
                counts.append(parameters(model))
            if mode == "centered":
                # The learned affine transform must still map a zero innovation to zero.
                with torch.no_grad():
                    for layer in model.encoder.layers:
                        layer.long.group_bias.fill_(.4)
            result = model(self.q, self.r)
            torch.testing.assert_close(result, -model(self.r, self.q), atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(model(self.q, self.q), torch.zeros_like(result), atol=1e-6, rtol=0)
            torch.testing.assert_close(result[:1], model(graphs(["CCO"]), graphs(["N"])), atol=1e-6, rtol=1e-5)
            restored = LayerSwap("retry_" + mode, 32, 2).eval()
            restored.load_state_dict(model.state_dict())
            torch.testing.assert_close(result, restored(self.q, self.r), atol=0, rtol=0)
            result.square().mean().backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            if mode not in ("full", "full_fp"):
                self.assertGreater(model.interactions[0].attention.in_proj_weight.grad.abs().sum(), 0)
            else:
                original = PairRegressor(hidden_size=32, num_layers=2).eval()
                original.load_shared(self.base)
                torch.testing.assert_close(result, original(self.q, self.r), atol=0, rtol=0)
        self.assertEqual(len(set(counts)), 1)


if __name__ == "__main__":
    unittest.main()
