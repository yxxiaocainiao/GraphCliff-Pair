import copy
import io
import unittest

import torch
from torch_geometric.data import Batch, Data
from graphcliff_pair.model import PairRegressor, parameters
from graphcliff_pair.vendor.dataset_utils import smiles_to_graph
from graphcliff_pair.vendor.model import GraphCliffRegressor

def graphs(smiles):
    values = []
    for s in smiles:
        g = smiles_to_graph(s, 0)
        values.append(Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr))
    return Batch.from_data_list(values)

class ModelContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.q = graphs(["CCO", "c1ccccc1O"])
        self.r = graphs(["N", "CCCC"])

    def test_sign_identity_gradient_reload(self):
        for variant in ["global_diff", "pair_mlp", "cross_attention"]:
            with self.subTest(variant=variant):
                model = PairRegressor(variant, hidden_size=32, num_layers=1).eval()
                pred = model(self.q, self.r)
                self.assertEqual(pred.shape, (2, 1))
                torch.testing.assert_close(pred, -model(self.r, self.q), atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(model(self.q, self.q), torch.zeros_like(pred), atol=1e-6, rtol=0)
                pred.square().mean().backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()))
                if variant == "cross_attention":
                    self.assertGreater(model.interaction.attention.in_proj_weight.grad.abs().sum(), 0)
                stream = io.BytesIO()
                torch.save(model.state_dict(), stream)
                stream.seek(0)
                loaded = copy.deepcopy(model)
                loaded.load_state_dict(torch.load(stream, weights_only=True))
                torch.testing.assert_close(pred, loaded(self.q, self.r))

    def test_ragged_padding_and_pair_isolation(self):
        model = PairRegressor("cross_attention", hidden_size=32, num_layers=1).eval()
        batched = model(self.q, self.r)
        single = model(graphs(["CCO"]), graphs(["N"]))
        torch.testing.assert_close(batched[:1], single, atol=2e-6, rtol=1e-5)
        changed = model(self.q, graphs(["N", "CCCCCCCCCCCC"]))
        torch.testing.assert_close(batched[:1], changed[:1], atol=2e-6, rtol=1e-5)
        with self.assertRaises(ValueError):
            model(self.q, graphs(["N"]))

    def test_capacity_control(self):
        attention = PairRegressor("cross_attention")
        mlp = PairRegressor("pair_mlp")
        added_a = parameters(attention.head) + parameters(attention.interaction)
        added_b = parameters(mlp.head)
        self.assertLess(abs(added_a-added_b) / added_a, 0.005)

    def test_shared_encoder_and_head_initialization(self):
        base=GraphCliffRegressor(38,13,hidden_size=32,num_layers=1,dropout=0).state_dict()
        global_model=PairRegressor('global_diff',hidden_size=32,num_layers=1)
        cross_model=PairRegressor('cross_attention',hidden_size=32,num_layers=1)
        global_model.load_shared(base)
        cross_model.load_shared(base)
        for key,value in global_model.head.state_dict().items():
            torch.testing.assert_close(value,cross_model.head.state_dict()[key],rtol=0,atol=0)
        for key,value in global_model.encoder.state_dict().items():
            torch.testing.assert_close(value,cross_model.encoder.state_dict()[key],rtol=0,atol=0)

if __name__ == "__main__":
    unittest.main()
