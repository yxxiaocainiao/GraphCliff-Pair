import tempfile
import unittest
from pathlib import Path
import torch
from torch import nn
from torch_geometric.data import Batch, Data
from graphcliff_pair.model import PairRegressor
from graphcliff_pair.vendor.dataset_utils import smiles_to_graph
from graphcliff_pair.vendor.model import GraphCliffRegressor
from experiments.long_branch_swap.model import BranchSwap
from experiments.long_branch_swap.run import safe_frame

def graphs(smiles):
    values = [smiles_to_graph(s, 0) for s in smiles]
    return Batch.from_data_list([Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr) for g in values])

class IdentityLong(nn.Module):
    def forward(self, x, *args):
        return x

class BranchContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.base = GraphCliffRegressor(38,13,hidden_size=32,num_layers=2,dropout=0).state_dict()
        self.q = graphs(["CCO", "c1ccccc1O"])
        self.r = graphs(["N", "CCCC"])

    def test_original_and_local_controls(self):
        original = PairRegressor(hidden_size=32,num_layers=2).eval()
        original.load_shared(self.base)
        full = BranchSwap("branch_full",32,2).eval()
        full.load_shared(self.base)
        torch.testing.assert_close(original(self.q,self.r),full(self.q,self.r),atol=0,rtol=0)
        short = BranchSwap("branch_short",32,2).eval()
        short.load_shared(self.base)
        for layer in original.encoder.layers:
            layer.long = IdentityLong()
        torch.testing.assert_close(original(self.q,self.r),short(self.q,self.r),atol=1e-6,rtol=1e-5)

    def test_pair_contracts_and_shared_initialization(self):
        for mode in ["full","short","cross"]:
            with self.subTest(mode=mode):
                model = BranchSwap("branch_"+mode,32,2).eval()
                shared = model.load_shared(self.base)
                for k,v in shared.items():
                    self.assertTrue(torch.equal(model.state_dict()[k],v))
                if mode != "full":
                    self.assertFalse(any(".long." in k for k in model.state_dict()))
                pred = model(self.q,self.r)
                torch.testing.assert_close(pred,-model(self.r,self.q),atol=1e-6,rtol=1e-5)
                torch.testing.assert_close(model(self.q,self.q),torch.zeros_like(pred),atol=1e-6,rtol=0)
                torch.testing.assert_close(pred[:1],model(graphs(["CCO"]),graphs(["N"])),atol=1e-6,rtol=1e-5)
                changed = model(self.q,graphs(["N","CCCCCCCCCC"]))
                torch.testing.assert_close(pred[:1],changed[:1],atol=1e-6,rtol=1e-5)
                loaded = BranchSwap("branch_"+mode,32,2).eval()
                loaded.load_state_dict(model.state_dict())
                torch.testing.assert_close(pred,loaded(self.q,self.r),atol=0,rtol=0)
                pred.square().mean().backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.encoder.layers[0].short.parameters()))
                if mode == "cross":
                    for interaction in model.branch_interactions:
                        self.assertGreater(interaction.attention.in_proj_weight.grad.abs().sum(),0)

    def test_test_labels_not_parsed(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"poison.csv"
            path.write_text("smiles,split,y,cliff_mol\nCC,train,1.2,1\nCCC,test,DO_NOT_PARSE,POISON\nCO,train,2.3,0\n")
            frame=safe_frame(path,[0,2])
            self.assertEqual(frame.loc[[0,2],"y"].tolist(),[1.2,2.3])
            self.assertTrue(frame.loc[1,["y","cliff_mol"]].isna().all())
            with self.assertRaises(ValueError):
                safe_frame(path,[0,1,2])
