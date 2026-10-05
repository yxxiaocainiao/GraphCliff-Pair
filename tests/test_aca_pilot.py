# SPDX-License-Identifier: GPL-3.0-only
import unittest
import torch
from torch_geometric.data import Batch,Data
from graphcliff_pair.vendor.dataset_utils import smiles_to_graph
from graphcliff_pair.vendor.model import GraphCliffRegressor
from graphcliff_pair.train import state_hash
from experiments.aca_pilot.run import CapturedGraphCliff,ACTIVE
from experiments.aca_pilot.loss_adapter import upstream_class,CapturedLoss

def batch():
    values=[smiles_to_graph(s,0) for s in ["CCO","CCN","CCCO","CCCN"]]
    return Batch.from_data_list([Data(x=g.x,edge_index=g.edge_index,edge_attr=g.edge_attr) for g in values])

class ACAContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
    def tearDown(self):
        ACTIVE.clear()
    def test_original_state_forward_and_reload(self):
        torch.manual_seed(42);a=GraphCliffRegressor(38,13,hidden_size=32,num_layers=2,dropout=0).eval()
        torch.manual_seed(42);b=CapturedGraphCliff(38,13,hidden_size=32,num_layers=2,dropout=0).eval()
        self.assertEqual(state_hash(a.state_dict()),state_hash(b.state_dict()))
        g=batch();x=a(g.x,g.edge_index,g.edge_attr,g.batch)
        torch.testing.assert_close(x,b(g.x,g.edge_index,g.edge_attr,g.batch),atol=0,rtol=0)
        self.assertIsNone(b.embedding)
        b.train();b(g.x,g.edge_index,g.edge_attr,g.batch)
        self.assertEqual(b.embedding.shape,(4,64))
        self.assertTrue(b.embedding.requires_grad)
        b.load_state_dict(a.state_dict());b.eval()
        torch.testing.assert_close(x,b(g.x,g.edge_index,g.edge_attr,g.batch),atol=0,rtol=0)
    def test_mse_identity_and_representation_gradient(self):
        y=torch.tensor([0.,.2,1.5,2.]);pred=torch.tensor([.1,.3,1.4,2.2],requires_grad=True)
        embedding=torch.tensor([[0.,0.],[3.,0.],[.1,0.],[.2,0.]],requires_grad=True)
        cls=upstream_class()
        base=cls(alpha=0,squared=True,p=2,cliff_lower=1,cliff_upper=1,dev_mode=True)
        result=base(y,pred,embedding)
        mse=torch.nn.functional.mse_loss(pred,y)
        torch.testing.assert_close(result[0],mse,atol=0,rtol=0)
        torch.testing.assert_close(torch.autograd.grad(result[0],pred,retain_graph=True)[0],torch.autograd.grad(mse,pred,retain_graph=True)[0],atol=0,rtol=0)
        aca=cls(alpha=.1,squared=True,p=2,cliff_lower=1,cliff_upper=1,dev_mode=True)
        r=aca(y,pred,embedding);self.assertGreater(r[5],0);self.assertGreater(r[6],0)
        r[0].backward();self.assertTrue(torch.isfinite(embedding.grad).all());self.assertGreater(embedding.grad.abs().sum(),0)
        empty=aca(torch.zeros(4),pred.detach(),embedding.detach())
        self.assertEqual(empty[5],0);self.assertEqual(empty[6],0)
        torch.testing.assert_close(empty[0],torch.nn.functional.mse_loss(pred.detach(),torch.zeros(4)))
    def test_hook_loss_gradient_and_clear(self):
        torch.manual_seed(42);m=CapturedGraphCliff(38,13,hidden_size=32,num_layers=2,dropout=0).train();g=batch()
        pred=m(g.x,g.edge_index,g.edge_attr,g.batch);stats=[]
        criterion=CapturedLoss(m,.1,1,1,stats)
        loss,w=criterion(pred,torch.tensor([0.,.2,1.5,2.]).reshape(-1,1),1);loss.backward()
        self.assertIsNone(m.embedding);self.assertTrue(torch.equal(w,torch.ones_like(w)))
        self.assertGreater(stats[0]['candidate_triplets'],0)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in m.encoder.parameters()))
