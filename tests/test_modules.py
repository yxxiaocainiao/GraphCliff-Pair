import unittest
import torch
from torch_geometric.data import Batch, Data
from graphcliff_pair.fingerprint import membership
from graphcliff_pair.loss import DeltaLoss
from graphcliff_pair.model import PairRegressor
from graphcliff_pair.vendor.dataset_utils import smiles_to_graph

def fp_graphs(smiles):
    graphs=[]
    for s in smiles:
        g=smiles_to_graph(s,0)
        graphs.append(Data(x=g.x,edge_index=g.edge_index,edge_attr=g.edge_attr,atom_fp=membership(g)))
    return Batch.from_data_list(graphs)

class ModuleContracts(unittest.TestCase):
    def test_weight_identity_gradient_and_bounds(self):
        p=torch.tensor([[0.3],[-1.]],requires_grad=True)
        y=torch.tensor([[0.],[-3.]])
        mse=torch.nn.functional.mse_loss(p,y)
        expected=torch.autograd.grad(mse,p,retain_graph=True)[0]
        for loss in [DeltaLoss(),DeltaLoss('dynamic',alpha_max=0),DeltaLoss('dynamic')]:
            value,weight=loss(p,y,epoch=0)
            torch.testing.assert_close(value,mse)
            torch.testing.assert_close(torch.autograd.grad(value,p,retain_graph=True)[0],expected)
            torch.testing.assert_close(weight,torch.ones_like(y))
        dynamic=DeltaLoss('dynamic')
        value,weight=dynamic(p,y,epoch=10)
        self.assertTrue(torch.isfinite(value))
        torch.testing.assert_close(weight.mean(),torch.tensor(1.))
        self.assertGreater(weight[1],weight[0])
        _,reversed_weight=dynamic(-p,-y,epoch=10)
        torch.testing.assert_close(weight,reversed_weight)
        with self.assertRaises(ValueError):
            dynamic(p,y.flatten())

    def test_fppool_forward_sign_gradient(self):
        torch.set_num_threads(2)
        q=fp_graphs(['CCO','c1ccccc1O'])
        r=fp_graphs(['CCN','CCCC'])
        for variant in ['global_diff','cross_attention']:
            model=PairRegressor(variant,hidden_size=32,num_layers=1,readout='fppool').eval()
            pred=model(q,r)
            torch.testing.assert_close(pred,-model(r,q),atol=2e-6,rtol=1e-5)
            pred.square().mean().backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.fppool.parameters()))

if __name__=='__main__':
    unittest.main()
