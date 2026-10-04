import unittest
import torch
from graphcliff_pair.model import PairRegressor
from graphcliff_pair.predict import graph_and_identity,infer

class PredictContracts(unittest.TestCase):
    def test_order_batch_and_reference_label_boundary(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        model=PairRegressor('cross_attention',hidden_size=32,num_layers=1).eval()
        bank={}
        for row,smiles in [(4,'CCN'),(9,'CCC')]:
            graph,canonical,fp=graph_and_identity(smiles)
            bank[row]=dict(graph=graph,canonical=canonical,fp=fp,y=5.)
        queries=['CCO','c1ccccc1O']
        first=infer(model,'cross_attention',queries,bank,batch_size=1)
        second=infer(model,'cross_attention',queries,bank,batch_size=2)
        self.assertEqual([r['smiles'] for r in first],queries)
        for a,b in zip(first,second):
            self.assertAlmostEqual(a['prediction'],b['prediction'],places=5)
            self.assertEqual(a['reference_row'],b['reference_row'])
        for value in bank.values(): value['y']+=2
        shifted=infer(model,'cross_attention',queries,bank,batch_size=2)
        for a,b in zip(first,shifted):
            self.assertAlmostEqual(b['prediction']-a['prediction'],2,places=5)
            self.assertAlmostEqual(a['delta_prediction'],b['delta_prediction'],places=5)
            self.assertEqual(a['reference_row'],b['reference_row'])

if __name__=='__main__':
    unittest.main()
