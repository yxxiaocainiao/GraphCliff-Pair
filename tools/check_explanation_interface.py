"""Synthetic PyG node-mask interface checks; no fitted explanations or training."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

import torch
import torch_geometric
from torch import nn
from torch_geometric.data import Data, Batch
from torch_geometric.explain import Explainer, GNNExplainer, Explanation
from torch_geometric.explain.metric import fidelity, unfaithfulness

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from graphcliff_pair.model import PairRegressor
from graphcliff_pair.vendor.model import GraphCliffRegressor


class PairView(nn.Module):
    """Explain query features with a fixed reference; membership/topology unchanged."""
    def __init__(self,core,query,reference):
        super().__init__()
        self.core,self.query,self.reference=core,query,reference

    def forward(self,x,edge_index,**kwargs):
        query=self.query.clone()
        query.x,query.edge_index=x,edge_index
        return self.core(query,self.reference)


def graph(n):
    edges=torch.tensor([[i for i in range(n-1)]+[i+1 for i in range(n-1)],
                        [i+1 for i in range(n-1)]+[i for i in range(n-1)]])
    return Batch.from_data_list([Data(x=torch.randn(n,38),edge_index=edges,
        edge_attr=torch.randn(edges.size(1),13),atom_fp=torch.ones(n,1024))])


def main(output):
    torch.manual_seed(20261005)
    query,reference=graph(3),graph(4)
    results=[]
    for name in ['direct','global_sag','cross_sag','global_fppool','cross_fppool']:
        if name=='direct':
            model=GraphCliffRegressor(38,13,hidden_size=32,num_layers=1,dropout=0).eval()
            kwargs=dict(edge_attr=query.edge_attr,batch=query.batch)
        else:
            variant='cross_attention' if name.startswith('cross') else 'global_diff'
            readout='fppool' if name.endswith('fppool') else 'sag'
            model=PairView(PairRegressor(variant,32,1,4,readout),query,reference).eval()
            kwargs={}
        before={k:v.clone() for k,v in model.state_dict().items()}
        explainer=Explainer(model=model,algorithm=GNNExplainer(),explanation_type='model',
                           model_config=dict(mode='regression',task_level='graph',return_type='raw'),
                           node_mask_type='object',edge_mask_type=None)
        base=explainer.get_prediction(query.x,query.edge_index,**kwargs)
        ones=explainer.get_masked_prediction(query.x,query.edge_index,node_mask=torch.ones(3,1),**kwargs)
        zero=explainer.get_masked_prediction(query.x,query.edge_index,node_mask=torch.zeros(3,1),**kwargs)
        assert base.shape==ones.shape==zero.shape==(1,1)
        assert torch.equal(base,ones) and torch.isfinite(zero).all()
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
        rejected=[]
        for metric in [fidelity,unfaithfulness]:
            try:
                metric(explainer,Explanation())
                raise AssertionError('Expected regression guard')
            except ValueError as error:
                assert 'regression' in str(error)
                rejected.append(metric.__name__)
        results.append(dict(model=name,all_one_mask_exact=True,all_zero_mask_finite=True,
                            parameters_unchanged=True,classification_metrics_rejected=rejected))
    files=[Path(inspect.getfile(Explainer)),Path(inspect.getfile(GNNExplainer)),Path(inspect.getfile(fidelity)),
           ROOT/'graphcliff_pair/model.py',ROOT/'graphcliff_pair/vendor/model.py',ROOT/'graphcliff_pair/fppool.py',
           ROOT/'external/fppool/pooling.py',Path(__file__)]
    output.parent.mkdir(parents=True,exist_ok=True)
    assert not output.exists()
    result=dict(torch=torch.__version__,pyg=torch_geometric.__version__,device='cpu',passed=True,checks=results,
        scope='Random synthetic H32 models, node-feature mask plumbing only. No trained checkpoint replay, learned explanation, edge-mask coverage or chemical validity claim.',
        no_optimizer_steps=True,input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,models=len(results),no_optimizer_steps=True),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    main(parser.parse_args().output)
