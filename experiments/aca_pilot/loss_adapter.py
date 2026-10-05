"""GPL-3.0: execute unmodified upstream label-only ACA definitions, not a rewrite."""
import ast
from pathlib import Path
from typing import Optional, List
import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

def upstream_class():
    path=Path(__file__).parent/"vendor/upstream_loss.py"
    tree=ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
    wanted=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ("ACALoss","_aca_loss")]
    if len(wanted)!=2:
        raise RuntimeError("upstream interface changed")
    namespace=dict(torch=torch,Tensor=Tensor,_Loss=_Loss,Optional=Optional,List=List)
    exec(compile(ast.Module(body=wanted,type_ignores=[]),str(path),"exec"),namespace)
    return namespace["ACALoss"]

class CapturedLoss:
    def __init__(self,model,alpha,lower,upper,stats):
        self.model,self.stats=model,stats
        self.seen_candidate=False
        self.criterion=upstream_class()(alpha=alpha,cliff_lower=lower,cliff_upper=upper,squared=True,p=2.,similarity_gate=False,dev_mode=True)
    def __call__(self,pred,target,epoch):
        if epoch>1 and not self.seen_candidate:
            raise RuntimeError("no candidate training triplets in epoch one")
        if self.model.embedding is None:
            raise RuntimeError("missing pooled embedding")
        loss,reg,tsm,ny,ns,nact,nhv=self.criterion(target,pred,self.model.embedding)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite ACA loss")
        self.seen_candidate=self.seen_candidate or nact>0
        self.stats.append(dict(epoch=epoch,reg=float(reg.detach()),tsm=float(tsm.detach()),candidate_triplets=nact,high_value_triplets=nhv,batch=len(target)))
        self.model.embedding=None
        return loss,torch.ones_like(target)
