"""GPL-3.0: thin ACA adapter; original direct model, trainer and inference retained."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from graphcliff_pair import data,train
from graphcliff_pair.vendor.model import GraphCliffRegressor
from experiments.long_branch_swap.run import safe_frame
from .loss_adapter import CapturedLoss
ROOT=train.ROOT
ACTIVE={}

class CapturedGraphCliff(GraphCliffRegressor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.embedding=None
        self.reg_head.register_forward_pre_hook(self.capture)
        ACTIVE["model"]=self
    def capture(self,module,args):
        if self.training:
            self.embedding=args[0]

def read_safe(path,split_seed=42):
    structure=pd.read_csv(path,usecols=["smiles","split"])
    rows=structure.index[structure.split=="train"].tolist()
    frame=safe_frame(path,rows)
    reader=data.pd.read_csv
    data.pd.read_csv=lambda request,**kw:frame.copy() if Path(request).resolve()==Path(path).resolve() else (_ for _ in ()).throw(ValueError("unexpected input"))
    try:
        result=data.read_development(path,split_seed)
    finally:
        data.pd.read_csv=reader
    # Calibration reads raw potency only on the newly carved training rows.
    tr=result[2];allowed=set(int(i) for i in tr)
    raw=pd.read_csv(path,usecols=["exp_mean [nM]"],skiprows=lambda line:line>0 and line-1 not in allowed)
    raw.index=sorted(allowed)
    values=raw["exp_mean [nM]"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values<=0).any():
        raise ValueError("invalid training concentration")
    unscaled=-np.log10(values)
    y=frame.loc[raw.index,"y"].to_numpy(dtype=float)
    slope,intercept=np.polyfit(unscaled,y,1)
    residual=float(np.max(np.abs(slope*unscaled+intercept-y)))
    if slope<=0 or residual>1e-5:
        raise ValueError("target is not an audited affine log-potency scale")
    ACTIVE["calibration"]=dict(slope=float(slope),intercept=float(intercept),max_residual=residual,rows=sorted(allowed),unit="y = slope * -log10(nM) + intercept; training-only fit")
    return result

def run(config_path,csv_root,output):
    cfg=json.loads(Path(config_path).read_text(encoding="utf-8"))
    expected=[dict(name="mse",variant="direct",readout="sag",loss="mse",aca_alpha=0.),dict(name="aca",variant="direct",readout="sag",loss="mse",aca_alpha=0.1)]
    if cfg["arms"]!=expected:
        raise ValueError("frozen arms changed")
    originals=train.GraphCliffRegressor,train.DeltaLoss,train.read_development,train.train_arm,train.save_json
    def make_loss(mode,**kwargs):
        slope=ACTIVE["calibration"]["slope"]
        return CapturedLoss(ACTIVE["model"],ACTIVE["alpha"],cfg["cliff_lower_log10"]*slope,cfg["cliff_upper_log10"]*slope,ACTIVE["stats"])
    def arm(config,choice,*args,**kwargs):
        ACTIVE["alpha"]=choice["aca_alpha"];ACTIVE["stats"]=[]
        result=originals[3](config,choice,*args,**kwargs)
        folder=Path(args[6] if len(args)>6 else kwargs["output"])
        stats=ACTIVE["stats"];epochs=[]
        for epoch in sorted({s["epoch"] for s in stats}):
            selected=[s for s in stats if s["epoch"]==epoch];n=sum(s["batch"] for s in selected)
            epochs.append(dict(epoch=epoch,reg=sum(s["reg"]*s["batch"] for s in selected)/n,tsm=sum(s["tsm"]*s["batch"] for s in selected)/n,candidate_triplets=sum(s["candidate_triplets"] for s in selected),high_value_triplets=sum(s["high_value_triplets"] for s in selected),zero_triplet_batches=sum(s["candidate_triplets"]==0 for s in selected),batches=len(selected)))
        originals[4](folder/"aca_history.json",epochs)
        result.update(aca_alpha=choice["aca_alpha"],calibration=ACTIVE["calibration"],aca_candidate_triplets=sum(s["candidate_triplets"] for s in stats),aca_high_value_triplets=sum(s["high_value_triplets"] for s in stats))
        originals[4](folder/"summary.json",result)
        return result
    def save(path,value):
        if Path(path).name=="manifest.json":
            value["experiment_source_sha256"]={str(p.relative_to(ROOT)):data.digest(p) for p in sorted(Path(__file__).parent.rglob("*.py"))}
            value["aca_sources_sha256"]=data.digest(Path(__file__).parent/"sources.json")
            value["test_label_policy"]="labels parsed only on official training rows; potency calibration on carved training rows only"
        originals[4](path,value)
    train.GraphCliffRegressor,train.DeltaLoss,train.read_development,train.train_arm,train.save_json=CapturedGraphCliff,make_loss,read_safe,arm,save
    try:
        train.run(config_path,csv_root,output)
    finally:
        train.GraphCliffRegressor,train.DeltaLoss,train.read_development,train.train_arm,train.save_json=originals
        ACTIVE.clear()

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--csv-root",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();run(a.config,a.csv_root,a.output)
