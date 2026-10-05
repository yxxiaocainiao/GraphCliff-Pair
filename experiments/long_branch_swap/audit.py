"""Validation-only audit and checkpoint replay on the recorded training device."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from graphcliff_pair import train, data
from graphcliff_pair.vendor.model import GraphCliffRegressor
from graphcliff_pair.vendor.training import set_seed
from .model import BranchSwap
from .run import frozen_read, ROOT

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def audit(output):
    output=Path(output)
    manifest=read(output/"manifest.json")
    config=manifest["config"]
    if manifest["test_evaluated"] or read(output/"completed.json")["test_evaluated"]:
        raise ValueError("test use detected")
    for key in ["source_sha256","experiment_source_sha256"]:
        for name,sha in manifest[key].items():
            if data.digest(ROOT/name)!=sha:
                raise ValueError("source changed: "+name)
    expected=len(config["datasets"])*len(config["seeds"])*len(config["arms"])
    summaries=read(output/"summary.json")
    if len(summaries)!=expected or read(output/"completed.json")["runs"]!=expected:
        raise ValueError("incomplete matrix")
    torch.set_num_threads(2)
    device=torch.device(manifest["device"])
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("replay requires training GPU")
    rows=[]
    for dataset in config["datasets"]:
        frozen_path=ROOT/"artifacts/interaction_seed42_20261004"/dataset/"pairs.json"
        if data.digest(frozen_path)!=manifest["frozen_pair_sha256"][dataset]:
            raise ValueError("frozen pair manifest changed")
        frame,graphs,tr,va,tp,vp=frozen_read(Path(r"D:/GraphCliff-main/benchmark_data")/(dataset+".csv"),config["split_seed"])
        tp=tp[:config.get("limit_train_queries",len(tp))]
        vp=vp[:config.get("limit_valid_queries",len(vp))]
        pairs=read(output/dataset/"pairs.json")
        if pairs["train_pairs"]!=tp or pairs["valid_pairs"]!=vp or pairs["train_rows"]!=tr.tolist() or pairs["valid_rows"]!=va.tolist():
            raise ValueError("recorded pairs differ")
        for seed in config["seeds"]:
            set_seed(SimpleNamespace(seed=seed))
            base=GraphCliffRegressor(38,13,hidden_size=config["hidden_size"],num_layers=config["num_layers"],dropout=0).state_dict()
            common_hashes=[]
            heads=[]
            for arm in config["arms"]:
                folder=output/dataset/f"seed{seed}"/arm["name"]
                summary=read(folder/"summary.json")
                history=read(folder/"history.json")
                set_seed(SimpleNamespace(seed=seed))
                model=BranchSwap(arm["variant"],config["hidden_size"],config["num_layers"],config["heads"],arm["readout"])
                shared=model.load_shared(base)
                init=read(folder/"initialization.json")
                if train.state_hash(model.state_dict())!=init["full_sha256"]:
                    raise ValueError("initial state differs")
                common={k:v for k,v in shared.items() if ".long." not in k}
                common_hashes.append(train.state_hash(common))
                heads.append(train.state_hash(model.head.state_dict()))
                if train.parameters(model)!=summary["parameters"]:
                    raise ValueError("parameter count differs")
                best=min(history,key=lambda x:x["valid_mse"])["epoch"]
                if best!=summary["best_epoch"] or len(history)!=summary["epochs_run"] or len(history)>config["epochs"]:
                    raise ValueError("checkpoint selection differs")
                if any(h["weight_min"]!=1 or h["weight_max"]!=1 for h in history):
                    raise ValueError("not ordinary MSE")
                saved=pd.read_csv(folder/"validation_predictions.csv")
                recomputed=train.metrics(saved.to_dict("records"))
                for k,v in recomputed.items():
                    if v is not None and not np.isclose(v,summary[k],atol=1e-10,rtol=1e-9):
                        raise ValueError("metric differs: "+k)
                model=model.to(device)
                checkpoint=torch.load(folder/"best.pt",map_location=device,weights_only=True)
                if checkpoint["epoch"]!=best:
                    raise ValueError("checkpoint epoch differs")
                model.load_state_dict(checkpoint["model_state_dict"])
                replay=pd.DataFrame(train.evaluate(model,arm,vp,graphs,frame,config["batch_size"],device))
                for k in ["query","reference","cliff_mol"]:
                    np.testing.assert_array_equal(saved[k],replay[k])
                for k in ["similarity","y","reference_y"]:
                    np.testing.assert_allclose(saved[k],replay[k],atol=1e-12,rtol=1e-12)
                np.testing.assert_allclose(saved.prediction,replay.prediction,atol=1e-5,rtol=1e-6)
                matching=[r for r in summaries if r["dataset"]==dataset and r["seed"]==seed and r["arm"]==arm["name"]]
                if len(matching)!=1 or any(matching[0][k]!=v for k,v in summary.items()):
                    raise ValueError("summary identity differs")
                rows.append(dict(dataset=dataset,seed=seed,arm=arm["name"],**recomputed,parameters=summary["parameters"],epochs=summary["epochs_run"],elapsed_seconds=summary["elapsed_seconds"],max_replay_error=float(np.max(np.abs(saved.prediction-replay.prediction)))))
                del model
            if len(set(common_hashes))!=1 or len(set(heads))!=1:
                raise ValueError("retained initialization differs across arms")
    result=dict(runs=len(rows),test_evaluated=False,checkpoint_replayed=True,rows=rows)
    train.save_json(output/"audit.json",result)
    print(json.dumps(result,indent=2),flush=True)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    audit(p.parse_args().output)
