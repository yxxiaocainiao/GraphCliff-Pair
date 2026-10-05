# SPDX-License-Identifier: GPL-3.0-only
"""Independent metric/source/initialization audit and actual GPU checkpoint replay."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from graphcliff_pair import data,train
from graphcliff_pair.vendor.training import set_seed
from .run import read_safe,CapturedGraphCliff,ACTIVE,ROOT

def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))

def audit(output,dataset):
    output=Path(output);manifest=read(output/"manifest.json");c=dict(manifest["config"]);c["datasets"]=[dataset]
    if manifest["test_evaluated"] or read(output/"stopped.json")["test_evaluated"]:
        raise ValueError("test use detected")
    for field in ["source_sha256","experiment_source_sha256"]:
        for name,sha in manifest[field].items():
            if data.digest(ROOT/name)!=sha:raise ValueError("source changed: "+name)
    if data.digest(Path(__file__).parent/"sources.json")!=manifest["aca_sources_sha256"]:raise ValueError("ACA provenance changed")
    provenance=read(Path(__file__).parent/"sources.json")
    for name,sha in provenance["files"].items():
        if data.digest(Path(__file__).parent/"vendor"/name)!=sha:raise ValueError("upstream byte identity changed")
    expected=len(c["datasets"])*len(c["seeds"])*len(c["arms"])
    all_aggregate=read(output/"summary.json");aggregate=[r for r in all_aggregate if r["dataset"]==dataset]
    if len(aggregate)!=expected or read(output/"stopped.json")["completed_runs"]!=len(all_aggregate):raise ValueError("incomplete scoped matrix or incorrect closure")
    device=torch.device(manifest["device"]);torch.set_num_threads(2);rows=[]
    for d in c["datasets"]:
        frame,graphs,tr,va,tp,vp=read_safe(Path(r"D:/GraphCliff-main/benchmark_data")/(d+".csv"),c["split_seed"])
        calibration=ACTIVE["calibration"].copy();tp=tp[:c.get("limit_train_queries",len(tp))];vp=vp[:c.get("limit_valid_queries",len(vp))]
        pair=read(output/d/"pairs.json")
        if pair["train_rows"]!=tr.tolist() or pair["valid_rows"]!=va.tolist() or pair["train_pairs"]!=tp or pair["valid_pairs"]!=vp or pair["input_sha256"]!=data.digest(Path(r"D:/GraphCliff-main/benchmark_data")/(d+".csv")):raise ValueError("data identity changed")
        for seed in c["seeds"]:
            initial=[]
            for arm in c["arms"]:
                folder=output/d/f"seed{seed}"/arm["name"];summary=read(folder/"summary.json");history=read(folder/"history.json");ah=read(folder/"aca_history.json")
                set_seed(SimpleNamespace(seed=seed));model=CapturedGraphCliff(38,13,hidden_size=c["hidden_size"],num_layers=c["num_layers"],dropout=0)
                sha=train.state_hash(model.state_dict());initial.append(sha)
                if sha!=read(folder/"initialization.json")["full_sha256"]:raise ValueError("initialization changed")
                if train.parameters(model)!=summary["parameters"] or summary["calibration"]!=calibration or summary["aca_alpha"]!=arm["aca_alpha"]:raise ValueError("parameter/calibration identity changed")
                if len(history)!=len(ah) or len(history)!=summary["epochs_run"] or len(history)>c["epochs"]:raise ValueError("epoch budget changed")
                for h,s in zip(history,ah):
                    if h["epoch"]!=s["epoch"] or h["weight_min"]!=1 or h["weight_max"]!=1:raise ValueError("unexpected weighting")
                    if not np.isclose(h["train_loss"],s["reg"]+arm["aca_alpha"]*s["tsm"],atol=1e-5,rtol=1e-6):raise ValueError("objective reconstruction failed")
                if sum(s["candidate_triplets"] for s in ah)!=summary["aca_candidate_triplets"] or sum(s["high_value_triplets"] for s in ah)!=summary["aca_high_value_triplets"] or not summary["aca_candidate_triplets"]:raise ValueError("triplet accounting failed")
                best=min(history,key=lambda h:h["valid_mse"])["epoch"]
                if best!=summary["best_epoch"]:raise ValueError("selection rule changed")
                saved=pd.read_csv(folder/"validation_predictions.csv");met=train.metrics(saved.to_dict("records"))
                for k,v in met.items():
                    if v is not None and not np.isclose(v,summary[k],atol=1e-10,rtol=1e-9):raise ValueError("saved metric changed")
                model=model.to(device);checkpoint=torch.load(folder/"best.pt",map_location=device,weights_only=True)
                if checkpoint["epoch"]!=best:raise ValueError("checkpoint epoch changed")
                model.load_state_dict(checkpoint["model_state_dict"])
                replay=pd.DataFrame(train.evaluate(model,arm,vp,graphs,frame,c["batch_size"],device))
                for k in ["query","reference","cliff_mol"]:np.testing.assert_array_equal(saved[k],replay[k])
                for k in ["similarity","y","reference_y"]:np.testing.assert_allclose(saved[k],replay[k],atol=1e-12,rtol=1e-12)
                np.testing.assert_allclose(saved.prediction,replay.prediction,atol=1e-5,rtol=1e-6)
                matching=[r for r in aggregate if r["dataset"]==d and r["seed"]==seed and r["arm"]==arm["name"]]
                if len(matching)!=1 or any(matching[0][k]!=v for k,v in summary.items()):raise ValueError("aggregate identity changed")
                rows.append(dict(dataset=d,seed=seed,arm=arm["name"],**met,epochs=summary["epochs_run"],best_epoch=best,parameters=summary["parameters"],elapsed_seconds=summary["elapsed_seconds"],candidate_triplets=summary["aca_candidate_triplets"],high_value_triplets=summary["aca_high_value_triplets"],max_replay_error=float(np.max(np.abs(saved.prediction-replay.prediction)))))
                del model
            if len(set(initial))!=1:raise ValueError("unmatched arms")
    result=dict(runs=len(rows),status="paused_early",dataset=dataset,planned_runs=len(manifest["config"]["datasets"])*len(c["seeds"])*len(c["arms"]),completed_runs=len(all_aggregate),test_evaluated=False,checkpoint_replayed=True,auditor_sha256=data.digest(__file__),rows=rows)
    train.save_json(output/"early_audit.json",result);ACTIVE.clear();print(json.dumps(result,indent=2),flush=True);return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);p.add_argument("--dataset",required=True);args=p.parse_args();audit(args.output,args.dataset)
