"""Independent validation-only report for the fixed FP retry pilot."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from experiments.mechanism_retry.audit import audit
from experiments.mechanism_retry.model import ResidualFP, SingleBaseline
from experiments.mechanism_retry.run import read_safe
from graphcliff_pair.data import digest
from graphcliff_pair.fingerprint import membership
from graphcliff_pair.train import ROOT, batches, set_seed


def replay(output):
    """Reload every real checkpoint and re-export validation predictions in memory."""
    output = Path(output)
    cfg = json.loads((output / "manifest.json").read_text())["config"]
    preflight = json.loads((ROOT / "experiments/mechanism_retry/fp_preflight_20261007.json").read_text())
    torch.set_num_threads(2)
    set_seed(SimpleNamespace(seed=cfg["seeds"][0]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checked = []
    for source in preflight["datasets"]:
        frame, graphs, tr, va, tp, vp = read_safe(source["input_path"],cfg["split_seed"])
        pairs = json.loads((output / source["dataset"] / "pairs.json").read_text())
        if tr.tolist() != pairs["train_rows"] or va.tolist() != pairs["valid_rows"] or vp != pairs["valid_pairs"]:
            raise ValueError("replay split or pairing changed")
        for row in {p[k] for p in vp for k in ("query","reference")}:
            g = graphs[row]
            g.atom_fp = membership(SimpleNamespace(smiles=frame.at[row,"smiles"],x=g.x,edge_index=g.edge_index,edge_attr=g.edge_attr))
        for arm in cfg["arms"]:
            folder = output / source["dataset"] / "seed42" / arm["name"]
            model = (SingleBaseline(38,13,hidden_size=cfg["hidden_size"],num_layers=cfg["num_layers"],dropout=0)
                     if arm["name"] == "base" else ResidualFP(cfg["hidden_size"],cfg["num_layers"],arm["space"],arm.get("frozen",False),arm.get("shuffle",False)))
            checkpoint = torch.load(folder / "best.pt",map_location="cpu",weights_only=True)
            summary = json.loads((folder / "summary.json").read_text())
            if checkpoint["epoch"] != summary["best_epoch"]:
                raise ValueError("checkpoint selection epoch differs")
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device).eval()
            actual = []
            with torch.no_grad():
                for q,r,yq,yr,selected in batches(vp,graphs,frame,cfg["batch_size"],device):
                    prediction = model(q) if isinstance(model,ResidualFP) else model(q.x,q.edge_index,q.edge_attr,q.batch)
                    actual.extend(prediction.flatten().cpu().tolist())
            saved = pd.read_csv(folder / "validation_predictions.csv").prediction.to_numpy()
            np.testing.assert_allclose(actual,saved,atol=2e-5,rtol=2e-5)
            checked.append(dict(dataset=source["dataset"],arm=arm["name"],count=len(saved),
                                max_abs_difference=float(np.max(np.abs(np.array(actual)-saved)))))
            print("REPLAY_OK",source["dataset"],arm["name"],flush=True)
            del model
    return dict(device=str(device),checkpoints=len(checked),records=checked,test_evaluated=False)


def report(output):
    output = Path(output)
    checked = audit(output)
    manifest = json.loads((output / "manifest.json").read_text())
    cfg = manifest["config"]
    preflight = json.loads((ROOT / "experiments/mechanism_retry/fp_preflight_20261007.json").read_text())
    dependencies = {}
    for relative in ("experiments/long_branch_swap/model.py","experiments/long_branch_swap/run.py"):
        committed = subprocess.check_output(["git","show",manifest["code_commit"]+":"+relative],cwd=ROOT)
        current = (ROOT / relative).read_bytes()
        if committed.replace(b"\r\n",b"\n") != current.replace(b"\r\n",b"\n"):
            raise ValueError("inherited dependency differs from training commit")
        dependencies[relative] = dict(workspace_sha256=digest(ROOT / relative),
                                      commit_lf_sha256=hashlib.sha256(committed.replace(b"\r\n",b"\n")).hexdigest())
    if digest(ROOT / "docs/sources.json") != manifest["source_manifest_sha256"]:
        raise ValueError("source provenance manifest changed")
    if str(torch.__version__) != manifest["torch"]:
        raise ValueError("torch environment changed")
    if manifest["config_sha256"] != preflight["config_sha256"] or any(k.startswith("limit_") for k in cfg):
        raise ValueError("formal configuration differs from the execution plan")
    if cfg["seeds"] != [42] or cfg["datasets"] != [d["dataset"] for d in preflight["datasets"]]:
        raise ValueError("tasks or seeds differ")
    rows, decisions, decompositions = [], [], []
    for source in preflight["datasets"]:
        dataset = source["dataset"]
        if digest(source["input_path"]) != source["input_sha256"]:
            raise ValueError("input changed")
        pairs = json.loads((output / dataset / "pairs.json").read_text())
        if pairs["input_sha256"] != source["input_sha256"] or len(pairs["train_pairs"]) != source["train_count"] or len(pairs["valid_pairs"]) != source["valid_count"]:
            raise ValueError("incomplete training or validation coverage")
        summaries, tables = {}, {}
        for arm in cfg["arms"]:
            folder = output / dataset / "seed42" / arm["name"]
            s = json.loads((folder / "summary.json").read_text())
            table = pd.read_csv(folder / "validation_predictions.csv")
            if table["query"].tolist() != [p["query"] for p in pairs["valid_pairs"]] or s["cliff_count"] != source["valid_cliff_count"]:
                raise ValueError("validation identity or cliff coverage changed")
            if s["epochs_run"] > cfg["epochs"] or s["count"] != source["valid_count"]:
                raise ValueError("run budget or validation count differs")
            summaries[arm["name"]], tables[arm["name"]] = s, table
        base = summaries["base"]
        for name, s in summaries.items():
            rows.append(dict(dataset=dataset, arm=name, overall_rmse=s["overall_rmse"], cliff_rmse=s["cliff_rmse"],
                             noncliff_rmse=s["noncliff_rmse"], best_epoch=s["best_epoch"], epochs_run=s["epochs_run"],
                             overall_improvement_percent=100*(base["overall_rmse"]-s["overall_rmse"])/base["overall_rmse"],
                             cliff_improvement_percent=100*(base["cliff_rmse"]-s["cliff_rmse"])/base["cliff_rmse"],
                             trainable_parameters=s["parameters"], total_parameters=s.get("total_parameters",s["parameters"]),
                             stage_seconds=s["elapsed_seconds"], peak_cuda_mib=s["peak_cuda_mb"],
                             lifecycle_seconds=s["elapsed_seconds"]+(base["elapsed_seconds"] if s.get("frozen") else 0)))
        main = summaries["output_frozen"]
        comparisons = {name:{metric:main[metric] < summaries[name][metric] for metric in ("overall_rmse","cliff_rmse")}
                       for name in ("base","feature_frozen","output_frozen_null")}
        decisions.append(dict(dataset=dataset, strict_comparisons=comparisons,
                              pass_all=all(v for metrics in comparisons.values() for v in metrics.values())))
        baseline_table = tables["base"]
        for name in ("feature_frozen", "output_frozen", "output_frozen_null"):
            table = tables[name]
            np.testing.assert_array_equal(table[["query","reference","y","cliff_mol"]].to_numpy(),
                                          baseline_table[["query","reference","y","cliff_mol"]].to_numpy())
            e = table.y.to_numpy() - baseline_table.prediction.to_numpy()
            c = table.prediction.to_numpy() - baseline_table.prediction.to_numpy()
            cliff = table.cliff_mol.to_numpy().astype(bool)
            for subset, mask in (("overall",np.ones(len(e),dtype=bool)),("cliff",cliff),("noncliff",~cliff)):
                gain = float(2*np.mean(e[mask]*c[mask])-np.mean(c[mask]**2))
                actual = float(np.mean(e[mask]**2)-np.mean((e[mask]-c[mask])**2))
                np.testing.assert_allclose(gain,actual,atol=1e-12)
                decompositions.append(dict(dataset=dataset,arm=name,subset=subset,count=int(mask.sum()),
                                           base_mse_minus_candidate_mse=actual,
                                           twice_error_correction_product=float(2*np.mean(e[mask]*c[mask])),
                                           correction_energy=float(np.mean(c[mask]**2))))
    return dict(date="2026-10-07",status="Go for further validation" if all(d["pass_all"] for d in decisions) else "No-Go for expansion",
                primary_candidate="output_frozen",decisions=decisions,rows=rows,risk_decompositions=decompositions,
                actual_fit_seconds=sum(r["stage_seconds"] for r in rows),null_coverage=preflight["datasets"],audit=checked,
                source_commit=manifest["code_commit"],config_sha256=manifest["config_sha256"],inherited_dependencies=dependencies,
                test_evaluated=False,independent_confirmation=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--json-output",required=True)
    args = parser.parse_args()
    result = report(args.output)
    result["checkpoint_replay"] = replay(args.output)
    Path(args.json_output).write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(result["status"])
    for row in result["rows"]:
        print(row["dataset"],row["arm"],f"Overall={row['overall_rmse']:.6f} Cliff={row['cliff_rmse']:.6f}")
