"""Audit and replay the fixed six-fit common-self-scale pilot."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from experiments.mechanism_retry.audit import audit
from experiments.mechanism_retry.context_scale import ContextScaleSwap
from experiments.mechanism_retry.model import LayerSwap
from experiments.mechanism_retry.run import read_safe
from graphcliff_pair.data import digest
from graphcliff_pair.train import ROOT, batches, evaluate, metrics, set_seed


def report(output):
    output = Path(output)
    checked = audit(output)
    manifest = json.loads((output / "manifest.json").read_text())
    cfg = manifest["config"]
    pre = json.loads((ROOT / "experiments/mechanism_retry/context_scale_preflight_20261008.json").read_text())
    path = ROOT / "experiments/mechanism_retry/context_scale_screen.json"
    if (manifest["config_sha256"] != pre["config_sha256"] or digest(path) != pre["config_sha256"]
            or json.loads(path.read_text()) != cfg or manifest["code_dirty"] or cfg["seeds"] != [42]
            or [a["name"] for a in cfg["arms"]] != ["self", "centered", "context"]):
        raise ValueError("not the fixed clean pilot")
    if str(torch.__version__) != manifest["torch"] or digest(ROOT / "docs/sources.json") != manifest["source_manifest_sha256"]:
        raise ValueError("environment or provenance changed")
    prior = json.loads((ROOT / "experiments/mechanism_retry/layer_results_20261007.json").read_text())
    historical = ROOT / "artifacts/mechanism_retry_layer_core_screen_20261007"
    historical_checked = audit(historical)
    old_manifest = json.loads((historical / "manifest.json").read_text())
    old_cfg, new_cfg = dict(old_manifest["config"]), dict(cfg)
    for key in ("arms", "status"):
        old_cfg.pop(key)
        new_cfg.pop(key)
    if old_cfg != new_cfg or any(old_manifest[k] != manifest[k] for k in ("torch", "pyg", "rdkit")):
        raise ValueError("repeated control protocol or environment changed")
    for key in ("source_sha256", "experiment_source_sha256"):
        if any(manifest[key].get(p) != h for p, h in old_manifest[key].items()):
            raise ValueError("historical training source changed")
    for relative, source in prior["inherited_dependencies"].items():
        if digest(ROOT / relative) != source["workspace_sha256"]:
            raise ValueError("inherited source changed")
    torch.set_num_threads(2)
    set_seed(SimpleNamespace(seed=42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, replays, decisions, control_repeats = [], [], [], []
    for source in pre["datasets"]:
        dataset = source["dataset"]
        if digest(source["input_path"]) != source["input_sha256"]:
            raise ValueError("input changed")
        frame, graphs, tr, va, tp, vp = read_safe(source["input_path"], cfg["split_seed"])
        pairs = json.loads((output / dataset / "pairs.json").read_text())
        if (tr.tolist() != pairs["train_rows"] or va.tolist() != pairs["valid_rows"] or tp != pairs["train_pairs"]
                or vp != pairs["valid_pairs"] or len(tp) != source["train_count"] or len(vp) != source["valid_count"]):
            raise ValueError("identity or coverage changed")
        summaries, initializations = {}, []
        for arm in cfg["arms"]:
            name = arm["name"]
            folder = output / dataset / "seed42" / name
            s = json.loads((folder / "summary.json").read_text())
            history = json.loads((folder / "history.json").read_text())
            table = pd.read_csv(folder / "validation_predictions.csv")
            init = json.loads((folder / "initialization.json").read_text())
            if (s["parameters"] != pre["parameters"] or init != s["initialization"]
                    or init["full_sha256"] != pre["initialization_sha256"] or len(history) != s["epochs_run"]
                    or s["epochs_run"] > cfg["epochs"] or s["best_epoch"] != min(history, key=lambda h: h["valid_mse"])["epoch"]
                    or s["train_graph_forwards"] != 2 * len(tp) * s["epochs_run"]):
                raise ValueError("initialization, parameters, or budget differs")
            initializations.append(init)
            if any(h["weight_min"] != 1 or h["weight_max"] != 1 for h in history):
                raise ValueError("not plain MSE")
            if table["query"].tolist() != [p["query"] for p in vp] or table["reference"].tolist() != [p["reference"] for p in vp]:
                raise ValueError("prediction pairing differs")
            for key, value in metrics(table.to_dict("records")).items():
                np.testing.assert_allclose(value, s[key], atol=1e-7)
            checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
            if checkpoint["epoch"] != s["best_epoch"]:
                raise ValueError("checkpoint epoch differs")
            model = (ContextScaleSwap(cfg["hidden_size"], cfg["num_layers"], cfg["heads"]) if name == "context"
                     else LayerSwap(arm["variant"], cfg["hidden_size"], cfg["num_layers"], cfg["heads"]))
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            actual = pd.DataFrame(evaluate(model, arm, vp, graphs, frame, cfg["batch_size"], device))
            np.testing.assert_array_equal(actual[["query", "reference", "cliff_mol"]], table[["query", "reference", "cliff_mol"]])
            np.testing.assert_allclose(actual[["y", "reference_y", "prediction"]], table[["y", "reference_y", "prediction"]], atol=2e-5, rtol=2e-5)
            q, r, *_ = next(batches(vp, graphs, frame, cfg["batch_size"], device))
            with torch.no_grad():
                exchange = float((model(q, r) + model(r, q)).abs().max())
                identical = float(model(q, q).abs().max())
            if max(exchange, identical) > 2e-5:
                raise ValueError("output contracts failed")
            replays.append(dict(dataset=dataset, arm=name, count=len(actual), checkpoint_sha256=digest(folder / "best.pt"),
                                max_abs_difference=float(np.max(np.abs(actual.prediction - table.prediction))),
                                exchange_max_abs_sum=exchange, identical_max_abs_delta=identical, contract_pairs_checked=len(q.ptr)-1))
            summaries[name] = s
            rows.append(dict(dataset=dataset, arm=name, overall_rmse=s["overall_rmse"], cliff_rmse=s["cliff_rmse"],
                             noncliff_rmse=s["noncliff_rmse"], best_epoch=s["best_epoch"], epochs_run=s["epochs_run"],
                             parameters=s["parameters"], fit_seconds=s["elapsed_seconds"], peak_cuda_mib=s["peak_cuda_mb"]))
            if name != "context":
                old_path = historical / dataset / "seed42" / name / "validation_predictions.csv"
                old = pd.read_csv(old_path)
                np.testing.assert_array_equal(old[["query", "reference", "y", "reference_y", "cliff_mol"]], table[["query", "reference", "y", "reference_y", "cliff_mol"]])
                old_metrics = metrics(old.to_dict("records"))
                control_repeats.append(dict(dataset=dataset, arm=name, historical_prediction_sha256=digest(old_path),
                                            max_abs_difference=float(np.max(np.abs(old.prediction - table.prediction))),
                                            mean_abs_prediction_difference=float(np.mean(np.abs(old.prediction - table.prediction))),
                                            old_overall_rmse=old_metrics["overall_rmse"], new_overall_rmse=s["overall_rmse"],
                                            old_cliff_rmse=old_metrics["cliff_rmse"], new_cliff_rmse=s["cliff_rmse"]))
            print("REPLAY_OK", dataset, name, flush=True)
            del model
        if any(i != initializations[0] for i in initializations):
            raise ValueError("three-arm initialization differs")
        comparisons = {n: {m: summaries["context"][m] < summaries[n][m] for m in ("overall_rmse", "cliff_rmse")}
                       for n in ("self", "centered")}
        decisions.append(dict(dataset=dataset, comparisons=comparisons, pass_all=all(v for d in comparisons.values() for v in d.values())))
    passed = all(d["pass_all"] for d in decisions)
    return dict(status="Go for further evidence only" if passed else "No-Go for expansion", primary_candidate="context",
                decisions=decisions, rows=rows, checkpoint_replays=replays, audit=checked, control_repeats=control_repeats,
                actual_fit_seconds=sum(r["fit_seconds"] for r in rows), source_commit=manifest["code_commit"],
                config_sha256=manifest["config_sha256"], test_evaluated=False, independent_confirmation=False,
                old_pilot_decision_changed=False, historical_control_audit=historical_checked,
                repeated_control_sources_hyperparameters_and_environment_equal=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--json-output", required=True)
    args = parser.parse_args()
    result = report(args.output)
    Path(args.json_output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(result["status"])
    for row in result["rows"]:
        print(row["dataset"], row["arm"], row["overall_rmse"], row["cliff_rmse"])
