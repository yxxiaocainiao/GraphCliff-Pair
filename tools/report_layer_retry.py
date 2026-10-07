"""Audit the fixed four-arm layer pilot and replay each selected checkpoint."""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from experiments.mechanism_retry.audit import audit
from experiments.mechanism_retry.model import LayerSwap
from experiments.mechanism_retry.run import read_safe
from graphcliff_pair.data import digest
from graphcliff_pair.train import ROOT, batches, set_seed


def report(output):
    output = Path(output)
    checked = audit(output)
    manifest = json.loads((output / "manifest.json").read_text())
    cfg = manifest["config"]
    preflight = json.loads((ROOT / "experiments/mechanism_retry/layer_preflight_20261007.json").read_text())
    config_path = ROOT / "experiments/mechanism_retry/layer_core_screen.json"
    if (manifest["config_sha256"] != preflight["core_config_sha256"] or digest(config_path) != manifest["config_sha256"]
            or json.loads(config_path.read_text()) != cfg or any(k.startswith("limit_") for k in cfg) or manifest["code_dirty"]):
        raise ValueError("configuration differs from fixed core pilot")
    if cfg["seeds"] != [42] or [a["name"] for a in cfg["arms"]] != ["full", "self", "cross", "centered"]:
        raise ValueError("unexpected arms or seeds")
    if cfg["datasets"] != [s["dataset"] for s in preflight["datasets"]]:
        raise ValueError("tasks differ from preflight")
    if str(torch.__version__) != manifest["torch"] or digest(ROOT / "docs/sources.json") != manifest["source_manifest_sha256"]:
        raise ValueError("environment or provenance changed")
    dependencies = {}
    for relative in ("experiments/long_branch_swap/model.py", "experiments/long_branch_swap/run.py"):
        committed = subprocess.check_output(["git", "show", manifest["code_commit"] + ":" + relative], cwd=ROOT)
        current = (ROOT / relative).read_bytes()
        if committed.replace(b"\r\n", b"\n") != current.replace(b"\r\n", b"\n"):
            raise ValueError("inherited dependency changed")
        dependencies[relative] = dict(workspace_sha256=digest(ROOT / relative),
                                      commit_lf_sha256=hashlib.sha256(committed.replace(b"\r\n", b"\n")).hexdigest())
    torch.set_num_threads(2)
    set_seed(SimpleNamespace(seed=42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = LayerSwap("retry_centered", cfg["hidden_size"], cfg["num_layers"], cfg["heads"]).encoder.layers[0].long
    zero = torch.zeros(1, cfg["hidden_size"], requires_grad=True)
    innovation = LayerSwap.transform(layer, zero) - LayerSwap.transform(layer, torch.zeros_like(zero))
    derivative = torch.autograd.grad(innovation[0, 0], zero)[0][0, 0].item()
    eps = torch.finfo(zero.dtype).eps if layer.norm.eps is None else layer.norm.eps
    np.testing.assert_allclose(derivative, .5 / math.sqrt(eps), rtol=1e-6)
    torch.testing.assert_close(innovation, torch.zeros_like(innovation), atol=0, rtol=0)
    analytic_check = dict(scope="initialization only; not failure attribution", effective_eps=eps,
                          local_diagonal_derivative=derivative, predicted_derivative=.5 / math.sqrt(eps),
                          zero_innovation_max_abs=0., optimization_steps=0)
    del layer, zero, innovation
    rows, decisions, replay = [], [], []
    for source in preflight["datasets"]:
        dataset = source["dataset"]
        if digest(source["input_path"]) != source["input_sha256"]:
            raise ValueError("input changed")
        frame, graphs, tr, va, tp, vp = read_safe(source["input_path"], cfg["split_seed"])
        pairs = json.loads((output / dataset / "pairs.json").read_text())
        if (pairs["input_sha256"] != source["input_sha256"] or tr.tolist() != pairs["train_rows"]
                or va.tolist() != pairs["valid_rows"] or tp != pairs["train_pairs"] or vp != pairs["valid_pairs"]
                or len(tp) != source["train_count"] or len(vp) != source["valid_count"]):
            raise ValueError("split, pairing, or coverage differs")
        summaries, initializations, identity = {}, {}, None
        for arm in cfg["arms"]:
            name = arm["name"]
            folder = output / dataset / "seed42" / name
            s = json.loads((folder / "summary.json").read_text())
            table = pd.read_csv(folder / "validation_predictions.csv")
            history = json.loads((folder / "history.json").read_text())
            initializations[name] = json.loads((folder / "initialization.json").read_text())
            expected_init = next(r["init_sha256"] for r in preflight["large_real_batch_checks"] if r["arm"] == name)
            if s["initialization"] != initializations[name] or initializations[name]["full_sha256"] != expected_init:
                raise ValueError("initialization differs from execution preflight")
            if (s["parameters"] != preflight["parameters"][name] or s["epochs_run"] > cfg["epochs"]
                    or len(history) != s["epochs_run"] or s["best_epoch"] != min(history, key=lambda h: h["valid_mse"])["epoch"]
                    or s["cliff_count"] != source["valid_cliff_count"]
                    or s["train_graph_forwards"] != 2 * len(tp) * s["epochs_run"]):
                raise ValueError("parameters, budget, or selection differs")
            if any(h["weight_min"] != 1 or h["weight_max"] != 1 for h in history):
                raise ValueError("not plain MSE")
            if table["query"].tolist() != [p["query"] for p in vp] or table["reference"].tolist() != [p["reference"] for p in vp]:
                raise ValueError("saved query/reference identity differs")
            ids = table[["query", "reference", "y", "reference_y", "cliff_mol"]].to_numpy()
            if identity is None:
                identity = ids
            np.testing.assert_array_equal(ids, identity)
            for column, ids_column in (("y", "query"), ("reference_y", "reference")):
                np.testing.assert_allclose(table[column], frame.loc[table[ids_column], "y"].to_numpy(), atol=1e-6)
            np.testing.assert_array_equal(table.cliff_mol, frame.loc[table["query"], "cliff_mol"].to_numpy())
            mask = ~table.cliff_mol.to_numpy().astype(bool)
            np.testing.assert_allclose(np.mean((table.prediction.to_numpy()[mask] - table.y.to_numpy()[mask])**2)**.5,
                                       s["noncliff_rmse"], atol=1e-7)
            model = LayerSwap(arm["variant"], cfg["hidden_size"], cfg["num_layers"], cfg["heads"], arm["readout"])
            checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
            if checkpoint["epoch"] != s["best_epoch"]:
                raise ValueError("checkpoint epoch differs")
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device).eval()
            actual, exchange_error, identical_error = [], 0., 0.
            with torch.no_grad():
                for index, (q, r, yq, yr, selected) in enumerate(batches(vp, graphs, frame, cfg["batch_size"], device)):
                    delta = model(q, r)
                    actual.extend((delta + yr).flatten().cpu().tolist())
                    if index == 0:
                        exchange_error = float((delta + model(r, q)).abs().max())
                        identical_error = float(model(q, q).abs().max())
            np.testing.assert_allclose(actual, table.prediction, atol=2e-5, rtol=2e-5)
            if exchange_error > 2e-5 or identical_error > 2e-5:
                raise ValueError("eval output contract failed")
            replay.append(dict(dataset=dataset, arm=name, count=len(actual),
                               max_abs_difference=float(np.max(np.abs(np.array(actual) - table.prediction.to_numpy()))),
                               exchange_max_abs_sum=exchange_error, identical_max_abs_delta=identical_error,
                               contract_pairs_checked=min(cfg["batch_size"], len(vp))))
            summaries[name] = s
            calls = {"full": 0, "self": 2, "cross": 2, "centered": 4}[name] * cfg["num_layers"]
            rows.append(dict(dataset=dataset, arm=name, overall_rmse=s["overall_rmse"], cliff_rmse=s["cliff_rmse"],
                             noncliff_rmse=s["noncliff_rmse"], best_epoch=s["best_epoch"], epochs_run=s["epochs_run"],
                             parameters=s["parameters"], fit_seconds=s["elapsed_seconds"], peak_cuda_mib=s["peak_cuda_mb"],
                             train_graph_forwards=s["train_graph_forwards"], attention_calls_per_batch=calls,
                             training_attention_calls=calls * math.ceil(len(tp) / cfg["batch_size"]) * s["epochs_run"]))
            print("REPLAY_OK", dataset, name, flush=True)
            del model
        if any(initializations[n] != initializations["self"] for n in ("cross", "centered")):
            raise ValueError("attention arm initialization differs")
        if any(initializations[n]["head_sha256"] != initializations["full"]["head_sha256"] for n in ("self", "cross", "centered")):
            raise ValueError("shared head initialization differs")
        comparisons = {n: {m: summaries["centered"][m] < summaries[n][m] for m in ("overall_rmse", "cliff_rmse")}
                       for n in ("full", "self", "cross")}
        decisions.append(dict(dataset=dataset, strict_comparisons=comparisons,
                              pass_all=all(v for metrics in comparisons.values() for v in metrics.values())))
        for row in rows:
            if row["dataset"] == dataset:
                for metric in ("overall_rmse", "cliff_rmse"):
                    base = summaries["full"][metric]
                    row[metric.removesuffix("_rmse") + "_improvement_percent_vs_full"] = 100 * (base - row[metric]) / base
    passed = all(d["pass_all"] for d in decisions)
    return dict(date="2026-10-07", status="Go for FP ablation only" if passed else "No-Go for expansion",
                primary_candidate="centered", optional_fp_ablation_allowed=passed, decisions=decisions, rows=rows,
                actual_fit_seconds=sum(r["fit_seconds"] for r in rows), audit=checked,
                checkpoint_replay=dict(device=str(device), checkpoints=len(replay), records=replay),
                source_commit=manifest["code_commit"], config_sha256=manifest["config_sha256"],
                inherited_dependencies=dependencies, attention_initialization_equal=True,
                head_initialization_equal=True, centered_local_derivative_check=analytic_check,
                test_evaluated=False, independent_confirmation=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--json-output", required=True)
    args = parser.parse_args()
    result = report(args.output)
    Path(args.json_output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(result["status"])
    for row in result["rows"]:
        print(row["dataset"], row["arm"], f"Overall={row['overall_rmse']:.6f} Cliff={row['cliff_rmse']:.6f}")
