"""Audit eight fits, independently replay checkpoints, apply the frozen gate."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphcliff_pair import train
import graphcliff_pair.model as shared
from experiments.mechanism_retry.audit import audit
from experiments.mechanism_retry.run import read_safe
from experiments.mechanism_retry.model import LayerSwap
from experiments.mechanism_retry.self_anchor import SelfAnchorSwap
from experiments.mechanism_retry.repro import stable_max_pool
from tools.run_self_anchor_pilot import CONFIG, PROTOCOL, PREFLIGHT


def main(output):
    torch = train.torch
    torch.set_num_threads(2)
    output = Path(output)
    suite = json.loads((output / "suite.json").read_text())
    done = json.loads((output / "completed.json").read_text())
    assert done["completed_fits"] == 8 and not done["test_evaluated"]
    assert all(train.digest(train.ROOT / p) == digest for p, digest in suite["source_sha256"].items())
    cfg = json.loads(CONFIG.read_text())
    preflight = json.loads(PREFLIGHT.read_text())
    frame, graphs, _, _, _, valid = read_safe("D:/GraphCliff-main/benchmark_data/CHEMBL234_Ki.csv", 42)
    rows, heads, identities, initializations = [], [], [], {}
    old_pool = shared.global_max_pool
    shared.global_max_pool = stable_max_pool
    try:
        for repeat in (1, 2):
            root = output / f"repeat{repeat}"
            checked = audit(root)
            assert checked["runs"] == 4
            manifest = json.loads((root / "manifest.json").read_text())
            assert manifest["config"] == cfg and not manifest["code_dirty"] and not manifest["test_evaluated"]
            assert manifest["protocol_sha256"] == train.digest(PROTOCOL)
            assert manifest["entrypoint_sha256"] == train.digest(train.ROOT / "tools/run_self_anchor_pilot.py")
            assert manifest["preflight_sha256"] == train.digest(PREFLIGHT)
            assert manifest["code_commit"] == suite["code_commit"]
            for arm in cfg["arms"]:
                folder = root / "CHEMBL234_Ki/seed42" / arm["name"]
                summary = json.loads((folder / "summary.json").read_text())
                history = json.loads((folder / "history.json").read_text())
                assert summary["epochs_run"] == len(history) == 20
                assert summary["parameters"] == next(x["parameters"] for x in preflight["records"] if x["arm"] == arm["name"])
                assert summary["best_epoch"] == min(history, key=lambda x: x["valid_mse"])["epoch"]
                initializations.setdefault(arm["name"], []).append(summary["initialization"]["full_sha256"])
                heads.append(summary["initialization"]["head_sha256"])
                variant = arm["variant"]
                train.set_seed(SimpleNamespace(seed=42))
                torch.use_deterministic_algorithms(True, warn_only=False)
                model = (SelfAnchorSwap(anchor=variant == "retry_anchor") if variant in ("retry_anchor", "retry_gated_delta")
                         else LayerSwap(variant)).cuda()
                checkpoint = torch.load(folder / "best.pt", map_location="cuda", weights_only=True)
                assert checkpoint["epoch"] == summary["best_epoch"]
                model.load_state_dict(checkpoint["model_state_dict"])
                replay = train.pd.DataFrame(train.evaluate(model, arm, valid, graphs, frame, 32, torch.device("cuda")))
                saved = train.pd.read_csv(folder / "validation_predictions.csv")
                for key in ("query", "reference", "y", "reference_y", "cliff_mol"):
                    train.np.testing.assert_allclose(replay[key], saved[key], atol=1e-12, rtol=0)
                error = float(train.np.max(train.np.abs(replay.prediction.to_numpy()-saved.prediction.to_numpy())))
                assert error < 1e-5, (arm["name"], error)
                identities.append(saved[["query", "reference", "y", "reference_y", "cliff_mol"]].to_dict("records"))
                gain = getattr(model, "innovation_gain", None)
                rows.append(dict(repeat=repeat, arm=arm["name"], overall_rmse=summary["overall_rmse"],
                                 cliff_rmse=summary["cliff_rmse"], noncliff_rmse=summary["noncliff_rmse"],
                                 best_epoch=summary["best_epoch"], epochs_run=20, parameters=summary["parameters"],
                                 elapsed_seconds=summary["elapsed_seconds"], peak_cuda_mb=summary["peak_cuda_mb"],
                                 replay_max_abs_error=error, checkpoint_sha256=train.digest(folder / "best.pt"),
                                 gain=None if gain is None else gain.detach().cpu().tolist(),
                                 effective_gain=None if gain is None else gain.tanh().detach().cpu().tolist()))
                del model, checkpoint
    finally:
        shared.global_max_pool = old_pool
    assert len(set(heads)) == 1 and all(len(set(v)) == 1 for v in initializations.values())
    assert initializations["anchor"] == initializations["gated_delta"]
    assert all(x == identities[0] for x in identities)
    comparisons = []
    for control in ("full", "self", "gated_delta"):
        for metric in ("overall_rmse", "cliff_rmse"):
            worst = max(x[metric] for x in rows if x["arm"] == "anchor")
            best = min(x[metric] for x in rows if x["arm"] == control)
            comparisons.append(dict(control=control, metric=metric, anchor_worst=worst, control_best=best, passed=worst < best))
    result = dict(status="exploratory Go" if all(x["passed"] for x in comparisons) else "No-Go for expansion",
                  source_commit=suite["code_commit"], suite_sha256=train.digest(output / "suite.json"),
                  protocol_sha256=train.digest(PROTOCOL), preflight_sha256=train.digest(PREFLIGHT),
                  reporter_sha256=train.digest(__file__), rows=rows, comparisons=comparisons,
                  source_files_verified=len(suite["source_sha256"]), formal_fits=8, train_epochs=160,
                  optimizer_steps=8*20*83, independent_checkpoint_replays=8, validation_predictions_replayed=8*292,
                  elapsed_seconds=sum(x["elapsed_seconds"] for x in rows), test_evaluated=False,
                  statistical_significance_claim=False, historical_NoGo_changed=False,
                  limit="one reused development task, seed42, two same-seed repeats, 20epoch budget; not independent confirmation")
    train.save_json(train.ROOT / "experiments/mechanism_retry/self_anchor_results_20261008.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    main(parser.parse_args().output)
