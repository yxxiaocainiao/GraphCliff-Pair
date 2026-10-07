"""Check saved retry identities and the exact residual-risk decomposition."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from graphcliff_pair.data import digest
from graphcliff_pair.train import ROOT


def audit(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    cfg = manifest["config"]
    completed = json.loads((output / "completed.json").read_text())
    expected = len(cfg["datasets"]) * len(cfg["seeds"]) * len(cfg["arms"])
    if completed["runs"] != expected or completed["test_evaluated"] or manifest["test_evaluated"]:
        raise ValueError("incomplete or test-contaminated run")
    hashes = manifest["source_sha256"] | manifest["experiment_source_sha256"]
    if any(digest(ROOT / p) != h for p,h in hashes.items()):
        raise ValueError("training source changed")
    if digest(ROOT / "external/fppool/pooling.py") != manifest["fppool_external_sha256"]:
        raise ValueError("external source changed")
    records, decompositions, frozen_tensors = [], [], 0
    for dataset in cfg["datasets"]:
        for seed in cfg["seeds"]:
            folder = output / dataset / f"seed{seed}"
            tables = {}
            for arm in cfg["arms"]:
                sub = folder / arm["name"]
                table = pd.read_csv(sub / "validation_predictions.csv").set_index("query")
                summary = json.loads((sub / "summary.json").read_text())
                error = table.prediction.to_numpy() - table.y.to_numpy()
                if not np.isfinite(error).all() or len(error) != summary["count"] or summary["test_evaluated"]:
                    raise ValueError("invalid predictions")
                np.testing.assert_allclose(np.mean(error**2)**.5, summary["overall_rmse"], atol=1e-7)
                if summary["cliff_count"]:
                    mask = table.cliff_mol.to_numpy().astype(bool)
                    np.testing.assert_allclose(np.mean(error[mask]**2)**.5, summary["cliff_rmse"], atol=1e-7)
                tables[arm["name"]] = table
                records.append(dict(dataset=dataset, seed=seed, arm=arm["name"], count=len(error)))
                if arm.get("frozen"):
                    baseline_path = folder / "base/best.pt"
                    if digest(baseline_path) != summary["baseline_checkpoint_sha256"]:
                        raise ValueError("baseline checkpoint changed")
                    baseline = torch.load(baseline_path, map_location="cpu", weights_only=True)["model_state_dict"]
                    trained = torch.load(sub / "best.pt", map_location="cpu", weights_only=True)["model_state_dict"]
                    for key, value in baseline.items():
                        target = key.replace("reg_head.", "head.").replace("head.4.", "head.3.")
                        torch.testing.assert_close(value, trained[target], atol=0, rtol=0)
                        frozen_tensors += 1
                    base = tables["base"]
                    if not table.index.equals(base.index):
                        raise ValueError("query identity differs")
                    np.testing.assert_array_equal(table.y, base.y)
                    e = table.y.to_numpy() - base.prediction.to_numpy()
                    c = table.prediction.to_numpy() - base.prediction.to_numpy()
                    difference = float(np.mean((e-c)**2) - np.mean(e**2))
                    expansion = float(np.mean(c**2) - 2*np.mean(e*c))
                    np.testing.assert_allclose(difference, expansion, atol=1e-12)
                    decompositions.append(dict(dataset=dataset, seed=seed, arm=arm["name"],
                                               mse_difference=difference, expansion=expansion))
    if len(records) != expected:
        raise ValueError("run count mismatch")
    return dict(output=str(output.resolve()), runs=expected, source_hashes_checked=len(hashes),
                frozen_tensors_checked=frozen_tensors, residual_risk_checks=decompositions,
                test_evaluated=False, performance_claim=False, records=records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    args = parser.parse_args()
    print(json.dumps(audit(args.output), indent=2, allow_nan=False))
