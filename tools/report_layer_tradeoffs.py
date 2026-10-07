"""Post-hoc descriptive error decomposition; no fits or candidate selection."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.mechanism_retry.audit import audit
from graphcliff_pair.data import digest


def report(output):
    output = Path(output)
    checked = audit(output)
    manifest = json.loads((output / "manifest.json").read_text())
    cfg = manifest["config"]
    if cfg["seeds"] != [42] or [a["name"] for a in cfg["arms"]] != ["full", "self", "cross", "centered"]:
        raise ValueError("expected the completed four-arm pilot")
    comparisons, hashes = [], {}
    for dataset in cfg["datasets"]:
        tables = {}
        for name in ("full", "self", "cross", "centered"):
            path = output / dataset / "seed42" / name / "validation_predictions.csv"
            tables[name] = pd.read_csv(path)
            hashes[str(path.relative_to(output))] = digest(path)
        center = tables["centered"]
        cliff = center.cliff_mol.to_numpy().astype(bool)
        for name in ("full", "self", "cross"):
            base = tables[name]
            np.testing.assert_array_equal(base[["query", "reference", "y", "reference_y", "cliff_mol"]],
                                          center[["query", "reference", "y", "reference_y", "cliff_mol"]])
            error = base.y.to_numpy() - base.prediction.to_numpy()
            correction = center.prediction.to_numpy() - base.prediction.to_numpy()
            gain = error**2 - (error - correction)**2
            groups = {}
            for label, mask in (("overall", np.ones(len(base), dtype=bool)), ("cliff", cliff), ("noncliff", ~cliff)):
                alignment = float(2 * np.mean(error[mask] * correction[mask]))
                energy = float(np.mean(correction[mask]**2))
                actual = float(gain[mask].mean())
                np.testing.assert_allclose(actual, alignment - energy, atol=1e-12, rtol=0)
                groups[label] = dict(count=int(mask.sum()), mse_gain=actual,
                                     twice_error_correction_product=alignment, correction_energy=energy,
                                     improved_queries=int((gain[mask] > 0).sum()),
                                     worsened_queries=int((gain[mask] < 0).sum()), tied_queries=int((gain[mask] == 0).sum()))
            weighted = sum(groups[k]["count"] * groups[k]["mse_gain"] for k in ("cliff", "noncliff")) / len(base)
            np.testing.assert_allclose(weighted, groups["overall"]["mse_gain"], atol=1e-12, rtol=0)
            comparisons.append(dict(dataset=dataset, comparator=name, groups=groups,
                                    weighted_overall_mse_gain=weighted))
    return dict(analysis_date="2026-10-08", scope="post-hoc descriptive analysis; not independent confirmation",
                candidate="centered", source_commit=manifest["code_commit"], manifest_sha256=digest(output / "manifest.json"),
                prediction_sha256=hashes, comparisons=comparisons, audit=checked,
                additional_fits=0, test_evaluated=False, changes_original_gate=False,
                statistical_significance_claim=False, chemical_causality_claim=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--json-output", required=True)
    args = parser.parse_args()
    result = report(args.output)
    Path(args.json_output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for row in result["comparisons"]:
        print(row["dataset"], row["comparator"], {k: round(v["mse_gain"], 9) for k, v in row["groups"].items()})
