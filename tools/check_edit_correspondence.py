"""Label-free bounded correspondence audit, plus mathematical counterexamples."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFMCS

ROOT = Path(__file__).resolve().parents[1]


def math_check():
    rng = np.random.default_rng(42)
    a, b, p = rng.normal(size=(5, 5)), rng.normal(size=(3, 3)), rng.normal(size=(5, 3))
    d = a @ p - p @ b
    errors = []
    for k in range(1, 5):
        left = np.linalg.matrix_power(a, k) @ p-p @ np.linalg.matrix_power(b, k)
        right = sum(np.linalg.matrix_power(a, j) @ d @ np.linalg.matrix_power(b, k-1-j) for j in range(k))
        np.testing.assert_allclose(left, right, atol=1e-10, rtol=1e-12)
        errors.append(float(np.max(np.abs(left-right))))
    q, r = np.array([0., 1., 3.]), np.array([1., 3., 0.])
    assert q.mean() == r.mean() and q.max() == r.max()
    delta = q-r
    nonlinear = float(np.mean(delta/(1+np.abs(delta))))
    assert delta.mean() == 0 and abs(nonlinear) > .1
    assert np.mean(-delta/(1+np.abs(delta))) == -nonlinear
    return dict(power_commutator_max_errors=errors, pooling_counterexample=nonlinear,
                note="algebra check only; not a trained model or chemical mechanism")


def main():
    old = ROOT / "artifacts/mechanism_retry_full_self_20261008/runs"
    records, decisions, inputs = [], [], []
    for dataset in ("CHEMBL234_Ki", "CHEMBL244_Ki"):
        path = Path("D:/GraphCliff-main/benchmark_data") / f"{dataset}.csv"
        data = pd.read_csv(path, usecols=["smiles", "split"])
        saved = old / dataset / "pairs.json"
        pairs = json.loads(saved.read_text())
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert sha == pairs["input_sha256"]
        inputs.append(dict(dataset=dataset, csv_sha256=sha, pairs_sha256=hashlib.sha256(saved.read_bytes()).hexdigest()))
        pool = {}
        for pair in pairs["train_pairs"]:
            q, r = pair["query"], pair["reference"]
            assert q in pairs["train_rows"] and r in pairs["train_rows"]
            assert data.at[q, "split"] == data.at[r, "split"] == "train"
            key = tuple(sorted((q, r)))
            if key not in pool:
                smiles = sorted(Chem.MolToSmiles(Chem.MolFromSmiles(data.at[i, "smiles"])) for i in key)
                pool[key] = hashlib.sha256("\0".join(smiles).encode()).hexdigest()
        selected = sorted(pool, key=lambda key: (pool[key], key))[:16]
        assert len(selected) == 16
        accepted = 0
        for q, r in selected:
            mols = [Chem.MolFromSmiles(data.at[i, "smiles"]) for i in (q, r)]
            parameters = rdFMCS.MCSParameters()
            parameters.Timeout, parameters.StoreAll = 1, True
            parameters.BondTyper = rdFMCS.BondCompare.CompareOrderExact
            parameters.AtomCompareParameters.MatchValences = True
            parameters.AtomCompareParameters.MatchFormalCharge = True
            parameters.AtomCompareParameters.MatchChiralTag = True
            parameters.AtomCompareParameters.RingMatchesRingOnly = True
            parameters.BondCompareParameters.RingMatchesRingOnly = True
            parameters.BondCompareParameters.CompleteRingsOnly = True
            parameters.BondCompareParameters.MatchStereo = True
            started = time.perf_counter()
            result = rdFMCS.FindMCS(mols, parameters)
            queries = result.degenerateSmartsQueryMolDict
            truncated, mappings = len(queries) > 8, set()
            if not result.canceled and not truncated:
                for query in queries.values():
                    matches = [m.GetSubstructMatches(query, uniquify=False, useChirality=True, maxMatches=17) for m in mols]
                    if any(len(x) >= 17 for x in matches):
                        truncated = True
                        break
                    for left in matches[0]:
                        for right in matches[1]:
                            mappings.add(tuple(sorted(zip(left, right))))
                            if len(mappings) > 64:
                                truncated = True
                                break
                        if truncated:
                            break
                    if truncated:
                        break
            fraction = result.numAtoms / max(m.GetNumAtoms() for m in mols)
            passed = not result.canceled and not truncated and fraction >= .7 and bool(mappings)
            accepted += passed
            records.append(dict(dataset=dataset, query=q, reference=r, sampling_sha256=pool[q, r],
                                atoms=[m.GetNumAtoms() for m in mols], common_atoms=result.numAtoms,
                                core_fraction=fraction, canceled=result.canceled, degenerate_queries=len(queries),
                                mapping_count_observed=len(mappings), enumeration_truncated=truncated,
                                accepted=passed, elapsed_seconds=time.perf_counter()-started))
            print("STRUCTURE", dataset, q, r, "accepted", passed, "maps", len(mappings), flush=True)
        decisions.append(dict(dataset=dataset, accepted=accepted, sampled=16, passed=accepted >= 12))
    result = dict(records=records, decisions=decisions, inputs=inputs, math=math_check(),
                  tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  label_columns_read=[], training_runs=0, test_evaluated=False,
                  limit="32 label-free train pairs only; not full-data mapping coverage or efficacy")
    destination = ROOT / "experiments/mechanism_retry/edit_correspondence_20261008.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(decisions, flush=True)


if __name__ == "__main__":
    main()
