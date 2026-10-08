"""Label-free operator feasibility; no model training or efficacy claims."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]


def operators(mol):
    types = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]
    adj = np.zeros((5, mol.GetNumAtoms(), mol.GetNumAtoms()))
    for bond in mol.GetBonds():
        category = 3 if bond.GetIsAromatic() else types.index(bond.GetBondType()) if bond.GetBondType() in types else 4
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[category, i, j] = adj[category, j, i] = 1
    degree = adj.sum(axis=(0, 1))
    inv = 1/np.sqrt(np.maximum(degree, 1))
    return adj*inv[None, :, None]*inv[None, None, :]


def check(ops):
    total = ops.sum(axis=0)
    polynomial = [np.linalg.matrix_power(total, k) for k in range(4)]
    bases, defects = [], []
    for b, c in itertools.combinations(range(5), 2):
        skew = (ops[b]@ops[c]-ops[c]@ops[b])/2
        np.testing.assert_allclose(skew.T, -skew, atol=1e-14, rtol=0)
        # Symmetric polynomial filters are Frobenius-orthogonal to this skew basis.
        errors = [abs(float(np.sum(skew*p))) for p in polynomial]
        assert max(errors) < 1e-10
        defects.append(max(errors))
        bases.append(skew)
    return bases, defects


def main():
    source = ROOT / "experiments/mechanism_retry/edit_correspondence_20261008.json"
    sampled = json.loads(source.read_text())
    records = []
    rng = np.random.default_rng(42)
    for dataset in ("CHEMBL234_Ki", "CHEMBL244_Ki"):
        path = Path("D:/GraphCliff-main/benchmark_data") / f"{dataset}.csv"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == next(x["csv_sha256"] for x in sampled["inputs"] if x["dataset"] == dataset)
        data = pd.read_csv(path, usecols=["smiles", "split"])
        rows = sorted({x[k] for x in sampled["records"] if x["dataset"] == dataset for k in ("query", "reference")})
        for row in rows:
            assert data.at[row, "split"] == "train"
            mol = Chem.MolFromSmiles(data.at[row, "smiles"])
            ops = operators(mol)
            bases, defects = check(ops)
            permutation = rng.permutation(mol.GetNumAtoms())
            permuted = operators(Chem.RenumberAtoms(mol, permutation.tolist()))
            np.testing.assert_allclose(permuted, ops[:, permutation][:, :, permutation], atol=1e-14, rtol=0)
            changed, _ = check(permuted)
            for a, b in zip(bases, changed):
                np.testing.assert_allclose(b, a[permutation][:, permutation], atol=1e-14, rtol=0)
            # RDKit aromatic flags survive a Kekule assignment; the new operator must too.
            kekule = Chem.Mol(mol)
            Chem.Kekulize(kekule, clearAromaticFlags=False)
            np.testing.assert_array_equal(operators(kekule), ops)
            norms = [float(np.linalg.norm(x)) for x in bases]
            records.append(dict(dataset=dataset, row=row, atoms=mol.GetNumAtoms(),
                                basis_frobenius_norms=norms, nonzero_bases=sum(x > 1e-12 for x in norms),
                                max_polynomial_inner_product=max(defects)))
        print(dataset, "molecules", len(rows), flush=True)
    result = dict(records=records, sampled_molecules=len(records), molecules_with_nonzero_basis=sum(x["nonzero_bases"] > 0 for x in records),
                  structural_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  node_permutation_checks_passed=True, kekule_flag_checks_passed=True,
                  training_runs=0, label_columns_read=[], test_evaluated=False,
                  limit="operator feasibility on a fixed train-only structural sample, not expressivity of the full nonlinear GNN or efficacy")
    (ROOT / "experiments/mechanism_retry/bond_commutator_20261008.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print("OPERATOR_OK", result["sampled_molecules"], result["molecules_with_nonzero_basis"], flush=True)


if __name__ == "__main__":
    main()
