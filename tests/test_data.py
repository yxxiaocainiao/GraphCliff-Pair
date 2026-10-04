import hashlib
import json
import unittest
from pathlib import Path

from rdkit import Chem
from graphcliff_pair.data import FP, top1

ROOT = Path(__file__).resolve().parents[1]

class DataContracts(unittest.TestCase):
    def test_source_integrity(self):
        sources = json.loads((ROOT / "docs/sources.json").read_text())
        for name, record in sources["graphcliff"]["files"].items():
            self.assertEqual(hashlib.sha256((ROOT / "graphcliff_pair/vendor" / name).read_bytes()).hexdigest(), record["sha256"])

    def test_self_duplicate_and_tie(self):
        mols = {i: Chem.MolFromSmiles(s) for i, s in enumerate(["CCO", "OCC", "CCN", "CCN"])}
        canonical = {i: Chem.MolToSmiles(m) for i, m in mols.items()}
        fps = {i: FP.GetFingerprint(m) for i, m in mols.items()}
        pair = top1([0], [3, 1, 2, 0], canonical, fps)[0]
        self.assertEqual(pair["reference"], 2)
        with self.assertRaises(ValueError):
            top1([0], [0, 1], canonical, fps)

if __name__ == "__main__":
    unittest.main()
