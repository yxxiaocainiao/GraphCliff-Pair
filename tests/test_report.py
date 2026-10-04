import importlib.util
import unittest
from pathlib import Path

import pandas as pd

spec=importlib.util.spec_from_file_location('audit_runs',Path(__file__).resolve().parents[1]/'tools/audit_runs.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class ReportContracts(unittest.TestCase):
    def test_signed_direction_and_insufficient_support(self):
        source=pd.DataFrame({'smiles':['CCO','OCC']})
        pred=pd.DataFrame({'query':[0,1],'y':[1.,3.],'prediction':[2.,4.],'cliff_mol':[0,1]})
        result=module.pair_metrics(pred,source)
        self.assertEqual(result['morgan_signed_delta_mae'],0)
        self.assertEqual(result['morgan_sign_accuracy'],1)
        self.assertIsNone(result['morgan_delta_spearman'])
        pred['prediction']=[4.,2.]
        result=module.pair_metrics(pred,source)
        self.assertEqual(result['morgan_signed_delta_mae'],4)
        self.assertEqual(result['morgan_sign_accuracy'],0)
        result=module.pair_metrics(pred.iloc[:1],source)
        self.assertEqual(result['morgan_valid_pair_count'],0)
        self.assertIsNone(result['morgan_signed_delta_mae'])

if __name__=='__main__':
    unittest.main()
