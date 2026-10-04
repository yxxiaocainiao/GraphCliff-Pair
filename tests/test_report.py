import importlib.util
import unittest
from pathlib import Path

import pandas as pd

spec=importlib.util.spec_from_file_location('audit_runs',Path(__file__).resolve().parents[1]/'tools/audit_runs.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class ReportContracts(unittest.TestCase):
    def test_variant_head_compatibility_and_loss_controls(self):
        def key(arm): return ('task',42,arm)
        variants={key('global_fp'):'global_diff',key('global_fp_dynamic'):'global_diff',
                  key('cross_fp'):'cross_attention',key('pair_mlp_fp'):'pair_mlp'}
        values={k:dict(encoder_sha256='shared',head_sha256='base',full_sha256='global') for k in variants}
        values[key('pair_mlp_fp')]['head_sha256']='6H-mlp'
        values[key('cross_fp')]['full_sha256']='cross'
        module.check_initializations(values,variants)
        values[key('cross_fp')]['head_sha256']='changed'
        with self.assertRaisesRegex(AssertionError,'兼容 head'): module.check_initializations(values,variants)
        values[key('cross_fp')]['head_sha256']='base'
        values[key('global_fp_dynamic')]['full_sha256']='changed'
        with self.assertRaisesRegex(AssertionError,'Loss 对照'): module.check_initializations(values,variants)
        values[key('global_fp_dynamic')]['full_sha256']='global'
        values[key('pair_mlp_fp')]['encoder_sha256']='changed'
        with self.assertRaisesRegex(AssertionError,'编码器'): module.check_initializations(values,variants)

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
