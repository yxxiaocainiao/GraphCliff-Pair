import importlib.util
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd

spec=importlib.util.spec_from_file_location('test_summary_tool',Path(__file__).resolve().parents[1]/'tools/summarize_test.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from summarize_results import TASKS,SEEDS,FULL

class FinalSummaryContracts(unittest.TestCase):
    def test_full_test_matrix_does_not_select_or_expand(self):
        rows=[dict(dataset=d,seed=s,arm=a,test_evaluated=True,overall_rmse=1.,cliff_rmse=1.,noncliff_rmse=1.,mae=1.)
              for d,s,a in itertools.product(TASKS,SEEDS,FULL)]
        result=module.summarize(rows,'synthetic-freeze')
        self.assertEqual(result['runs'],78)
        self.assertTrue(result['test_evaluated'])
        self.assertFalse(result['model_selection_on_test'])
        self.assertNotIn('interaction_expansion_gate',result)
        with self.assertRaises(ValueError): module.summarize(rows[:-1],'synthetic-freeze')
        rows[0]['test_evaluated']=False
        with self.assertRaises(ValueError): module.summarize(rows,'synthetic-freeze')

    def test_recompute_rejects_metric_tampering(self):
        # 合成78条结果只验证报告核对流程；冻结权重验证另有契约测试。
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'test';output.mkdir()
            freeze=root/'freeze.json';freeze.write_text('{}')
            models=[];records=[];data_hashes={}
            for task in TASKS:
                source=root/f'{task}.csv'
                pd.DataFrame({'smiles':['CCC','CCN','CCO'],'split':['train','test','test'],
                              'y':[99.,1.,2.],'cliff_mol':[0,0,1]}).to_csv(source,index=False)
                data_hashes[task]=module.digest(source)
            for task,seed,arm in itertools.product(TASKS,SEEDS,FULL):
                path=output/f'{task}_seed{seed}_{arm}_predictions.csv'
                pd.DataFrame({'query_index':[0,1],'smiles':['CCN','CCO'],'prediction':[1.5,2.5]}).to_csv(path,index=False)
                models.append(dict(dataset=task,seed=seed,arm=arm,checkpoint_sha256='synthetic'))
                records.append(dict(dataset=task,seed=seed,arm=arm,test_evaluated=True,
                                    checkpoint_sha256='synthetic',predictions_sha256=module.digest(path),
                                    **module.metrics([dict(prediction=1.5,y=1.,cliff_mol=0),dict(prediction=2.5,y=2.,cliff_mol=1)])))
            (output/'completed.json').write_text(json.dumps(dict(runs=78,test_evaluated=True,model_selection_on_test=False)))
            report=dict(records=records,freeze_sha256=module.digest(freeze),test_evaluated=True,model_selection_on_test=False)
            (output/'test_metrics.json').write_text(json.dumps(report))
            with patch.object(module,'validate_freeze',return_value=dict(models=models,data_sha256=data_hashes)):
                module.run(freeze,output,root,root/'verified')
                self.assertTrue((root/'verified.md').is_file())
                records[0]['cliff_rmse']=.1
                (output/'test_metrics.json').write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError,'测试指标不一致'):
                    module.run(freeze,output,root,root/'tampered')

if __name__=='__main__': unittest.main()
