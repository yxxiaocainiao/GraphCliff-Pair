import importlib.util
import unittest
from pathlib import Path

spec=importlib.util.spec_from_file_location('summary_results',Path(__file__).resolve().parents[1]/'tools/summarize_results.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def fixture(arms):
    records=[]
    for task in module.TASKS:
        for seed in module.SEEDS:
            for arm in arms:
                value=.95 if arm=='cross' else 1.
                records.append(dict(dataset=task,seed=seed,arm=arm,overall_rmse=value,cliff_rmse=value,noncliff_rmse=value,mae=value,
                                    parameters=100,epochs_run=20,optimizer_steps=200,train_graph_forwards=400))
    return {'records':records,'test_evaluated':False}

class SummaryContracts(unittest.TestCase):
    def test_paired_gate_and_missing_matrix(self):
        audit=fixture(module.INTERACTION)
        result=module.analyze(audit,'interaction')
        self.assertEqual(result['runs'],24)
        self.assertTrue(result['interaction_expansion_gate']['go'])
        self.assertEqual(result['comparisons'][0]['cliff_rmse']['improved_seeds'],3)
        for row in audit['records']:
            if row['arm']=='cross': row['overall_rmse']=1.02
        self.assertFalse(module.analyze(audit,'interaction')['interaction_expansion_gate']['go'])
        audit['records'].pop()
        with self.assertRaises(ValueError): module.analyze(audit,'interaction')

    def test_complete_factorial_and_static_controls(self):
        result=module.analyze(fixture(module.FULL),'full')
        self.assertEqual(result['runs'],78)
        self.assertEqual(result['summary'][0]['optimizer_steps']['mean'],200)
        self.assertEqual(result['summary'][0]['train_graph_forwards']['n'],3)
        labels=[r['comparison'] for r in result['comparisons']]
        self.assertEqual(sum(s.startswith('attention_conditional') for s in labels),8)
        self.assertEqual(sum(s.startswith('fppool_conditional') for s in labels),8)
        self.assertEqual(sum(s.startswith('dynamic_conditional') for s in labels),8)
        self.assertIn('dynamic_vs_static_cross_fp',labels)
        self.assertIn('attention_vs_mlp_fp',labels)

if __name__=='__main__':
    unittest.main()
