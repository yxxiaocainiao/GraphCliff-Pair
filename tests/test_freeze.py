import importlib.util
import itertools
import json
import tempfile
import unittest
from pathlib import Path

spec=importlib.util.spec_from_file_location('freeze_validation',Path(__file__).resolve().parents[1]/'tools/freeze_validation.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from replay_validation import TOLERANCES,run_identity

class FreezeContracts(unittest.TestCase):
    def test_portable_frozen_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            p=root/'a'/'b.json';p.parent.mkdir();p.write_text('{}')
            self.assertEqual(module.relative(p,root),'a/b.json')

    def test_missing_freeze_blocks_test(self):
        with self.assertRaises(ValueError): module.validate_freeze(Path('missing_validation_freeze.json'))

    def test_complete_identity_and_checkpoint_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            report=root/'report.json';report.write_text(json.dumps({'phase':'full','runs':78,'test_evaluated':False}))
            sources={}
            for name in module.REQUIRED_SOURCES:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture-source')
                sources[name]=module.digest(p)
            models=[]
            replays=[]
            core_sources={name:sha for name,sha in sources.items() if name.startswith('graphcliff_pair/') and name!='graphcliff_pair/predict.py'}
            for task,seed,arm in itertools.product(module.TASKS,module.SEEDS,module.FULL):
                run=root/'artifacts'/'stage'/task/f'seed{seed}'/arm
                run.mkdir(parents=True)
                (run/'best.pt').write_bytes(b'fixture-identity')
                manifest=run.parents[2]/'manifest.json';manifest.write_text(json.dumps({'source_sha256':core_sources,'device':'cpu'}))
                pairs=run.parents[1]/'pairs.json';pairs.write_text('{}')
                (run/'validation_predictions.csv').write_text('prediction\n1.0\n')
                (run/'history.json').write_text('[]')
                (run/'summary.json').write_text(json.dumps({'count':1,'best_epoch':1}))
                models.append(dict(dataset=task,seed=seed,arm=arm,run_dir=module.relative(run,root),
                                   checkpoint_sha256=module.digest(run/'best.pt'),manifest_sha256=module.digest(manifest),pairs_sha256=module.digest(pairs)))
                replays.append(dict(dataset=task,seed=seed,arm=arm,**run_identity(run,root),
                                    epoch=1,batch_size=1,count=1,status='matched',device='cpu',training_device='cpu',max_prediction_abs_error=0.,max_metric_abs_error=0.))
            freeze={'status':'validation_complete_frozen','runs':78,'test_evaluated':False,'report':'report.json','report_sha256':module.digest(report),'source_sha256':sources,'data_sha256':{d:'fixture-data-hash' for d in module.TASKS},'models':models}
            replay=root/'replay.json'
            replay.write_text(json.dumps({'status':'validation_replay_passed','runs':78,'test_evaluated':False,
                                          'tolerances':TOLERANCES,'training_source_sha256':core_sources,
                                          'data_sha256':freeze['data_sha256'],'models':replays,
                                          'replay_source_sha256':{'tools/replay_validation.py':sources['tools/replay_validation.py']}}))
            freeze['validation_replay']={'path':'replay.json','sha256':module.digest(replay)}
            path=root/'freeze.json';path.write_text(json.dumps(freeze))
            self.assertEqual(module.validate_freeze(path,root)['runs'],78)
            removed=freeze.pop('validation_replay')
            path.write_text(json.dumps(freeze))
            with self.assertRaisesRegex(ValueError,'重放证明'): module.validate_freeze(path,root)
            freeze['validation_replay']=removed
            path.write_text(json.dumps(freeze))
            (root/models[0]['run_dir']/'best.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError): module.validate_freeze(path,root)

if __name__=='__main__':
    unittest.main()
