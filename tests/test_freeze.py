import importlib.util
import itertools
import json
import tempfile
import unittest
from pathlib import Path

spec=importlib.util.spec_from_file_location('freeze_validation',Path(__file__).resolve().parents[1]/'tools/freeze_validation.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

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
            for task,seed,arm in itertools.product(module.TASKS,module.SEEDS,module.FULL):
                run=root/'artifacts'/'stage'/task/f'seed{seed}'/arm
                run.mkdir(parents=True)
                (run/'best.pt').write_bytes(b'fixture-identity')
                manifest=run.parents[2]/'manifest.json';manifest.write_text('{}')
                pairs=run.parents[1]/'pairs.json';pairs.write_text('{}')
                models.append(dict(dataset=task,seed=seed,arm=arm,run_dir=module.relative(run,root),
                                   checkpoint_sha256=module.digest(run/'best.pt'),manifest_sha256=module.digest(manifest),pairs_sha256=module.digest(pairs)))
            freeze={'status':'validation_complete_frozen','runs':78,'test_evaluated':False,'report':'report.json','report_sha256':module.digest(report),'source_sha256':sources,'data_sha256':{d:'fixture-data-hash' for d in module.TASKS},'models':models}
            path=root/'freeze.json';path.write_text(json.dumps(freeze))
            self.assertEqual(module.validate_freeze(path,root)['runs'],78)
            (root/models[0]['run_dir']/'best.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError): module.validate_freeze(path,root)

if __name__=='__main__':
    unittest.main()
