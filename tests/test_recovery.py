import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch

spec=importlib.util.spec_from_file_location('recover_validation',Path(__file__).resolve().parents[1]/'tools/recover_validation.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RecoveryContracts(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve()
        self.source=self.root/'source';self.source.mkdir()
        self.config=dict(datasets=['toy'],seeds=[42],arms=[dict(name='done'),dict(name='interrupted')])
        self.config_path=self.root/'config.json';self.config_path.write_text(json.dumps(self.config))
        self.frame=pd.DataFrame(dict(y=[1.,2.,3.],cliff_mol=[0,1,0]))
        self.csv=self.root/'toy.csv';self.frame.to_csv(self.csv,index=False)
        self.pairs=[dict(query=1,reference=0,similarity=.5),dict(query=2,reference=0,similarity=.6)]
        self.base={'encoder.weight':torch.ones(1)}
        self.initialization=dict(encoder_sha256=module.train.state_hash(self.base))
        records=[dict(**p,y=self.frame.at[p['query'],'y'],reference_y=1.,prediction=value,
                      cliff_mol=int(self.frame.at[p['query'],'cliff_mol'])) for p,value in zip(self.pairs,[2.5,2.8])]
        self.summary=dict(**module.train.metrics(records),arm='done',seed=42,best_epoch=1,epochs_run=1,
                          test_evaluated=False,initialization=self.initialization)
        done=self.source/'toy/seed42/done';done.mkdir(parents=True)
        torch.save(dict(epoch=1,model_state_dict={}),done/'best.pt')
        for name,value in [('summary.json',self.summary),('initialization.json',self.initialization),
                           ('history.json',[dict(epoch=1,valid_mse=self.summary['overall_rmse']**2)])]:
            (done/name).write_text(json.dumps(value))
        pd.DataFrame(records).to_csv(done/'validation_predictions.csv',index=False)
        (self.source/'summary.json').write_text(json.dumps([dict(self.summary,dataset='toy')]))
        (self.source/'manifest.json').write_text(json.dumps(dict(config=self.config,config_sha256=module.digest(self.config_path),device='cpu',code_commit='synthetic')))
        (done.parents[1]/'pairs.json').write_text(json.dumps(dict(train_pairs=self.pairs,valid_pairs=self.pairs,
            input_sha256=module.digest(self.csv),scale=1.)))
        partial=self.source/'toy/seed42/interrupted';partial.mkdir()
        torch.save(dict(epoch=1,model_state_dict={}),partial/'best.pt')
        self.done=done

    def test_only_complete_records_reused_and_partial_retained(self):
        completed,expected=module.completed_models(self.source,self.config)
        self.assertEqual(set(completed),{('toy',42,'done')})
        self.assertEqual(len(expected),2)
        output=self.root/'recovered'
        calls=[]

        def fresh(config,arm,seed,frame,graphs,tp,vp,base,destination,device,scale):
            calls.append(arm['name'])
            destination.mkdir(parents=True)
            return dict(self.summary,arm=arm['name'])

        def pipeline(config_path,csv_root,out):
            out.mkdir()
            for arm in self.config['arms']:
                destination=out/'toy'/f'seed42'/arm['name']
                result=module.train.train_arm(self.config,arm,42,self.frame,{},self.pairs,self.pairs,self.base,destination,torch.device('cpu'),1.)
                self.assertEqual(result['arm'],arm['name'])

        hashes={name:module.digest(self.done/name) for name in module.MODEL_FILES}
        with patch.object(module,'ROOT',self.root),patch.object(module,'check_training_sources'),patch.object(module.torch.cuda,'is_available',return_value=False),patch.object(module.train,'train_arm',side_effect=fresh),patch.object(module.train,'run',side_effect=pipeline):
            module.recover(self.source,self.config_path,self.root,output)
        self.assertEqual(calls,['interrupted'])
        for name,sha in hashes.items():
            self.assertEqual(module.digest(output/'toy/seed42/done'/name),sha)
            self.assertEqual(module.digest(self.done/name),sha)
        self.assertTrue((self.source/'toy/seed42/interrupted/best.pt').is_file())
        evidence=json.loads((output/'recovery.json').read_text())
        self.assertEqual(len(evidence['completed_copied']),1)
        self.assertEqual(evidence['restarted_from_initialization'][0]['arm'],'interrupted')

    def test_completed_prediction_tamper_rejected(self):
        path=self.done/'validation_predictions.csv'
        values=pd.read_csv(path);values.loc[0,'prediction']+=1.;values.to_csv(path,index=False)
        with self.assertRaisesRegex(AssertionError,'已完成模型'):
            module.completed_models(self.source,self.config)

    def test_changed_configuration_or_existing_output_rejected(self):
        config=self.root/'changed.json';config.write_text(json.dumps(dict(self.config,seeds=[43])))
        with patch.object(module,'ROOT',self.root):
            with self.assertRaisesRegex(ValueError,'原冻结配置'):
                module.recover(self.source,config,self.root,self.root/'new')
            existing=self.root/'existing';existing.mkdir()
            with self.assertRaises(FileExistsError):
                module.recover(self.source,self.config_path,self.root,existing)
            with self.assertRaisesRegex(ValueError,'互不包含'):
                module.recover(self.source,self.config_path,self.root,self.source/'toy')


if __name__=='__main__':
    unittest.main()
