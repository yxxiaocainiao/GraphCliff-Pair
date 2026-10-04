import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

spec=importlib.util.spec_from_file_location('replay_validation',Path(__file__).resolve().parents[1]/'tools/replay_validation.py')
replay=importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class ValidationReplayContracts(unittest.TestCase):
    def test_training_bytes_and_path_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            hashes={}
            for name in replay.CORE_SOURCES:
                path=root/name
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(b'# retained CRLF\r\n')
                hashes[name.replace('/','\\')]=replay.digest(path)
            (root/'graphcliff_pair/predict.py').write_text('independent CLI changed')
            manifest={'source_sha256':hashes}
            normalized=replay.check_training_sources(manifest,root)
            self.assertEqual(set(normalized),replay.CORE_SOURCES)
            (root/'graphcliff_pair/vendor/training.py').write_bytes(b'# retained CRLF\n')
            with self.assertRaisesRegex(ValueError,'核心源码与manifest不符'):
                replay.check_training_sources(manifest,root)
            with self.assertRaisesRegex(ValueError,'越出'):
                replay.protected('../outside.py',root)

    def test_checkpoint_epoch_and_predictions_are_checked_independently(self):
        records=[dict(query=4,reference=1,similarity=.8,y=1.,reference_y=2.,prediction=1.2,cliff_mol=1),
                 dict(query=5,reference=2,similarity=.7,y=3.,reference_y=4.,prediction=2.7,cliff_mol=0)]
        summary=dict(**replay.metrics(records),best_epoch=2,epochs_run=3)
        history=[dict(epoch=1,valid_mse=.2),dict(epoch=2,valid_mse=summary['overall_rmse']**2),dict(epoch=3,valid_mse=.3)]
        checkpoint={'epoch':2,'model_state_dict':{}}
        best=replay.check_checkpoint_metadata(checkpoint,summary,history)
        matched=replay.compare_predictions(records,pd.DataFrame(records),summary,best)
        self.assertEqual(matched['max_prediction_abs_error'],0)
        with self.assertRaisesRegex(ValueError,'epoch'):
            replay.check_checkpoint_metadata(dict(checkpoint,epoch=1),summary,history)
        changed=copy.deepcopy(records)
        changed[0]['prediction']+=.001
        with self.assertRaisesRegex(ValueError,'未恢复原验证预测'):
            replay.compare_predictions(changed,pd.DataFrame(records),summary,best)
        changed=copy.deepcopy(records)
        changed[0]['reference']=2
        with self.assertRaisesRegex(ValueError,'身份错位'):
            replay.compare_predictions(changed,pd.DataFrame(records),summary,best)

    def test_actual_checkpoint_replay_and_tamper(self):
        """CPU真实小模型forward+保存/加载，而非仅验证JSON标记。"""
        from graphcliff_pair.predict import graph_and_identity
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            run=root/'artifacts/stage/toy/seed42/cross'
            run.mkdir(parents=True)
            config=dict(hidden_size=16,num_layers=1,heads=4,batch_size=2,
                        arms=[dict(name='cross',variant='cross_attention',readout='sag')])
            manifest=dict(config=config,device='cpu')
            (run.parents[2]/'manifest.json').write_text(json.dumps(manifest))
            pairs=dict(valid_pairs=[dict(query=2,reference=0,similarity=.5),dict(query=3,reference=1,similarity=.6)])
            (run.parents[1]/'pairs.json').write_text(json.dumps(pairs))
            frame=pd.DataFrame(dict(smiles=['CCO','CCN','CCCO','CCCN'],y=[1.,2.,3.,4.],cliff_mol=[0,0,1,0]))
            graphs={i:graph_and_identity(smiles)[0] for i,smiles in enumerate(frame.smiles)}
            torch.set_num_threads(2)
            replay.set_seed(replay.SimpleNamespace(seed=42))
            model=replay.PairRegressor('cross_attention',16,1,4,'sag')
            records=replay.evaluate(model,config['arms'][0],pairs['valid_pairs'],graphs,frame,2,torch.device('cpu'))
            summary=dict(**replay.metrics(records),arm='cross',seed=42,best_epoch=1,epochs_run=1,
                         parameters=replay.parameters(model),test_evaluated=False)
            torch.save(dict(epoch=1,model_state_dict=model.state_dict()),run/'best.pt')
            (run/'summary.json').write_text(json.dumps(summary))
            (run/'history.json').write_text(json.dumps([dict(epoch=1,valid_mse=summary['overall_rmse']**2)]))
            pd.DataFrame(records).to_csv(run/'validation_predictions.csv',index=False)
            result=replay.replay_one(run,manifest,frame,graphs,pairs,torch.device('cpu'),root)
            self.assertEqual(result['count'],2)
            # CSV十进制解析可能引入末位float64舍入；实际forward仍相同。
            self.assertLessEqual(result['max_prediction_abs_error'],1e-12)
            with self.assertRaisesRegex(ValueError,'原训练设备类型'):
                replay.replay_one(run,dict(manifest,device='cuda'),frame,graphs,pairs,torch.device('cpu'),root)
            checkpoint=torch.load(run/'best.pt',weights_only=True)
            # epoch保持正确，改变实际权重也必须由逐预测重放发现。
            # 只改有限回归head，避免损坏上游滤波器而先触发NaN保护。
            checkpoint['model_state_dict']['head.3.weight'].zero_()
            torch.save(checkpoint,run/'best.pt')
            with self.assertRaisesRegex(ValueError,'未恢复原验证预测'):
                replay.replay_one(run,manifest,frame,graphs,pairs,torch.device('cpu'),root)


if __name__=='__main__':
    unittest.main()
