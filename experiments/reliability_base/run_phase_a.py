"""Pinned development-only Chemprop CLI / roughness / RF baseline orchestration."""
import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[2]
GENERIC = ['prediction', 'nn_sim', 'local_dens', 'mol_size', 'rf_var']
EXTRA = ['nbr_disp', 'sali_mean']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + '\n', encoding='utf-8')


def cq_function(source):
    tree = ast.parse(source.read_text(encoding='utf-8'))
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'cq']
    assert len(fn) == 1
    env = {'np': np}
    exec(compile(ast.Module(body=fn, type_ignores=[]), str(source), 'exec'), env)
    return env['cq']


def metrics(frame, scores, cal, cal_scores, cq, protocol):
    error = frame.prediction.to_numpy() - frame.y.to_numpy()
    assert len(scores) == len(frame) and np.isfinite(scores).all()
    order = sorted(range(len(frame)), key=lambda i: (scores[i], frame.canonical.iloc[i], int(frame.source_row.iloc[i])))
    curve = []
    for fraction in protocol['coverage_grid']:
        idx = order[:math.ceil(fraction * len(frame))]
        curve.append(float(np.sqrt(np.mean(error[idx] ** 2))))
    scale = np.maximum(scores, protocol['risk_floor'])
    cal_scale = np.maximum(cal_scores, protocol['risk_floor'])
    q = float(cq(np.abs(cal.prediction.to_numpy() - cal.y.to_numpy()) / cal_scale, protocol['conformal_alpha']))
    threshold = float(np.quantile(cal_scores, .7))
    accept = scores <= threshold
    covered = np.abs(error) <= q * scale
    cliff = frame.cliff_mol.to_numpy() == 1
    result = dict(curve_rmse=curve, curve_mean_rmse=float(np.mean(curve)), full_rmse=float(np.sqrt(np.mean(error**2))),
                  full_mae=float(np.mean(np.abs(error))), interval_coverage=float(np.mean(covered)),
                  interval_mean_width=float(np.mean(2*q*scale)), calibration_q=q, acceptance_threshold=threshold,
                  deployment_acceptance=float(np.mean(accept)), deployment_rmse=float(np.sqrt(np.mean(error[accept]**2))) if accept.any() else None)
    for name, mask in [('cliff',cliff),('noncliff',~cliff)]:
        result[name+'_rows'] = int(mask.sum())
        result[name+'_rmse'] = float(np.sqrt(np.mean(error[mask]**2))) if mask.any() else None
        result[name+'_coverage'] = float(np.mean(covered[mask])) if mask.any() else None
    return result


def run_command(command, cwd, timeout, env=None):
    env = dict(os.environ if env is None else env)
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    cwd.mkdir(parents=True, exist_ok=True)
    save(cwd/'command.json', command)
    tick=time.monotonic()
    with (cwd/'stdout.txt').open('w',encoding='utf-8') as stdout, (cwd/'stderr.txt').open('w',encoding='utf-8') as stderr:
        proc = subprocess.run(command, cwd=cwd, stdout=stdout, stderr=stderr, env=env, timeout=timeout)
    save(cwd/'exit.json', dict(returncode=proc.returncode,seconds=time.monotonic()-tick))
    assert proc.returncode == 0, f'Command failed: see {cwd}/stderr.txt'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','run'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--partitions',type=Path,required=True)
    parser.add_argument('--author',type=Path,required=True)
    args=parser.parse_args()
    args.output=args.output.resolve()
    assert importlib.metadata.version('chemprop') == '2.2.3'
    protocol=json.loads((ROOT/'experiments/reliability_base/fixed_protocol.json').read_text(encoding='utf-8'))
    for source, expected in protocol['source_sha256'].items():
        assert sha(Path(source)) == expected,source
    cq=cq_function(args.author/'src/conformal.py')
    for n in [1,2,10,100,179,426]:
        for alpha in [.1,.5]:
            x=np.arange(n,dtype=float)
            assert cq(x,alpha)==x[min(math.ceil((n+1)*(1-alpha)),n)-1]
    assert np.isinf(cq(np.array([]),.1))
    rows_by_task={};jobs=[]
    for summary in protocol['tasks']:
        task=summary['dataset'];source=args.data_root/(task+'.csv');manifest=args.partitions/(task+'.json')
        assert sha(source)==summary['csv_sha256'] and sha(manifest)==summary['partition_manifest_sha256']
        m=json.loads(manifest.read_text(encoding='utf-8'));roles=m['roles'];folds=m['oof_folds']
        selected=set().union(*map(set,roles.values()))
        assert len(selected)==sum(map(len,roles.values()))
        assert set().union(*map(set,folds.values()))==set(roles['fit'])
        assert sum(map(len,folds.values()))==len(roles['fit'])
        rows={};canonical_roles={}
        role_for_row={row:k for k,ids in roles.items() for row in ids}
        with source.open(encoding='utf-8-sig',newline='') as f:
            for i,row in enumerate(csv.DictReader(f)):
                if i not in selected:continue
                assert row['split']=='train'
                mol=Chem.MolFromSmiles(row['smiles']);assert mol is not None
                canon=Chem.MolToSmiles(mol);role=role_for_row[i]
                assert canon not in canonical_roles or canonical_roles[canon]==role
                canonical_roles[canon]=role
                rows[i]=dict(source_row=i,smiles=row['smiles'],canonical=canon,y=float(row['y']),cliff_mol=int(row['cliff_mol']))
        assert set(rows)==selected and all(np.isfinite(v['y']) for v in rows.values())
        rows_by_task[task]=(rows,roles)
        for fold in range(3):
            query=folds[str(fold)];fit=sorted(set(roles['fit'])-set(query))
            assert not ({rows[i]['canonical'] for i in fit}&{rows[i]['canonical'] for i in query})
            jobs.append(dict(task=task,name=f'fold{fold}',fit=fit,query=query,monitor=roles['monitor']))
        jobs.append(dict(task=task,name='full',fit=roles['fit'],query=roles['calibration']+roles['evaluation'],monitor=roles['monitor']))
    cli=[sys.executable,'-u','-c','from chemprop.cli.main import main; main()']
    from chemprop.cli.main import construct_parser
    if args.mode=='prepare':
        assert not args.output.exists(),'Refuse existing output'
        args.output.mkdir(parents=True)
        # Four points: perfect lower-risk selection gives known curve mean 1.780... vs arbitrary sort.
        f=pd.DataFrame(dict(prediction=[0.,1.,2.,3.],y=[0.]*4,canonical=['a','b','c','d'],source_row=range(4),cliff_mol=[0,1,0,1]))
        test=metrics(f,np.arange(4.),f,np.arange(4.),cq,protocol)
        assert test['curve_rmse'][0]==math.sqrt(.5) and test['full_rmse']==math.sqrt(3.5)
        for job in jobs:
            folder=args.output/job['task']/job['name'];folder.mkdir(parents=True)
            rows,_=rows_by_task[job['task']]
            fit=pd.DataFrame([rows[i] for i in job['fit']]);monitor=pd.DataFrame([rows[i] for i in job['monitor']])
            fit['partition']='train';monitor['partition']='val'
            pd.concat([fit,monitor])[['smiles','y','partition']].to_csv(folder/'train_val.csv',index=False)
            pd.DataFrame([rows[i] for i in job['query']])[['smiles']].to_csv(folder/'query.csv',index=False)
            reference=[dict(smiles=rows[i]['smiles'],y=rows[i]['y'],cliff_mol=0,split='train') for i in job['fit']]
            query=[dict(smiles=rows[i]['smiles'],y=0.,cliff_mol=0,split='test') for i in job['query']]
            (folder/'features/data').mkdir(parents=True)
            pd.DataFrame(reference+query).to_csv(folder/'features/data/fixture.csv',index=False)
            command=cli+['train','-i',str(folder/'train_val.csv'),'-o',str(folder/'training'),'--smiles-columns','smiles','--target-columns','y',
                         '--splits-column','partition','--task-type','regression','--loss-function','mse','--epochs','50','--patience','15',
                         '--batch-size','64','--num-workers','0','--accelerator','gpu','--devices','1','--data-seed','42','--pytorch-seed','42','--save-data-splits']
            parsed=construct_parser().parse_args(command[len(cli):])
            save(folder/'effective_cli_defaults.json',vars(parsed))
            save(folder/'train_command.json',command)
        source_meta=json.loads((ROOT/'docs/research/roughness_method_review_20261005/provenance.json').read_text(encoding='utf-8'))
        expected={s['path']:s['sha256'] for s in source_meta['sources'] if s['repo'].endswith('/qsar-landscape-roughness')}
        for name in ['LICENSE','src/config.py','src/build_features.py','src/conformal.py']:
            assert sha(args.author/name)==expected[name]
            target=args.output/'author'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(args.author/name,target)
        first=jobs[0];folder=args.output/'interface_prediction'
        folder.mkdir();pd.DataFrame([{'smiles':rows_by_task[first['task']][0][i]['smiles']} for i in first['fit'][:2]]).to_csv(folder/'query.csv',index=False)
        old=Path('D:/WORK_SPACE/WORK_SPACE/my_work/graphcliff_baselines')
        binding=json.loads((old/'results/chemprop/CHEMBL234_Ki/manifest.json').read_text(encoding='utf-8'))
        weight=old/binding['checkpoint'];assert sha(weight)==binding['checkpoint_sha256']
        run_command(cli+['predict','-i',str(folder/'query.csv'),'-o',str(folder/'pred.csv'),'--model-paths',str(weight),
                         '--smiles-columns','smiles','--num-workers','0','--accelerator','gpu','--devices','1'],folder,120)
        pred=pd.read_csv(folder/'pred.csv');assert len(pred)==2 and np.isfinite(pred.y).all()
        assert pred.smiles.tolist()==pd.read_csv(folder/'query.csv').smiles.tolist()
        hashes={str(p.relative_to(args.output)):sha(p) for p in args.output.rglob('*') if p.is_file()}
        save(args.output/'prepare.json',dict(passed=True,jobs=jobs,files=hashes,protocol_sha256=sha(ROOT/'experiments/reliability_base/fixed_protocol.json'),
             runner_sha256=sha(Path(__file__)),fits=0,interface_prediction_jobs=1,query_rows=2,quantile_checks=13,risk_metric_check=True,
             feature_invariance_evidence='M21 three-variant checks; features use query y=0 in this adapter',
             RF_adapter='installed sklearn following UNIQUE fit/predict; no full UNIQUE pipeline'))
        print('PREPARE PASSED:12 CLI jobs,0 fits,2-row old-checkpoint interface only',flush=True)
        return
    prepared=json.loads((args.output/'prepare.json').read_text(encoding='utf-8'));assert prepared['passed']
    assert sha(Path(__file__))==prepared['runner_sha256']
    assert sha(ROOT/'experiments/reliability_base/fixed_protocol.json')==prepared['protocol_sha256']
    for name,expected in prepared['files'].items():assert sha(args.output/name)==expected,name
    assert not (args.output/'execution.json').exists(),'Refuse repeated execution; preserve interruption'
    started=time.monotonic();state=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),runner_sha256=sha(Path(__file__)),protocol_sha256=prepared['protocol_sha256'],status='running',chemprop_fits_started=0,author_RF_fits_started=0,risk_RF_fits_started=0,completed_jobs=0)
    save(args.output/'execution.json',state)
    by_task={}
    try:
        for job in jobs:
            assert time.monotonic()-started<180*60,'Stage wall-clock budget'
            task=job['task'];folder=args.output/task/job['name'];rows,roles=rows_by_task[task]
            state['chemprop_fits_started']+=1;save(args.output/'execution.json',state)
            print('START',task,job['name'],'Chemprop',state['chemprop_fits_started'],'/12',flush=True)
            run_command(json.loads((folder/'train_command.json').read_text(encoding='utf-8')),folder/'fit_log',min(900,180*60-(time.monotonic()-started)))
            weight=folder/'training/model_0/best.pt';assert weight.exists()
            run_command(cli+['predict','-i',str(folder/'query.csv'),'-o',str(folder/'predictions.csv'),'--model-paths',str(weight),
                        '--smiles-columns','smiles','--num-workers','0','--accelerator','gpu','--devices','1'],folder/'predict_log',120)
            pred=pd.read_csv(folder/'predictions.csv')
            assert pred.smiles.tolist()==[rows[i]['smiles'] for i in job['query']] and np.isfinite(pred.y).all()
            env=os.environ.copy();base=folder/'features'
            env.update(MOLECULEACE_DATA=str(base/'data'),QSAR_DATA=str(base/'data'),QSAR_CACHE=str(base/'cache'),
                       QSAR_RESULTS=str(base/'results'),QSAR_FIGURES=str(base/'figures'),PYTHONIOENCODING='utf-8')
            state['author_RF_fits_started']+=1;save(args.output/'execution.json',state)
            run_command([sys.executable,str(args.output/'author/src/build_features.py'),'fixture'],base/'log',300,env)
            feats=pd.read_csv(base/'cache/fixture.csv')
            assert feats.smiles.tolist()==pred.smiles.tolist()
            frame=pd.DataFrame([rows[i] for i in job['query']]);frame['prediction']=pred.y.to_numpy()
            for name in GENERIC[1:]+EXTRA:frame[name]=feats[name].to_numpy()
            assert np.isfinite(frame[GENERIC+EXTRA]).all().all()
            frame.to_csv(folder/'joined.csv',index=False)
            by_task.setdefault(task,[]).append(frame)
            state['completed_jobs']+=1;save(args.output/'execution.json',state)
            print('DONE',task,job['name'],len(frame),'queries',flush=True)
        results=[]
        for summary in protocol['tasks']:
            task=summary['dataset'];folder=args.output/task;rows,roles=rows_by_task[task]
            oof=pd.concat(by_task[task][:3]).sort_values('source_row');assert set(oof.source_row)==set(roles['fit']) and oof.source_row.is_unique
            final=by_task[task][3].set_index('source_row',drop=False)
            cal=final.loc[roles['calibration']].reset_index(drop=True);ev=final.loc[roles['evaluation']].reset_index(drop=True)
            target=np.abs(oof.prediction.to_numpy()-oof.y.to_numpy());scores={}
            scores['distance']=(1-cal.nn_sim.to_numpy(),1-ev.nn_sim.to_numpy())
            scores['sali']=(cal.sali_mean.to_numpy(),ev.sali_mean.to_numpy())
            for name,columns in [('generic',GENERIC),('augmented',GENERIC+EXTRA)]:
                state['risk_RF_fits_started']+=1;save(args.output/'execution.json',state)
                rf=RandomForestRegressor(n_estimators=200,max_features='sqrt',min_samples_leaf=5,random_state=42,n_jobs=-1).fit(oof[columns],target)
                scores[name]=(rf.predict(cal[columns]),rf.predict(ev[columns]))
            record=dict(dataset=task,seed=42,arms={})
            for name,(sc,se) in scores.items():
                record['arms'][name]=metrics(ev,se,cal,sc,cq,protocol)
                ev[name+'_risk']=se;cal[name+'_risk']=sc
            ev.to_csv(folder/'evaluation.csv',index=False);cal.to_csv(folder/'calibration.csv',index=False);oof.to_csv(folder/'oof.csv',index=False)
            save(folder/'metrics.json',record);results.append(record)
        improvements=[(v['arms']['generic']['curve_mean_rmse']-v['arms']['augmented']['curve_mean_rmse'])/v['arms']['generic']['curve_mean_rmse'] for v in results]
        macro={a:float(np.mean([v['arms'][a]['curve_mean_rmse'] for v in results])) for a in scores}
        passed=sum(v>=.05 for v in improvements)>=2 and min(improvements)>=-.05 and np.mean(improvements)>=.03 and macro['augmented']<min(macro['distance'],macro['sali'])
        save(args.output/'results.json',dict(results=results,relative_improvements=improvements,macro_curve_rmse=macro,phase_A_pass=bool(passed),
             decision='Eligible for fixed seed confirmation; not novel method evidence' if passed else 'Stop roughness-augmentation candidate; do not run phase B',
             official_test_rows_used=0))
        state['status']='complete';state['seconds']=time.monotonic()-started;save(args.output/'execution.json',state)
        print('PHASE A COMPLETE; PASS=',passed,flush=True)
    except BaseException as error:
        state.update(status='interrupted',error=repr(error),seconds=time.monotonic()-started);save(args.output/'execution.json',state)
        raise


if __name__=='__main__':
    main()
