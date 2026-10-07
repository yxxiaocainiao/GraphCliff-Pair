"""M57: one frozen CPU Chemprop inference, then descriptive training-reference cases."""
import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DEST = ROOT/'artifacts/query_reference_prediction_20261007'
BASE = ROOT/'artifacts/reliability_phase_a_20261005_v3'
OLD = ROOT/'docs/research/query_reference_activity_support_20261007'
env = dict(hashlib=hashlib, json=json, os=os, time=time, subprocess=subprocess)
for path, names in [
    (ROOT/'docs/research/model_conditioned_pair_diagnostic_20261007/diagnose.py', ['sha','save']),
    (ROOT/'experiments/reliability_base/run_phase_a.py', ['run_command'])]:
    nodes=[n for n in ast.parse(path.read_text(encoding='utf-8')).body
           if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
sha, save, run_command = (env[k] for k in ['sha','save','run_command'])


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def split_error(eq, er):
    assert math.isfinite(eq) and math.isfinite(er)
    c, d = (eq+er)/2, (eq-er)/2
    assert math.isclose(c*c+d*d,(eq*eq+er*er)/2,rel_tol=1e-12,abs_tol=1e-12)
    assert math.isclose(c*c-d*d,eq*er,rel_tol=1e-12,abs_tol=1e-12)
    return c*c,d*d


def selfcheck():
    for eq,er in [(2,0),(2,2),(2,-2),(.3,-.1),(-7,1e-8)]:
        c2,d2=split_error(eq,er)
        if er==0: assert c2==d2==eq*eq/4
    for x in [float('nan'),float('inf')]:
        try: split_error(x,0)
        except AssertionError: pass
        else: raise AssertionError('Nonfinite error accepted')
    assert split_error(2,2)==(4,0) and split_error(2,-2)==(0,4)
    print('PASS: decomposition, same/opposite signs, zero-reference limit, finite guard')


def frozen_cases():
    p=read(OLD/'protocol.json')
    for k,h in p['inputs_sha256'].items(): assert sha(ROOT/k)==h,k
    state=read(ROOT/'artifacts/query_reference_activity_support_20261007/execution.json')
    source=ROOT/'artifacts/query_reference_activity_support_20261007/labeled_pairs.json'
    assert state['status']=='completed' and sha(source)==state['labeled_pairs_sha256']
    manifest=read(ROOT/p['manifest'])
    frames={role:{int(r['source_row']):r for r in rows(BASE/'CHEMBL234_Ki'/name)}
            for role,name in [('fit','oof.csv'),('calibration','calibration.csv'),('evaluation','evaluation.csv')]}
    for role,frame in frames.items(): assert set(frame)==set(manifest['roles'][role])
    cases=[]; inputs=[]
    for role,records in read(source).items():
        selected=[r for r in records if abs(r['query_y']-r['reference_y'])>=1]
        assert len(selected)=={'calibration':5,'evaluation':10}[role]
        for r in selected:
            q,ref=frames[role][r['q']],frames['fit'][r['r']]
            assert float(q['y'])==r['query_y'] and float(ref['y'])==r['reference_y']
            assert .9<=r['similarity']<1 and q['canonical']!=ref['canonical']
            assert math.isfinite(float(q['prediction']))
            cases.append(dict(r,role=role,query_prediction=float(q['prediction'])))
            inputs.extend([dict(kind='reference',role='fit',source_row=r['r'],smiles=ref['smiles']),
                           dict(kind='replay',role=role,source_row=r['q'],smiles=q['smiles'])])
    assert len(cases)==15 and len({r['r'] for r in cases})==15 and len({r['q'] for r in cases})==15
    assert len({f['canonical'] for f in frames['fit'].values()} &
               {f['canonical'] for role in ['calibration','evaluation'] for f in frames[role].values()})==0
    return cases,inputs


def analyze(cases, predictions, tolerance):
    assert len(predictions)==30 and all(math.isfinite(float(r['y'])) for r in predictions)
    replay=[]; detailed=[]
    for i,r in enumerate(cases):
        pr,pq=float(predictions[2*i]['y']),float(predictions[2*i+1]['y'])
        replay.append(abs(pq-r['query_prediction']))
        eq,er=r['query_prediction']-r['query_y'],pr-r['reference_y']
        c2,d2=split_error(eq,er)
        detailed.append(dict(r,reference_prediction=pr,query_error=eq,reference_error=er,
                             common_squared=c2,contrast_squared=d2,zero_reference_contrast_squared=eq*eq/4))
    assert max(replay)<=tolerance, f'CPU replay guard failed: {max(replay)}'
    result={}
    for role in ['calibration','evaluation']:
        rr=[r for r in detailed if r['role']==role]; n=len(rr)
        mean=lambda key: sum(r[key] for r in rr)/n
        eq2=sum(r['query_error']**2 for r in rr)/n
        er2=sum(r['reference_error']**2 for r in rr)/n
        c2,d2=mean('common_squared'),mean('contrast_squared')
        result[role]=dict(pairs=n,query_rmse=math.sqrt(eq2),reference_training_rmse=math.sqrt(er2),
            query_mae=sum(abs(r['query_error']) for r in rr)/n,
            reference_training_mae=sum(abs(r['reference_error']) for r in rr)/n,
            delta_mae=sum(abs(r['query_error']-r['reference_error']) for r in rr)/n,
            delta_sign_accuracy=sum((r['query_prediction']-r['reference_prediction'])*(r['query_y']-r['reference_y'])>0 for r in rr)/n,
            pair_mse=(eq2+er2)/2,common_squared=c2,contrast_squared=d2,
            contrast_fraction=d2/(c2+d2) if c2+d2 else None,
            zero_reference_contrast_squared=eq2/4)
    return dict(roles=result,replay_rows=15,replay_max_abs=max(replay)), detailed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['selfcheck','prepare','run','audit']); mode=parser.parse_args().mode
    if mode=='selfcheck': selfcheck(); return
    if mode=='prepare':
        assert not DEST.exists() and not (HERE/'protocol.json').exists()
        cases,inputs=frozen_cases()
        checkpoint=BASE/'CHEMBL234_Ki/full/training/model_0/best.pt'
        binding=read(BASE/'artifact_binding.json')
        assert sha(checkpoint)==binding['CHEMBL234_Ki\\full\\training\\model_0\\best.pt']
        DEST.mkdir();save(DEST/'cases.json',cases);save(DEST/'identities.json',inputs)
        with (DEST/'input.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['smiles']);w.writeheader();w.writerows({'smiles':r['smiles']} for r in inputs)
        command=[sys.executable,'-u','-c','from chemprop.cli.main import main; main()',
                 'predict','-i',str(DEST/'input.csv'),'-o',str(DEST/'predictions.csv'),
                 '--model-paths',str(checkpoint),'--smiles-columns','smiles','--num-workers','0',
                 '--accelerator','cpu','--devices','1','--batch-size','64']
        # Parser validation imports installed code but does not invoke prediction or load weights.
        from chemprop.cli.main import construct_parser
        args=construct_parser().parse_args(command[4:]);assert args.mode=='predict' and args.accelerator=='cpu'
        hashes=dict(read(OLD/'protocol.json')['inputs_sha256'])
        paths=[HERE/'diagnose.py',OLD/'protocol.json',BASE/'artifact_binding.json',checkpoint,
               ROOT/'experiments/reliability_base/run_phase_a.py',
               ROOT/'artifacts/query_reference_activity_support_20261007/labeled_pairs.json',
               ROOT/'artifacts/query_reference_activity_support_20261007/execution.json',
               DEST/'cases.json',DEST/'identities.json',DEST/'input.csv']
        installed=Path(sys.prefix)/'Lib/site-packages/chemprop/cli'
        paths += [installed/n for n in ['predict.py','main.py','common.py']]
        for path in paths: hashes[str(path)]=sha(path)
        save(HERE/'protocol.json',dict(milestone='M57',status='frozen_before_inference',
            authorization='User: 继续吧设定计划按照计划进行; bounded existing-checkpoint reference inference',
            input_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            command=command,inputs_sha256=hashes,cli_timeout_seconds=180,total_wall_seconds=240,
            max_cli_calls=1,checkpoint_paths=1,new_reference_predictions=15,query_replay_rows=15,
            replay_absolute_tolerance=1e-4,
            tolerance_note='Project CPU/GPU float32 numerical guard, not a paper parameter or statistical confidence bound',
            versions={k:importlib.metadata.version(k) for k in ['chemprop','torch','lightning','rdkit','numpy']},
            analysis='Both roles, all 15 preselected pairs: query/reference RMSE/MAE, delta MAE/sign accuracy, common/contrast MSE and er=0 algebraic limit; no groups or threshold search',
            limits='0 training/risk fits/external risk feature recomputation/official test; CLI molecular graph featurization is part of inference',
            caveat='Training-reference predictions are in-sample. Label-selected cases are posthoc only, never deployment neighbors/features. Not OOF, chemical mechanism or algorithm novelty.',
            stopping='Any failure/hash/replay/timeout stops; no automatic retry/device switch or budget expansion'))
        print('PREPARED: 15 references + 15 query replay rows; parser validated; zero weight loads');return
    p=read(HERE/'protocol.json')
    for k,h in p['inputs_sha256'].items(): assert sha(ROOT/k)==h,k
    cases,inputs=frozen_cases();assert cases==read(DEST/'cases.json') and inputs==read(DEST/'identities.json')
    if mode=='audit':
        state=read(DEST/'execution.json'); assert state['status']=='completed'
        assert sha(DEST/'predictions.csv')==state['prediction_sha256']
    else:
        assert not (DEST/'execution.json').exists(),'No overwrite or automatic retry'
        state=dict(status='running',cli_calls_started=0,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
        save(DEST/'execution.json',state)
    tick=time.monotonic()
    try:
        if mode=='run':
            state['cli_calls_started']=1;save(DEST/'execution.json',state)
            run_command(p['command'],DEST/'cli',p['cli_timeout_seconds'])
        predictions=rows(DEST/'predictions.csv')
        assert [r['smiles'] for r in predictions]==[r['smiles'] for r in inputs]
        summary,detailed=analyze(cases,predictions,p['replay_absolute_tolerance'])
        for k,h in p['inputs_sha256'].items(): assert sha(ROOT/k)==h,k
        if mode=='audit':
            assert summary==state['summary'] and detailed==read(DEST/'case_errors.json')
            for r in detailed:
                assert math.isclose(r['common_squared']+r['contrast_squared'],
                                    (r['query_error']**2+r['reference_error']**2)/2,abs_tol=1e-12)
            save(HERE/'results.json',{k:v for k,v in state.items() if k!='prediction_sha256'})
            print('AUDIT PASSED: identities, hashes, 15 replay rows, all pair identities and aggregate replay');return
        assert time.monotonic()-tick<p['total_wall_seconds']
        save(DEST/'case_errors.json',detailed)
        state.update(status='completed',summary=summary,prediction_sha256=sha(DEST/'predictions.csv'),
                     wall_seconds=time.monotonic()-tick,training_fits=0,official_test_rows_read=0)
        save(DEST/'execution.json',state);print(json.dumps(summary),flush=True)
    except BaseException as e:
        if mode=='run':
            state.update(status='failed',error=repr(e),wall_seconds=time.monotonic()-tick)
            save(DEST/'execution.json',state)
        raise


if __name__=='__main__': main()
