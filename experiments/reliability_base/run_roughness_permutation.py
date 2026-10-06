"""Pinned 18-fit block-permutation diagnostic; no Chemprop or test evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from run_phase_a import ROOT, GENERIC, EXTRA, cq_function, metrics, save, sha


def permute(frame, dataset, role, repeat):
    token = f'20261006|roughness-block|{dataset}|{role}|{repeat}'
    seed = int(hashlib.sha256(token.encode()).hexdigest()[:16], 16)
    indices = np.random.default_rng(seed).permutation(len(frame))
    result = frame.copy()
    result[EXTRA] = frame[EXTRA].to_numpy()[indices]
    return result, seed, indices


def self_check():
    frame = pd.DataFrame(dict(prediction=[1.,2.,3.,4.],y=[4.,3.,2.,1.],nbr_disp=[0.,1.,2.,3.],sali_mean=[10.,11.,12.,13.]))
    shuffled, seed, indices = permute(frame,'fixture','oof',0)
    assert sorted(map(tuple,shuffled[EXTRA].to_numpy()))==sorted(map(tuple,frame[EXTRA].to_numpy()))
    assert shuffled.drop(columns=EXTRA).equals(frame.drop(columns=EXTRA))
    assert permute(frame,'fixture','oof',0)[0].equals(shuffled)
    assert permute(frame,'fixture','evaluation',0)[1]!=seed
    assert np.array_equal(shuffled.sali_mean-shuffled.nbr_disp,np.full(4,10.))
    print('PASS: joint permutation, deterministic seeds, role separation, unchanged labels/features')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check',action='store_true')
    parser.add_argument('--input',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.self_check:
        self_check(); return
    assert args.input and args.output
    src,dest=args.input.resolve(),args.output.resolve()
    assert not dest.exists(),'Refuse previous output; failed fits count against budget'
    fixed=ROOT/'experiments/reliability_base/permutation_protocol.json'
    plan=json.loads(fixed.read_text(encoding='utf-8'))
    protected={ROOT/p:h for p,h in plan['inputs'].items()}
    assert all(sha(p)==h for p,h in protected.items()),'Frozen input changed'
    assert src==ROOT/plan['input_directory']
    base=json.loads((ROOT/plan['base_protocol']).read_text(encoding='utf-8'))
    cq=cq_function(ROOT/plan['author_conformal'])
    bindings=json.loads((src/'artifact_binding.json').read_text(encoding='utf-8'))
    frames={}
    for task in plan['tasks']:
        frames[task]={}
        summary=next(x for x in base['tasks'] if x['dataset']==task)
        manifest=ROOT/plan['partition_directory']/(task+'.json')
        assert sha(manifest)==summary['partition_manifest_sha256']
        roles=json.loads(manifest.read_text(encoding='utf-8'))['roles']
        for role in ['oof','calibration','evaluation']:
            p=src/task/(role+'.csv'); assert sha(p)==bindings[str(p.relative_to(src))]
            f=pd.read_csv(p).sort_values('source_row').reset_index(drop=True)
            assert f.source_row.is_unique and set(f.source_row)==set(roles['fit' if role=='oof' else role])
            assert np.isfinite(f[GENERIC+EXTRA+['y']]).all().all()
            frames[task][role]=f
        sets=[set(frames[task][r].canonical) for r in ['oof','calibration','evaluation']]
        assert not any(sets[i]&sets[j] for i in range(3) for j in range(i))
    dest.mkdir(parents=True)
    state=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        runner_sha256=sha(Path(__file__)),protocol_sha256=sha(fixed),fits_started=0,fits_completed=0,
        status='running',chemprop_fits=0,official_test_rows=0,runs=[])
    tick=time.monotonic();results=[]
    try:
        for task in plan['tasks']:
            original=frames[task];target=np.abs(original['oof'].prediction.to_numpy()-original['oof'].y.to_numpy())
            records=[]
            for repeat in [-1]+plan['permutation_repeats']:
                arm='true_replay' if repeat==-1 else f'permuted_{repeat}'
                folder=dest/task/arm;folder.mkdir(parents=True)
                current={};seed_meta={}
                for role,f in original.items():
                    if repeat==-1: current[role]=f
                    else:
                        current[role],seed,indices=permute(f,task,role,repeat)
                        assert current[role].drop(columns=EXTRA).equals(f.drop(columns=EXTRA))
                        seed_meta[role]=dict(seed=seed,permutation_sha256=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest())
                        pd.DataFrame(dict(source_row=f.source_row,donor_source_row=f.source_row.to_numpy()[indices])).to_csv(folder/(role+'_permutation.csv'),index=False)
                assert state['fits_started']<plan['max_fits']
                state['fits_started']+=1;save(dest/'execution.json',state)
                start=time.monotonic()
                rf=RandomForestRegressor(**plan['rf']).fit(current['oof'][GENERIC+EXTRA],target)
                state['fits_completed']+=1
                sc=rf.predict(current['calibration'][GENERIC+EXTRA]);se=rf.predict(current['evaluation'][GENERIC+EXTRA])
                error=None
                if repeat==-1:
                    error=max(float(np.max(np.abs(sc-original['calibration'].augmented_risk))),float(np.max(np.abs(se-original['evaluation'].augmented_risk))))
                    assert error<=plan['replay_tolerance'],f'{task}: replay mismatch {error}'
                result=metrics(original['evaluation'],se,original['calibration'],sc,cq,base)
                record=dict(arm=arm,repeat=repeat,curve_rmse=result['curve_rmse'],curve_mean_rmse=result['curve_mean_rmse'],
                    replay_max_abs_error=error,seconds=time.monotonic()-start,permutations=seed_meta)
                records.append(record);state['runs'].append(dict(dataset=task,**record));save(folder/'metrics.json',record)
                joblib.dump(rf,folder/'risk_rf.joblib',compress=3)
                for role,scores in [('calibration',sc),('evaluation',se)]:
                    pd.DataFrame(dict(source_row=original[role].source_row,risk=scores)).to_csv(folder/(role+'_risk.csv'),index=False)
                save(dest/'execution.json',state);print('DONE',task,arm,flush=True)
            true=records[0]['curve_mean_rmse'];median=float(np.median([r['curve_mean_rmse'] for r in records[1:]]))
            results.append(dict(dataset=task,seed=42,runs=records,permuted_median=median,true_rmse=true,
                absolute_advantage=median-true,relative_advantage_pct=100*(median-true)/median))
        differences=[r['absolute_advantage'] for r in results]
        passed=sum(x>0 for x in differences)>=plan['required_tasks'] and np.mean(differences)>0
        assert all(sha(p)==h for p,h in protected.items()),'Input changed during fits'
        save(dest/'results.json',dict(results=results,gate_pass=bool(passed),mean_absolute_advantage=float(np.mean(differences)),
            fits=state['fits_started'],fits_completed=state['fits_completed'],code_commit=state['code_commit'],
            protocol_sha256=state['protocol_sha256'],runner_sha256=state['runner_sha256'],input_sha256=plan['inputs'],official_test_rows=0))
        state['status']='complete';state['gate_pass']=bool(passed)
    except BaseException as error:
        state['status']='failed';state['exception']=repr(error)
        raise
    finally:
        state['seconds']=time.monotonic()-tick;save(dest/'execution.json',state)


if __name__=='__main__':
    main()
