"""M78 thin adapter: reuse M77 exactly, changing only risk RF seed."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import run_reference_marginalization as pilot
from run_phase_a import ROOT, save, sha, run_command

DOC=ROOT/'docs/research/reference_marginalization_seeds_20261008'
PROTOCOL=DOC/'protocol.json'
FILES=['source.csv','augmented.csv','targets.csv','absolute_error.csv','identities.csv']


def seeded(p,seed):
    assert seed in p['risk_seeds']
    result=dict(p,seed=seed,risk_seed=seed,adapter_sha256=sha(Path(__file__)))
    result['risk_rf']=dict(p['risk_rf'],random_state=seed)
    return result


def inputs():
    p=json.loads(PROTOCOL.read_text(encoding='utf-8'))
    for name,h in p['input_sha256'].items():
        assert sha(ROOT/name)==h,name
    pilot.pinned()  # Verify original software, roles and cached provenance; no record parsing.
    base=ROOT/p['base'];out=ROOT/p['output']
    assert out.parent==ROOT/'artifacts' and p['risk_seeds']==[43,44]
    original=json.loads((base/'inputs.json').read_text(encoding='utf-8'))
    assert original['runner_sha256']==sha(Path(pilot.__file__))
    assert all(sha(base/f)==original['files'][f] for f in FILES)
    assert json.loads((base/'state.json').read_text(encoding='utf-8'))['status']=='completed'
    return p,base,out


def scores_locked(p,out):
    for seed in p['risk_seeds']:
        dest=out/str(seed)
        lock=json.loads((dest/'score_lock.json').read_text(encoding='utf-8'))
        assert lock['protocol_sha256']==sha(dest/'protocol.json')
        assert lock['runner_sha256']==sha(Path(pilot.__file__))
        assert lock['inputs_sha256']==sha(dest/'inputs.json')
        assert lock['scores_sha256']==sha(dest/'scores.csv')
        assert lock['contexts_sha256']==sha(dest/'contexts.csv')


def worker(p,base,out,mode):
    root=json.loads((out/'state.json').read_text(encoding='utf-8'))
    assert root['adapter_sha256']==sha(Path(__file__)) and root['protocol_sha256']==sha(PROTOCOL)
    if mode=='fit':
        assert root['status']=='running' and not any((out/str(s)).exists() for s in p['risk_seeds'])
        for seed in p['risk_seeds']:
            dest=out/str(seed);dest.mkdir(exist_ok=False)
            sp=seeded(p,seed);save(dest/'protocol.json',sp)
            pilot.PROTOCOL=dest/'protocol.json'
            for f in FILES:shutil.copyfile(base/f,dest/f)
            save(dest/'inputs.json',dict(protocol_sha256=sha(pilot.PROTOCOL),runner_sha256=sha(Path(pilot.__file__)),
                 files={f:sha(dest/f) for f in FILES},evaluation_truth_rows_read=0))
            state=dict(status='prepared',risk_fits_started=0,risk_predict_calls_started=0,risk_prediction_rows=0,
                 evaluation_truth_rows_read=0,cached_evaluation_truth_rows_read=0,automatic_retries=0)
            save(dest/'state.json',state)
            pilot.fit_score(sp,dest,state)
        scores_locked(p,out)
        root['status']='all_scores_locked';save(out/'state.json',root)
        return
    assert root['status'] in ('all_scores_locked','completed')
    scores_locked(p,out)  # Both new seeds locked before any cached evaluation truth parsing.
    results={}
    for seed in p['risk_seeds']:
        dest=out/str(seed);pilot.PROTOCOL=dest/'protocol.json'
        sp=json.loads(pilot.PROTOCOL.read_text(encoding='utf-8'))
        assert sp==seeded(p,seed)
        state=json.loads((dest/'state.json').read_text(encoding='utf-8'))
        if mode=='evaluate':
            assert state['status']=='scored' and not (dest/'evaluation_truth.csv').exists()
            shutil.copyfile(base/'evaluation_truth.csv',dest/'evaluation_truth.csv')
            state.update(status='evaluating',truth_sha256=sha(dest/'evaluation_truth.csv'))
        else:
            assert state['status']=='completed'
        results[str(seed)]=pilot.evaluate(sp,dest,state)
        if mode=='evaluate':
            state['cached_evaluation_truth_rows_read']=585
            save(dest/'state.json',state)
    result=dict(point_model_seed=42,risk_seeds=p['risk_seeds'],results=results,
        continuation_condition_met=all(v['continuation_condition_met'] for v in results.values()),no_automatic_expansion=True)
    if mode=='audit':
        assert json.loads((out/'results.json').read_text(encoding='utf-8'))==result
    else:
        save(out/'results.json',result)
        root.update(status='completed',risk_fits=6,risk_predict_calls=6,risk_prediction_rows=14040,
             original_csv_truth_rows_read=0,cached_truth_rows_parsed=1170,new_features=0)
        save(out/'state.json',root)
    print(json.dumps(result,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['selfcheck','run','audit','worker-fit','worker-evaluate'])
    args=ap.parse_args()
    if args.mode=='selfcheck':
        p=json.loads(PROTOCOL.read_text(encoding='utf-8'))
        for seed in p['risk_seeds']:
            sp=seeded(p,seed)
            assert sp['point_model_seed']==42 and sp['seed']==seed
            assert {k:v for k,v in sp['risk_rf'].items() if k!='random_state'}=={k:v for k,v in p['risk_rf'].items() if k!='random_state'}
        assert len(FILES)==5 and 2*3*(4*585)==14040
        print('PASS: only risk seed changes; fixed point seed, exact six-fit prediction budget; 0 fits');return
    p,base,out=inputs()
    if args.mode.startswith('worker-') or args.mode=='audit':
        worker(p,base,out,args.mode.removeprefix('worker-'));return
    out.mkdir(exist_ok=False)
    state=dict(status='running',adapter_sha256=sha(Path(__file__)),protocol_sha256=sha(PROTOCOL),automatic_retries=0)
    save(out/'state.json',state)
    started=time.monotonic()
    try:
        for mode,cap in [('fit',p['budgets']['fit_predict_seconds']),('evaluate',p['budgets']['evaluation_seconds'])]:
            remaining=p['budgets']['whole_seconds']-(time.monotonic()-started)
            assert remaining>0
            run_command([sys.executable,str(Path(__file__).resolve()),'worker-'+mode],out/(mode+'_log'),min(cap,remaining))
        state=json.loads((out/'state.json').read_text(encoding='utf-8'))
        state['run_seconds']=time.monotonic()-started;save(out/'state.json',state)
        print(json.dumps(state,indent=2))
    except BaseException as error:
        state=json.loads((out/'state.json').read_text(encoding='utf-8'))
        state.update(status='failed_or_interrupted',error=repr(error))
        state['risk_fits_started']=sum(json.loads(f.read_text(encoding='utf-8'))['risk_fits_started'] for f in out.glob('*/state.json'))
        save(out/'state.json',state);raise


if __name__=='__main__':
    main()
