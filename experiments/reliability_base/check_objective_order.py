"""Zero-fit descriptive MAE/MSE ordering diagnostic on previously seen caches."""
import argparse
import itertools
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from run_phase_a import ROOT, cq_function, metrics, save, sha


def groups(scores, boundaries):
    return np.searchsorted(boundaries, scores, side='right')


def summary(errors):
    errors=np.asarray(errors,dtype=float)
    assert np.isfinite(errors).all() and (errors>=0).all()
    if not len(errors):
        return dict(rows=0,mae=None,mse=None,rmse=None,top_tenth_squared_share=None)
    squared=errors**2
    tail=np.sort(squared)[-int(np.ceil(.1*len(errors))):].sum()
    return dict(rows=len(errors),mae=float(errors.mean()),mse=float(squared.mean()),
                rmse=float(np.sqrt(squared.mean())),top_tenth_squared_share=float(tail/squared.sum()) if squared.sum()>0 else None)


def inversion(a,b):
    if not len(a) or not len(b): return False
    x,y=summary(a),summary(b)
    return bool((x['mae']-y['mae'])*(x['mse']-y['mse'])<0)


def compare(a,b,min_rows):
    eligible=len(a)>=min_rows and len(b)>=min_rows
    versions=[]
    if len(a) and len(b):
        ta=np.delete(a,np.argmax(a));tb=np.delete(b,np.argmax(b))
        versions=[inversion(ta,b),inversion(a,tb),inversion(ta,tb)]
    raw=inversion(a,b)
    return dict(eligible=eligible,raw_inversion=raw,one_tail_removal_inversions=versions,
                robust_inversion=bool(eligible and raw and all(versions)))


def self_check():
    a=np.full(30,2.);b=np.array([0.]*27+[10.]*3)
    assert compare(a,b,20)['robust_inversion']
    assert not compare(a,a,20)['raw_inversion']
    assert not compare(a[:5],b,20)['eligible']
    assert not compare(np.array([]),b,20)['raw_inversion']
    assert groups(np.array([0.,1.,2.,3.]),np.array([1.,2.,3.])).tolist()==[0,1,2,3]
    assert summary(np.zeros(3))['top_tenth_squared_share'] is None
    print('PASS: inversion, sensitivity, small/empty bins, ties, boundary roles')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.self_check: self_check();return
    assert args.output is not None and not args.output.exists(),'Refuse to overwrite diagnostic'
    plan_path=ROOT/'experiments/reliability_base/objective_diagnostic_protocol.json'
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    inputs={ROOT/p:h for p,h in plan['inputs'].items()}
    inputs[plan_path]=sha(plan_path);inputs[Path(__file__)]=sha(Path(__file__))
    assert all(sha(p)==h for p,h in inputs.items()),'Changed input'
    src=ROOT/plan['input_directory'];base=json.loads((ROOT/plan['base_protocol']).read_text(encoding='utf-8'))
    binding=json.loads((src/'artifact_binding.json').read_text(encoding='utf-8'))
    old=json.loads((ROOT/'experiments/reliability_base/phase_a_results.json').read_text(encoding='utf-8'))
    cq=cq_function(ROOT/'artifacts/roughness_sources_20261005/qsar-landscape-roughness/src/conformal.py')
    args.output.mkdir(parents=True)
    state=dict(status='running',analysis_attempts=1,fits=0,checkpoint_predictions=0,official_test_rows=0,
               code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    save(args.output/'execution.json',state)
    tick=time.monotonic()
    try:
        results=[]
        for task in plan['tasks']:
            manifest=json.loads((ROOT/plan['partition_directory']/(task+'.json')).read_text(encoding='utf-8'))
            frames={}
            for role in ['calibration','evaluation']:
                p=src/task/(role+'.csv');assert sha(p)==binding[str(p.relative_to(src))]
                f=pd.read_csv(p,float_precision='round_trip').sort_values('source_row').reset_index(drop=True)
                assert f.source_row.is_unique and set(f.source_row)==set(manifest['roles'][role])
                assert np.isfinite(f[['prediction','y',plan['risk_column']]]).all().all()
                frames[role]=f
            cal,ev=frames['calibration'],frames['evaluation']
            assert not set(cal.canonical)&set(ev.canonical)
            boundaries=np.quantile(cal[plan['risk_column']],plan['quantiles'],method=plan['quantile_method'])
            bins=groups(ev[plan['risk_column']].to_numpy(),boundaries)
            errors=np.abs(ev.prediction.to_numpy()-ev.y.to_numpy())
            subsets=[errors[bins==i] for i in range(4)]
            records=[dict(bin=i,**summary(e)) for i,e in enumerate(subsets)]
            pairs=[dict(left=i,right=j,**compare(subsets[i],subsets[j],plan['min_bin_rows'])) for i,j in itertools.combinations(range(4),2)]
            # Existing metrics verify that cached predictions/risk still reproduce the old curve.
            curve=metrics(ev,ev[plan['risk_column']].to_numpy(),cal,cal[plan['risk_column']].to_numpy(),cq,base)['curve_rmse']
            saved=next(r for r in old['results'] if r['dataset']==task)['arms']['augmented']['curve_rmse']
            assert np.allclose(curve,saved,atol=1e-12,rtol=0)
            results.append(dict(dataset=task,seed=42,calibration_rows=len(cal),evaluation_rows=len(ev),
                                risk_boundaries=boundaries.tolist(),bins=records,pairs=pairs,
                                has_robust_inversion=any(p['robust_inversion'] for p in pairs),curve_rmse=curve))
        assert all(sha(p)==h for p,h in inputs.items()),'Changed input during analysis'
        passed=sum(r['has_robust_inversion'] for r in results)>=2
        output=dict(results=results,issue_screen=passed,new_fits=0,new_checkpoint_predictions=0,official_test_rows=0,
                    scope=plan['scope'],input_sha256={p.relative_to(ROOT).as_posix():h for p,h in inputs.items()},
                    decision='Issue-specific evidence review only; no new algorithm established' if passed else 'End this objective diagnostic; no new training',
                    code_commit=state['code_commit'])
        save(args.output/'results.json',output)
        state.update(status='complete',seconds=time.monotonic()-tick)
        print(json.dumps(dict(screen=passed,counts=[sum(p['robust_inversion'] for p in r['pairs']) for r in results],seconds=state['seconds'])))
    except Exception as exc:
        state.update(status='failed',error=repr(exc),seconds=time.monotonic()-tick)
        raise
    finally:
        save(args.output/'execution.json',state)


if __name__=='__main__': main()
