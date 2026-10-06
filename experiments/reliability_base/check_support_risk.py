"""Fixed zero-fit subgroup diagnostic; local_dens is similarity, not sample size."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from run_phase_a import ROOT, cq_function, metrics, save, sha


def curves(frame,cal,cq,base):
    assert len(frame)>0
    result={}
    for arm in ['generic','augmented']:
        value=metrics(frame,frame[arm+'_risk'].to_numpy(),cal,cal[arm+'_risk'].to_numpy(),cq,base)
        result[arm]={k:value[k] for k in ['curve_rmse','curve_mean_rmse']}
    result['gain']=result['generic']['curve_mean_rmse']-result['augmented']['curve_mean_rmse']
    return result


def tail_removed(frame):
    error=np.abs(frame.prediction.to_numpy()-frame.y.to_numpy())
    # Fixed identity ordering breaks equal-error ties before selecting a maximum.
    assert frame.source_row.is_unique
    keep=np.ones(len(frame),dtype=bool);keep[int(np.argmax(error))]=False
    return frame.loc[keep].reset_index(drop=True)


def condition(low,high):
    return bool(low<0 and high>0)


def self_check():
    f=pd.DataFrame(dict(source_row=range(4),canonical=list('abcd'),prediction=[0.,1.,2.,3.],y=[0.]*4,
                        cliff_mol=[0,1,0,1],generic_risk=[0.,1.,2.,3.],augmented_risk=[3.,2.,1.,0.]))
    base={'coverage_grid':[.5,1.],'risk_floor':1e-6,'conformal_alpha':.1}
    value=curves(f,f,lambda x,a:1.,base)
    assert abs(value['generic']['curve_rmse'][0]-np.sqrt(.5))<1e-12
    assert abs(value['augmented']['curve_rmse'][0]-np.sqrt(6.5))<1e-12
    assert value['gain']<0 and tail_removed(f).source_row.tolist()==[0,1,2]
    assert condition(-.1,.1) and not condition(.1,.1)
    assert not condition(-.1,-.1) and not condition(0.,.1)
    print('PASS: identical-input risk comparison, shared tail exclusion, signed gain and screen')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.self_check: self_check();return
    assert args.output is not None and not args.output.exists(),'Refuse existing output'
    pp=ROOT/'experiments/reliability_base/support_diagnostic_protocol.json'
    plan=json.loads(pp.read_text(encoding='utf-8'))
    protected={ROOT/p:h for p,h in plan['inputs'].items()}
    protected[pp]=sha(pp);protected[Path(__file__)]=sha(Path(__file__))
    assert all(sha(p)==h for p,h in protected.items()),'Input changed'
    src=ROOT/plan['input_directory']
    binding=json.loads((src/'artifact_binding.json').read_text(encoding='utf-8'))
    base=json.loads((ROOT/plan['base_protocol']).read_text(encoding='utf-8'))
    cq=cq_function(ROOT/'artifacts/roughness_sources_20261005/qsar-landscape-roughness/src/conformal.py')
    args.output.mkdir(parents=True)
    state=dict(status='running',analysis_attempts=1,fits=0,checkpoint_predictions=0,official_test_rows=0,
               code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    save(args.output/'execution.json',state);tick=time.monotonic()
    try:
        results=[]
        for task in plan['tasks']:
            roles=json.loads((ROOT/plan['partition_directory']/(task+'.json')).read_text(encoding='utf-8'))['roles']
            frames={}
            for role in ['calibration','evaluation']:
                p=src/task/(role+'.csv');assert sha(p)==binding[str(p.relative_to(src))]
                f=pd.read_csv(p,float_precision='round_trip').sort_values('source_row').reset_index(drop=True)
                assert f.source_row.is_unique and set(f.source_row)==set(roles[role])
                assert np.isfinite(f[['y','prediction','local_dens','generic_risk','augmented_risk']]).all().all()
                frames[role]=f
            cal,ev=frames['calibration'],frames['evaluation']
            assert not set(cal.canonical)&set(ev.canonical)
            threshold=float(np.quantile(cal.local_dens,.5,method='linear'))
            groups={}
            for name,mask in [('low',ev.local_dens<threshold),('high',ev.local_dens>=threshold)]:
                f=ev.loc[mask].reset_index(drop=True)
                value={'rows':len(f),'eligible':len(f)>=plan['min_evaluation_group_rows']}
                if len(f)>=2:
                    value.update(original=curves(f,cal,cq,base),tail_removed=curves(tail_removed(f),cal,cq,base))
                else: value.update(original=None,tail_removed=None)
                groups[name]=value
            eligible=all(g['eligible'] for g in groups.values())
            versions={}
            for v in ['original','tail_removed']:
                low,high=groups['low'][v],groups['high'][v]
                versions[v]=dict(pattern=condition(low['gain'],high['gain']) if low and high else None,
                                 high_minus_low_gain=high['gain']-low['gain'] if low and high else None)
            results.append(dict(dataset=task,seed=42,calibration_rows=len(cal),evaluation_rows=len(ev),
                                threshold=threshold,groups=groups,versions=versions,
                                eligible=eligible,robust_pattern=eligible and all(v['pattern'] for v in versions.values())))
        interactions={v:float(np.mean([r['versions'][v]['high_minus_low_gain'] for r in results]))
                      if all(r['versions'][v]['high_minus_low_gain'] is not None for r in results) else None
                      for v in ['original','tail_removed']}
        passed=bool(sum(r['robust_pattern'] for r in results)>=2 and all(x is not None and x>0 for x in interactions.values()))
        assert all(sha(p)==h for p,h in protected.items()),'Input changed during analysis'
        save(args.output/'results.json',dict(results=results,mean_interaction=interactions,issue_screen=passed,
             new_fits=0,new_checkpoint_predictions=0,official_test_rows=0,scope=plan['scope'],code_commit=state['code_commit'],
             input_sha256={p.relative_to(ROOT).as_posix():h for p,h in protected.items()},
             decision='Only pursue specific mechanism evidence; no core or training established' if passed else 'End support-proxy diagnostic; no new training'))
        state.update(status='complete',seconds=time.monotonic()-tick)
        print(json.dumps(dict(screen=passed,robust_patterns=[r['robust_pattern'] for r in results],mean_interaction=interactions)))
    except Exception as exc:
        state.update(status='failed',error=repr(exc),seconds=time.monotonic()-tick)
        raise
    finally:save(args.output/'execution.json',state)


if __name__=='__main__':main()
