"""M77: bounded six-arm risk pilot; no point/auxiliary model execution."""
import argparse
import ast
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor

from run_phase_a import ROOT, GENERIC, EXTRA, save, sha, run_command
from run_conditional_pilot import read_selected, SOURCE
from run_query_pair_pilot import geometry, independent_curve

DOC = ROOT/'docs/research/reference_marginalization_pilot_20261008'
PROTOCOL = DOC/'protocol.json'
FEATURES = GENERIC+EXTRA


def read(path, **kw):
    return pd.read_csv(path, float_precision='round_trip', **kw)


def pinned():
    p = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    for name, expected in p['input_sha256'].items():
        assert sha(ROOT/name) == expected, 'Pinned input changed: '+name
    for name, expected in p['software'].items():
        assert importlib.metadata.version(name) == expected, name
    assert sha(SOURCE) == p['source_sha256']
    out = ROOT/p['output']
    assert out.parent == ROOT/'artifacts'
    return p, ROOT/p['base'], out


def ordering(frame, score):
    assert len(frame) == len(score) and np.isfinite(score).all()
    return sorted(range(len(frame)), key=lambda i: (float(score[i]), frame.canonical.iloc[i], int(frame.source_row.iloc[i])))


def roughness(qids, rids, ids, fps, labels, deadline):
    assert len(rids) >= 10 and not set(qids)&set(rids)
    assert not set(ids.loc[qids,'canonical'])&set(ids.loc[rids,'canonical'])
    ref = [fps[i] for i in rids]
    y = labels.loc[rids].to_numpy()
    result = []
    for i in qids:
        assert time.monotonic() < deadline, 'Preparation budget exceeded'
        sim = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], ref))
        nn = np.argsort(-sim)[:10]  # Same author tie rule and reference order.
        selected = [ref[j] for j in nn]
        pair = np.array([DataStructs.BulkTanimotoSimilarity(fp, selected) for fp in selected])
        _, raw = geometry(sim[nn], pair, y[nn])  # Existing vetted std/SALI implementation.
        result.append(raw)
    return np.asarray(result)


def prepare(p, base, out, state):
    deadline = time.monotonic()+p['budgets']['prepare_seconds']
    prior = json.loads((ROOT/'docs/research/conditional_pilot_run_20261008/verification.json').read_text())
    for name, expected in prior['evidence_hashes'].items():
        assert sha(base/name) == expected, name
    old = json.loads((base/'inputs.json').read_text())
    for name, expected in old['files'].items():
        assert sha(base/name) == expected, name
    review = json.loads((ROOT/'docs/research/reference_context_review_20261008/verification.json').read_text())
    jobs = json.loads((base/'jobs.json').read_text())
    roles = json.loads((ROOT/p['roles_manifest']).read_text())
    role_ids = lambda role: sorted(r['source_row'] for r in roles if r['role'] == role)
    fit, ev = jobs[-1]['fit'], jobs[-1]['query']
    assert fit == role_ids('fit') and ev == role_ids('evaluation')
    assert len(fit) == 1403 and len(ev) == 585
    folds = [j['query'] for j in jobs[:3]]
    assert [len(f) for f in folds] == [468,468,467]
    assert sorted(sum(folds, [])) == fit
    for k, j in enumerate(jobs[:3]):
        assert j['fit'] == sorted(set(fit)-set(folds[k]))
        assert j['monitor'] == role_ids('monitor')
        assert folds[k] == sorted(r['source_row'] for r in roles if r['role']=='fit' and r['oof_fold']==k)
    ids = read(base/'identities.csv', index_col='source_row', usecols=['source_row','smiles','canonical']).loc[fit+ev]
    assert ids.index.is_unique and ids.canonical.is_unique
    labels = read(base/'source_labels.csv', index_col='source_row').y
    assert labels.index.tolist() == fit and np.isfinite(labels).all()
    author = base/'author/src/build_features.py'
    fn = next(n for n in ast.parse(author.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='featurize')
    env = dict(np=np, Chem=Chem, DataStructs=DataStructs,
               _gen=rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048))
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(author), 'exec'), env)
    bits, _, _, ok = env['featurize'](ids.smiles.tolist())
    assert ok.all() and len(bits)==1988
    assert [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in ids.smiles] == ids.canonical.tolist()
    fps = dict(zip(ids.index, bits))
    frames, max_replay = [], 0.
    for j in jobs:
        folder = base/p['task']/j['name']
        assert sha(folder/'features/cache/fixture.csv') == review['cache_hashes_observed_now'][j['name']]
        feat = read(folder/'features/cache/fixture.csv', usecols=['smiles']+FEATURES[1:])
        pred = read(folder/'predictions.csv', usecols=['smiles','y'])  # y here is prediction, not truth.
        assert feat.smiles.tolist() == pred.smiles.tolist() == ids.loc[j['query'],'smiles'].tolist()
        f = feat[FEATURES[1:]].copy()
        f.insert(0,'prediction',pred.y.to_numpy())
        f.index = pd.Index(j['query'], name='source_row')
        assert np.isfinite(f).all().all()
        raw = roughness(j['query'], j['fit'], ids, fps, labels, deadline)
        delta = float(np.max(np.abs(raw-f[EXTRA].to_numpy())))
        assert delta <= 1e-10, (j['name'],delta)
        max_replay = max(max_replay,delta)
        frames.append(f)
        state['replay_rows'] += len(f)
        save(out/'state.json', state)
    src, target = pd.concat(frames[:3]).sort_index(), frames[-1]
    assert src.index.tolist()==fit and target.index.tolist()==ev
    augmented = []
    for k, q in enumerate(folds):
        for l, refs in enumerate(folds):
            if k==l:
                continue
            f=src.loc[q].copy()
            f[EXTRA]=roughness(q,refs,ids,fps,labels,deadline)
            f['context']=l
            augmented.append(f)
            state['context_rows'] += len(f)
            state['new_jobs'] += 1
            save(out/'state.json',state)
    aug = pd.concat(augmented)
    assert aug.groupby(level=0).size().eq(2).all()
    targets=[target.assign(context=-1)]
    for l, refs in enumerate(folds):
        f=target.copy()
        f[EXTRA]=roughness(ev,refs,ids,fps,labels,deadline)
        targets.append(f.assign(context=l))
        state['context_rows'] += len(f)
        state['new_jobs'] += 1
        save(out/'state.json',state)
    for name, frame in [('source.csv',src),('augmented.csv',aug),('targets.csv',pd.concat(targets)),
                        ('identities.csv',ids),('absolute_error.csv',(src.prediction-labels).abs().rename('a').to_frame())]:
        frame.to_csv(out/name)
    state.update(status='prepared',replay_max_abs_difference=max_replay,fingerprint_molecules=len(fps))
    assert state['context_rows']==4561 and state['replay_rows']==1988 and state['new_jobs']==9
    save(out/'inputs.json',dict(protocol_sha256=sha(PROTOCOL),runner_sha256=sha(Path(__file__)),
         files={f.name:sha(f) for f in out.glob('*.csv')},evaluation_truth_rows_read=0))
    save(out/'state.json',state)


def evaluate(p,out,state):
    deadline=time.monotonic()+p['budgets']['evaluation_seconds']
    lock=json.loads((out/'score_lock.json').read_text())
    assert lock['protocol_sha256']==sha(PROTOCOL) and lock['runner_sha256']==sha(Path(__file__))
    assert lock['inputs_sha256']==sha(out/'inputs.json')
    for name,expected in json.loads((out/'inputs.json').read_text())['files'].items():
        assert sha(out/name)==expected
    assert sha(out/'scores.csv')==lock['scores_sha256']
    assert sha(out/'contexts.csv')==lock['contexts_sha256']
    scores=read(out/'scores.csv',index_col='source_row')
    ctx=read(out/'contexts.csv',index_col='source_row')
    ids=read(out/'identities.csv',index_col='source_row')
    target=read(out/'targets.csv',index_col='source_row').query('context == -1')
    if state['status']=='scored':
        assert sha(SOURCE)==p['source_sha256']
        truth=read_selected(SOURCE,scores.index.tolist(),['y']).y
        assert truth.index.tolist()==scores.index.tolist() and np.isfinite(truth).all()
        truth.to_csv(out/'evaluation_truth.csv')
        state.update(status='evaluating',evaluation_truth_rows_read=len(truth))
        save(out/'state.json',state)
    else:
        truth=read(out/'evaluation_truth.csv',index_col='source_row').y
        assert sha(out/'evaluation_truth.csv')==state['truth_sha256']
    frame=ids.loc[scores.index].copy()
    frame['source_row']=frame.index
    frame['prediction']=target.prediction
    frame['y']=truth
    e2=(frame.prediction-frame.y).to_numpy()**2
    curves,orders={},{}
    for arm in p['arms']:
        order=ordering(frame,scores[arm].to_numpy());orders[arm]=order
        curve=[float(np.sqrt(np.mean(e2[order[:math.ceil(c*len(frame))]]))) for c in p['coverage_grid']]
        independent,counts=independent_curve(frame,scores[arm].to_numpy(),p['coverage_grid'])
        assert np.allclose(curve,independent,rtol=0,atol=1e-12)
        curves[arm]=dict(rmse=curve,mean_six_rmse=float(np.mean(curve)),accepted=counts)
    assert len({round(c['rmse'][-1],12) for c in curves.values()})==1
    exchanges=[];private=[]
    pairs=[('H0-full',a) for a in p['arms'][1:]]+[('Hc-full','Hc-mean')]
    for baseline,arm in pairs:
        for z,c in enumerate(p['coverage_grid']):
            assert time.monotonic()<deadline
            n=math.ceil(c*len(frame)); a=set(orders[baseline][:n]);b=set(orders[arm][:n])
            common,added,removed=a&b,b-a,a-b
            assert a==common|removed and b==common|added and len(added)==len(removed)
            assert not(common&added or common&removed or added&removed)
            contribution=np.array([e2[i]*((i in added)-(i in removed))/n for i in range(len(frame))])
            delta=float(contribution.sum())
            assert math.isclose(delta,curves[arm]['rmse'][z]**2-curves[baseline]['rmse'][z]**2,abs_tol=1e-12)
            j=int(np.argmax(np.abs(contribution)))
            exchanges.append(dict(baseline=baseline,arm=arm,coverage=c,accepted=n,common=len(common),added=len(added),removed=len(removed),
                added_sse=float(e2[list(added)].sum()),removed_sse=float(e2[list(removed)].sum()),mse_delta=delta,
                largest_abs_contribution=float(abs(contribution[j])),mse_delta_without_largest=float(delta-contribution[j])))
            if arm=='Hc-mean' and baseline=='H0-full':
                private.append(pd.DataFrame({'source_row':frame.index,'coverage':c,'signed_mse_contribution':contribution}))
    diagnostics=[]
    signed=pd.concat(private).groupby('source_row').signed_mse_contribution.mean().reindex(frame.index)
    for model in ['H0','Hd','Hc']:
        block=ctx[[model+'-'+str(l) for l in range(3)]].to_numpy()
        avg=block.mean(axis=1)
        assert np.allclose(avg,scores[model+'-mean'],rtol=0,atol=1e-12)
        variance=block.var(axis=1)
        assert np.allclose(((block-np.sqrt(e2)[:,None])**2).mean(axis=1)-(avg-np.sqrt(e2))**2,variance,rtol=0,atol=1e-12)
        modified=block.copy();modified[:,0]=block[:,1]
        assert np.allclose(modified.mean(axis=1)-avg,(block[:,1]-block[:,0])/3,rtol=0,atol=1e-12)
        spread=pd.Series(variance,index=frame.index)
        diagnostics.append(dict(model=model,mean_context_score_variance=float(spread.mean()),
             spearman_variance_squared_error=float(spread.corr(pd.Series(e2,index=frame.index),method='spearman')),
             spearman_variance_signed_exchange=float(spread.corr(signed,method='spearman'))))
    candidate=curves['Hc-mean']['mean_six_rmse']
    result=dict(task=p['task'],seed=p['seed'],exploratory=True,curves=curves,
         continuation_condition_met=all(candidate<curves[a]['mean_six_rmse'] for a in p['primary_controls']),
         no_automatic_expansion=True,diagnostics=diagnostics,checks=dict(curve_points=36,exchange_rows=len(exchanges),
         partitions=True,equal_swaps=True,exchange_identity=True,full_acceptance_equal=True,mean_response_identity=True))
    if state['status']=='completed':
        previous=json.loads((out/'results.json').read_text())
        assert previous==result
        assert np.allclose(read(out/'exchanges.csv').select_dtypes('number'),pd.DataFrame(exchanges).select_dtypes('number'),rtol=0,atol=1e-12)
        return result
    save(out/'results.json',result)
    pd.DataFrame(exchanges).to_csv(out/'exchanges.csv',index=False)
    pd.concat(private).to_csv(out/'individual_contributions.csv',index=False)
    state.update(status='completed',truth_sha256=sha(out/'evaluation_truth.csv'),results_sha256=sha(out/'results.json'))
    save(out/'state.json',state)
    return result


def fit_score(p,out,state):
    manifest=json.loads((out/'inputs.json').read_text())
    assert manifest['protocol_sha256']==sha(PROTOCOL) and manifest['runner_sha256']==sha(Path(__file__))
    for name,expected in manifest['files'].items():
        assert sha(out/name)==expected
    source=read(out/'source.csv',index_col='source_row')
    aug=read(out/'augmented.csv',index_col='source_row')
    target=read(out/'targets.csv',index_col='source_row')
    absolute=read(out/'absolute_error.csv',index_col='source_row').a
    assert list(source.columns)==FEATURES and np.isfinite(aug[FEATURES]).all().all()
    deadline=time.monotonic()+p['budgets']['fit_predict_seconds']
    state['status']='fitting';save(out/'state.json',state)
    scores,contexts={},{}
    for name,train in [('H0',source),('Hd',pd.concat([source,source])),('Hc',aug)]:
        assert time.monotonic()<deadline and state['risk_fits_started']<3
        state['risk_fits_started']+=1;save(out/'state.json',state)
        model=RandomForestRegressor(**p['risk_rf'])
        weights=np.ones(len(train)) if name=='H0' else np.full(len(train),.5)
        model.fit(train[FEATURES],absolute.loc[train.index].to_numpy(),sample_weight=weights)
        state['risk_predict_calls_started']+=1;save(out/'state.json',state)
        values=model.predict(target[FEATURES]).reshape(4,585)
        assert time.monotonic()<deadline, 'Fit/predict budget exceeded'
        assert np.isfinite(values).all()
        scores[name+'-full']=values[0]
        scores[name+'-mean']=values[1:].mean(axis=0)
        for l in range(3):contexts[name+'-'+str(l)]=values[l+1]
        state['risk_prediction_rows']+=values.size
        save(out/'state.json',state)
    index=target.index[:585]
    assert all(target.index[b*585:(b+1)*585].equals(index) for b in range(4))
    pd.DataFrame(scores,index=index)[p['arms']].to_csv(out/'scores.csv')
    pd.DataFrame(contexts,index=index).to_csv(out/'contexts.csv')
    save(out/'score_lock.json',dict(scores_sha256=sha(out/'scores.csv'),contexts_sha256=sha(out/'contexts.csv'),
         inputs_sha256=sha(out/'inputs.json'),protocol_sha256=sha(PROTOCOL),runner_sha256=sha(Path(__file__)),evaluation_truth_rows_read=0))
    state.update(status='scored');save(out/'state.json',state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['selfcheck','prepare','run','audit','worker-prepare','worker-run','worker-evaluate'])
    args=parser.parse_args()
    if args.mode=='selfcheck':
        f=pd.DataFrame(dict(canonical=['b','a','a'],source_row=[0,2,1]))
        assert ordering(f,np.zeros(3))==[2,1,0]
        _,raw=geometry(np.array([.5,.5,0.]),np.eye(3),np.array([0.,0.,2.]))
        assert math.isclose(raw[1],4/3)
        assert math.ceil(.5*3)==2
        print('PASS: reused roughness, tied ranking, ceil; real fits=0');return
    p,base,out=pinned()
    if args.mode=='prepare':
        out.mkdir(exist_ok=False)
        state=dict(status='preparing',risk_fits_started=0,risk_predict_calls_started=0,risk_prediction_rows=0,
             replay_rows=0,context_rows=0,new_jobs=0,evaluation_truth_rows_read=0,automatic_retries=0)
        save(out/'state.json',state)
    else:
        state=json.loads((out/'state.json').read_text())
    if args.mode.startswith('worker-'):
        if args.mode=='worker-prepare':
            assert state['status']=='preparing' and state['context_rows']==state['replay_rows']==0
            prepare(p,base,out,state)
        elif args.mode=='worker-run':
            assert state['status']=='prepared' and state['risk_fits_started']==0
            fit_score(p,out,state)
        else:
            assert state['status']=='scored' and state['evaluation_truth_rows_read']==0
            evaluate(p,out,state)
        return
    if args.mode=='audit':
        assert state['status']=='completed' and sha(out/'results.json')==state['results_sha256']
        print(json.dumps(evaluate(p,out,state),indent=2));return
    assert (args.mode,state['status']) in [('prepare','preparing'),('run','prepared')]
    log=out/(args.mode+'_log')
    assert not log.exists(), 'Refuse repeat execution'
    started=time.monotonic()
    cap=p['budgets']['prepare_seconds'] if args.mode=='prepare' else p['budgets']['fit_predict_seconds']
    cap=min(cap,p['budgets']['whole_seconds']-state.get('prepare_seconds',0))
    try:
        run_command([sys.executable,str(Path(__file__).resolve()),'worker-'+args.mode],log/'compute',cap)
        if args.mode=='run':
            remaining=p['budgets']['whole_seconds']-state.get('prepare_seconds',0)-(time.monotonic()-started)
            run_command([sys.executable,str(Path(__file__).resolve()),'worker-evaluate'],log/'evaluate',min(remaining,p['budgets']['evaluation_seconds']))
        state=json.loads((out/'state.json').read_text())
        state[args.mode+'_seconds']=time.monotonic()-started
        save(out/'state.json',state)
        print(json.dumps(state,indent=2))
    except BaseException as error:
        state=json.loads((out/'state.json').read_text())
        state.update(status='failed_or_interrupted',error=repr(error),failed_stage=args.mode)
        save(out/'state.json',state)
        raise


if __name__=='__main__':
    main()
