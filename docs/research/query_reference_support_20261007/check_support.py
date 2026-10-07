"""M55: label-free, one-task query/reference structural support; no models."""
import argparse
import ast
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
from collections import Counter

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pair_diagnostic', ROOT/'docs/research/model_conditioned_pair_diagnostic_20261007/diagnose.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
sha, save = helper.sha, helper.save


def summarize(edges, query_ids):
    assert len(set(query_ids)) == len(query_ids)
    assert all(q in set(query_ids) and .9 <= s < 1 for q,r,s in edges)
    assert len({(q,r) for q,r,s in edges}) == len(edges)
    qdegree = Counter(q for q,r,s in edges); rdegree = Counter(r for q,r,s in edges)
    used_q = set(); used_r = set(); chosen = []
    for q,r,s in sorted(edges, key=lambda e:(-e[2],e[0],e[1])):
        if q not in used_q and r not in used_r:
            used_q.add(q); used_r.add(r); chosen.append((q,r,s))
    result = dict(query_rows=len(query_ids), pairs=len(edges), queries_with_pairs=len(qdegree),
        query_coverage=len(qdegree)/len(query_ids) if query_ids else None, unique_references=len(rdegree),
        query_degree_histogram={str(k):v for k,v in sorted(Counter(qdegree[q] for q in query_ids).items())},
        max_reference_degree=max(rdegree.values(),default=0), greedy_disjoint_pairs=len(chosen))
    assert sum(result['query_degree_histogram'].values()) == len(query_ids)
    assert sum(int(k)*v for k,v in result['query_degree_histogram'].items()) == len(edges)
    return result


def selfcheck():
    x = summarize([(1,10,.9),(1,11,.95),(2,11,.95),(3,12,.9)], [1,2,3,4])
    assert x['pairs'] == 4 and x['queries_with_pairs'] == 3 and x['greedy_disjoint_pairs'] == 2
    assert x['query_degree_histogram'] == {'0':1,'1':2,'2':1}
    assert summarize([], [1,2])['query_coverage'] == 0
    for bad in [[(1,10,1.)],[(1,10,.89)],[(9,10,.9)],[(1,10,.9)]*2]:
        try: summarize(bad,[1,2])
        except AssertionError: pass
        else: raise AssertionError('Invalid support accepted')
    print('PASS: bipartite coverage, duplicate/identity/boundary guards, degree accounting and greedy matching')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=['selfcheck','run','audit']);a=ap.parse_args()
    if a.mode == 'selfcheck':selfcheck();return
    p=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/k)==h for k,h in p['inputs_sha256'].items())
    assert all(importlib.metadata.version(k)==v for k,v in p['software'].items())
    dest=ROOT/p['output'];assert dest.parent==ROOT/'artifacts'
    manifest=json.loads((ROOT/p['manifest']).read_text(encoding='utf-8'))
    if a.mode == 'audit':
        state=json.loads((dest/'execution.json').read_text(encoding='utf-8'));assert state['status']=='completed'
        edges=json.loads((dest/'pairs.json').read_text(encoding='utf-8'));assert sha(dest/'pairs.json')==state['pairs_sha256']
        reference_sets={}
        for role in ['calibration','evaluation']:
            rows=edges[role];assert all(r in set(manifest['roles']['fit']) for q,r,s in rows)
            assert summarize(rows,manifest['roles'][role])==state['roles'][role]['support']
            reference_sets[role]={r for q,r,s in rows}
        shared=len(reference_sets['calibration']&reference_sets['evaluation']);assert shared==state['shared_reference_count']
        save(HERE/'results.json',{k:v for k,v in state.items() if k!='pairs_sha256'})
        print('AUDIT PASSED: both roles, identity, coverage, degree sums and reference reuse');return
    assert not dest.exists(),'Refuse overwrite or automatic restart'
    dest.mkdir();start=time.monotonic();state=dict(status='running',roles={},task=p['task'],
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    save(dest/'execution.json',state)
    try:
        base=ROOT/p['cache_root']/p['task'];frames={}
        for role in ['oof','calibration','evaluation']:
            columns=['source_row','smiles','canonical']+(['nn_sim'] if role!='oof' else [])
            f=pd.read_csv(base/(role+'.csv'),usecols=columns,float_precision='round_trip')
            assert f.source_row.is_unique and f.canonical.is_unique
            assert set(f.source_row)==set(manifest['roles']['fit' if role=='oof' else role]);frames[role]=f
        assert sum(len(f) for f in frames.values())==p['max_fingerprint_rows']
        assert len(set().union(*(set(f.canonical) for f in frames.values())))==p['max_fingerprint_rows']
        jobs=json.loads((ROOT/p['cache_root']/'prepare.json').read_text(encoding='utf-8'))['jobs']
        job=next(x for x in jobs if x['task']==p['task'] and x['name']=='full')
        assert job['fit']==manifest['roles']['fit'] and job['query']==manifest['roles']['calibration']+manifest['roles']['evaluation']
        fixture=pd.read_csv(base/'full/features/data/fixture.csv',usecols=['smiles','split'])
        reference=fixture[fixture.split.eq('train')].smiles.tolist();query=fixture[fixture.split.eq('test')].smiles.tolist()
        by_id=pd.concat(list(frames.values())).set_index('source_row')
        assert by_id.index.is_unique
        assert reference==by_id.loc[job['fit'],'smiles'].tolist() and query==by_id.loc[job['query'],'smiles'].tolist()
        assert len(fixture)==len(reference)+len(query)
        source=ROOT/p['author_features'];tree=ast.parse(source.read_text(encoding='utf-8'))
        fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='featurize')
        env=dict(Chem=Chem,np=np,DataStructs=DataStructs,_gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048))
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),env)
        fps,_,_,ok=env['featurize'](reference);qfps,_,_,qok=env['featurize'](query);assert ok.all() and qok.all()
        edges={role:[] for role in ['calibration','evaluation']};identical=Counter();replay=0.
        role_for_id={i:role for role in edges for i in manifest['roles'][role]}
        for qid,qfp in zip(job['query'],qfps):
            assert time.monotonic()-start<p['wall_seconds']
            sims=np.array(DataStructs.BulkTanimotoSimilarity(qfp,fps));role=role_for_id[qid]
            replay=max(replay,abs(float(sims.max())-float(by_id.loc[qid,'nn_sim'])))
            identical[role]+=int(np.sum(sims==1.))
            edges[role].extend((qid,job['fit'][int(i)],float(sims[i])) for i in np.flatnonzero((sims>=.9)&(sims<1.)))
        assert replay<=1e-10
        for role in edges:
            state['roles'][role]=dict(support=summarize(edges[role],manifest['roles'][role]),excluded_identical_pairs=identical[role])
        state['shared_reference_count']=len({r for q,r,s in edges['calibration']}&{r for q,r,s in edges['evaluation']})
        save(dest/'pairs.json',edges)
        assert all(sha(ROOT/k)==h for k,h in p['inputs_sha256'].items()) and time.monotonic()-start<p['wall_seconds']
        state.update(status='completed',reference_rows=len(reference),fingerprint_rows=len(fps)+len(qfps),
            nn_sim_replay_max_abs=replay,wall_seconds=time.monotonic()-start,pairs_sha256=sha(dest/'pairs.json'),
            fits=0,model_loads=0,model_predict_calls=0,labels_used=False,official_test_rows_read=0)
        save(dest/'execution.json',state);print(json.dumps(state['roles']),flush=True)
    except BaseException as e:
        state.update(status='failed',error=repr(e),wall_seconds=time.monotonic()-start);save(dest/'execution.json',state);raise


if __name__=='__main__':main()
