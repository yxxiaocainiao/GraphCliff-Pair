"""M56: count a fixed structural-pair label subset. Stdlib, no predictions."""
import argparse
import ast
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
env = dict(Counter=Counter, hashlib=hashlib, json=json)
for path, names in [
    (ROOT/'docs/research/model_conditioned_pair_diagnostic_20261007/diagnose.py', ['sha','save']),
    (ROOT/'docs/research/query_reference_support_20261007/check_support.py', ['summarize'])]:
    tree = ast.parse(path.read_text(encoding='utf-8'))
    selected = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),env)
sha, save, summarize = (env[k] for k in ['sha','save','summarize'])


def qualifies(yq, yr):
    assert math.isfinite(yq) and math.isfinite(yr)
    return abs(yq-yr) >= 1.


def selfcheck():
    assert qualifies(0.,1.) and qualifies(1.,0.) and qualifies(-2.,-.999)
    assert not qualifies(0.,.999) and not qualifies(3.,3.)
    for x in [float('nan'),float('inf')]:
        try: qualifies(x,0.)
        except AssertionError: pass
        else: raise AssertionError('Nonfinite label accepted')
    assert summarize([], [1,2])['pairs'] == 0
    assert summarize([(1,10,.9),(1,11,.95)], [1,2])['greedy_disjoint_pairs'] == 1
    print('PASS: threshold boundary/sign, finite labels, empty subset and repeated query')


def labels(path, wanted, expected_ids):
    seen = set(); selected = {}
    with path.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            i = int(row['source_row']); assert i not in seen; seen.add(i)
            if i in wanted:
                value=float(row['y']); assert math.isfinite(value); selected[i]=value
    assert seen == set(expected_ids) and set(selected) == wanted
    return selected


def analyze(records, manifest, original):
    roles={}; references={}
    for role in ['calibration','evaluation']:
        rows=records[role]; original_edges=[(r['q'],r['r'],r['similarity']) for r in rows]
        assert summarize(original_edges,manifest['roles'][role]) == original['roles'][role]['support']
        assert all(r['r'] in set(manifest['roles']['fit']) for r in rows)
        subset=[(r['q'],r['r'],r['similarity']) for r in rows if qualifies(r['query_y'],r['reference_y'])]
        support=summarize(subset,manifest['roles'][role]); all_support=original['roles'][role]['support']
        roles[role]=dict(all_structural=all_support,delta_at_least_1=support,
            below_one_pairs=len(rows)-len(subset),
            fraction_of_structural_pairs=len(subset)/len(rows) if rows else None,
            fraction_of_supported_queries=support['queries_with_pairs']/all_support['queries_with_pairs'] if all_support['queries_with_pairs'] else None)
        references[role]={r for q,r,s in subset}
    return dict(roles=roles,distinct_required_references=len(set.union(*references.values())),
        references_shared_across_roles=len(references['calibration']&references['evaluation']))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=['selfcheck','run','audit']);a=ap.parse_args()
    if a.mode=='selfcheck':selfcheck();return
    p=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/k)==h for k,h in p['inputs_sha256'].items())
    dest=ROOT/p['output'];assert dest.parent==ROOT/'artifacts'
    manifest=json.loads((ROOT/p['manifest']).read_text(encoding='utf-8'))
    original=json.loads((ROOT/p['prior_results']).read_text(encoding='utf-8'))
    pairs=json.loads((ROOT/p['pairs']).read_text(encoding='utf-8'))
    if a.mode=='audit':
        state=json.loads((dest/'execution.json').read_text(encoding='utf-8'));assert state['status']=='completed'
        records=json.loads((dest/'labeled_pairs.json').read_text(encoding='utf-8'))
        assert sha(dest/'labeled_pairs.json')==state['labeled_pairs_sha256']
        assert analyze(records,manifest,original)==state['summary']
        base=ROOT/p['cache_root']/p['task']
        reference=labels(base/'oof.csv',{r for rows in pairs.values() for q,r,s in rows},manifest['roles']['fit'])
        for role,rows in records.items():
            query=labels(base/(role+'.csv'),{q for q,r,s in pairs[role]},manifest['roles'][role])
            assert all(r['query_y']==query[r['q']] and r['reference_y']==reference[r['r']] for r in rows)
            assert [(r['q'],r['r'],r['similarity']) for r in rows] == [tuple(e) for e in pairs[role]]
            # Independent scalar count; no new pair selection or model operations.
            expected=sum(abs(r['query_y']-r['reference_y'])>=1. for r in rows)
            assert expected==state['summary']['roles'][role]['delta_at_least_1']['pairs']
        save(HERE/'results.json',{k:v for k,v in state.items() if k!='labeled_pairs_sha256'})
        print('AUDIT PASSED: frozen 87 pairs, both roles and activity-subset counts');return
    assert not dest.exists(),'No overwrite or automatic restart'
    dest.mkdir();start=time.monotonic();state=dict(status='running',code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    save(dest/'execution.json',state)
    try:
        wanted_ref={r for rows in pairs.values() for q,r,s in rows}
        assert len(wanted_ref)<=82 and sum(map(len,pairs.values()))==87
        base=ROOT/p['cache_root']/p['task'];reference=labels(base/'oof.csv',wanted_ref,manifest['roles']['fit'])
        records={};query_count=0
        for role,edges in pairs.items():
            wanted_query={q for q,r,s in edges};query_count+=len(wanted_query)
            query=labels(base/(role+'.csv'),wanted_query,manifest['roles'][role])
            records[role]=[dict(q=q,r=r,similarity=s,query_y=query[q],reference_y=reference[r]) for q,r,s in edges]
        assert query_count<=74
        save(dest/'labeled_pairs.json',records)
        summary=analyze(records,manifest,original)
        assert all(sha(ROOT/k)==h for k,h in p['inputs_sha256'].items()) and time.monotonic()-start<p['wall_seconds']
        state.update(status='completed',summary=summary,selected_reference_labels=len(reference),selected_query_labels=query_count,
            labeled_pairs_sha256=sha(dest/'labeled_pairs.json'),wall_seconds=time.monotonic()-start,
            fits=0,model_loads=0,model_predict_calls=0,fingerprint_recomputations=0,official_test_rows_read=0)
        save(dest/'execution.json',state);print(json.dumps(summary),flush=True)
    except BaseException as e:
        state.update(status='failed',error=repr(e),wall_seconds=time.monotonic()-start);save(dest/'execution.json',state);raise


if __name__=='__main__':main()
