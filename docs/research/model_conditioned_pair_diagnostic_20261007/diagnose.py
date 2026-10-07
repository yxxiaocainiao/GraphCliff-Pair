"""M54: within-role cached-prediction diagnosis, not a deployable risk method."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def structural_pairs(fps):
    edges = []; identical = 0
    for i, fp in enumerate(fps):
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps[i+1:])
        for j, similarity in enumerate(sims, i+1):
            if similarity == 1.:
                identical += 1
            elif similarity >= .9:
                edges.append((i, j, float(similarity)))
    return sorted(edges, key=lambda e: (-e[2], e[0], e[1])), identical


def matching(edges):
    used = set(); result = []
    for i, j, s in edges:
        if i not in used and j not in used:
            result.append((i, j, s)); used.update([i, j])
    return result


def summarize(frame):
    if frame.empty:
        return dict(pairs=0, unique_endpoints=0, pair_rmse=None, contrast_rms=None,
                    common_bias_rms=None, contrast_mse_share=None, delta_mae=None,
                    sign_accuracy=None, correlations=None)
    contrast = float(np.mean(frame.contrast2)); common = float(np.mean(frame.common2))
    mse = float(np.mean(frame.pair_mse))
    assert math.isclose(mse, contrast+common, abs_tol=1e-12)
    corrs = {}
    for feature in ['mean_sali', 'mean_augmented_risk']:
        for target in ['contrast2', 'common2', 'pair_mse']:
            # Descriptive only: overlapping endpoints do not create independent samples.
            value = frame[feature].rank().corr(frame[target].rank()) if len(frame) >= 3 and frame[feature].nunique() > 1 and frame[target].nunique() > 1 else None
            corrs[feature+'_vs_'+target] = float(value) if value is not None and np.isfinite(value) else None
    cliff = frame.abs_true_delta >= 1.
    return dict(pairs=len(frame), unique_endpoints=len(set(frame.source_i)|set(frame.source_j)),
                pair_rmse=math.sqrt(mse), contrast_rms=math.sqrt(contrast), common_bias_rms=math.sqrt(common),
                contrast_mse_share=contrast/mse if mse else None, delta_mae=float(frame.abs_delta_error.mean()),
                sign_accuracy=float(frame.loc[cliff, 'correct_sign'].mean()) if cliff.any() else None,
                correlations=corrs)


def selfcheck():
    y = np.array([0., 2.])
    for pred, expected in [(np.array([5., 7.]), [0., 25., 25.]), (np.array([0., 0.]), [1., 1., 2.])]:
        e = pred-y
        values = [(e[0]-e[1])**2/4, (e[0]+e[1])**2/4, float(np.mean(e**2))]
        assert np.allclose(values, expected) and math.isclose(values[0]+values[1], values[2])
    fps = []
    for n in [9, 10, 9]:
        fp = DataStructs.ExplicitBitVect(16)
        for i in range(n): fp.SetBit(i)
        fps.append(fp)
    edges, identical = structural_pairs(fps)
    assert edges == [(0, 1, .9), (1, 2, .9)] and identical == 1
    assert matching(edges) == [(0, 1, .9)]
    assert summarize(pd.DataFrame())['pairs'] == 0
    print('PASS: pair-MSE identity, common-bias blind spot, boundary, identical fingerprints, disjoint matching, empty subset')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['selfcheck', 'run', 'audit'])
    a = ap.parse_args()
    if a.mode == 'selfcheck': selfcheck(); return
    p = json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/k) == h for k,h in p['inputs_sha256'].items())
    assert all(importlib.metadata.version(k) == v for k,v in p['software'].items())
    dest = ROOT/p['output']; assert dest.parent == ROOT/'artifacts'
    if a.mode == 'audit':
        state = json.loads((dest/'execution.json').read_text(encoding='utf-8')); assert state['status'] == 'completed'
        deltas = []
        for item in state['batches']:
            pairs = pd.read_csv(dest/item['file'], float_precision='round_trip')
            assert sha(dest/item['file']) == item['sha256']
            manifest = json.loads((ROOT/p['manifest_root']/(item['task']+'.json')).read_text(encoding='utf-8'))
            assert set(pairs.source_i)|set(pairs.source_j) <= set(manifest['roles'][item['role']])
            assert not pairs[['source_i','source_j']].duplicated().any()
            independent = pairs[pairs.disjoint.astype(bool)]
            assert len(set(independent.source_i)|set(independent.source_j)) == 2*len(independent)
            for row in pairs.itertuples():
                assert .9 <= row.similarity < 1. and row.source_i != row.source_j
                delta = (row.pred_i-row.pred_j)-(row.y_i-row.y_j)
                common = ((row.pred_i-row.y_i)+(row.pred_j-row.y_j))/2
                mse = ((row.pred_i-row.y_i)**2+(row.pred_j-row.y_j)**2)/2
                assert row.abs_true_delta == abs(row.y_i-row.y_j)
                assert row.correct_sign == ((row.pred_i-row.pred_j)*(row.y_i-row.y_j)>0)
                deltas.extend([abs(delta**2/4-row.contrast2), abs(common**2-row.common2), abs(mse-row.pair_mse)])
            for entry in item['summaries']:
                subset = pairs if entry['pairing'] == 'all' else pairs[pairs.disjoint.astype(bool)]
                if entry['subset'] == 'delta_at_least_1': subset = subset[subset.abs_true_delta >= 1.]
                fresh = summarize(subset)
                assert fresh == entry['values']
        maximum = max(deltas, default=0.); assert maximum <= 1e-12
        save(HERE/'results.json', dict(passed=True, batches=[{k:v for k,v in x.items() if k not in ['file','sha256']} for x in state['batches']],
            code_commit=state['code_commit'], wall_seconds=state['wall_seconds'], independent_pair_decomposition_max_abs=maximum,
            fits=0, model_loads=0, model_predict_calls=0, official_test_rows_read=0, source_sha256=p['inputs_sha256']))
        print('AUDIT PASSED:', sum(b['pairs'] for b in state['batches']), 'same-role pairs; six batches')
        return
    assert not dest.exists(), 'No overwriting or automatic restart'
    dest.mkdir(); start = time.monotonic()
    state = dict(status='running', batches=[], code_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip())
    save(dest/'execution.json', state)
    try:
        source = ROOT/p['author_features']; tree = ast.parse(source.read_text(encoding='utf-8'))
        fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'featurize')
        env = dict(Chem=Chem, np=np, DataStructs=DataStructs, _gen=rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048))
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),env)
        for task in p['tasks']:
            manifest = json.loads((ROOT/p['manifest_root']/(task+'.json')).read_text(encoding='utf-8')); seen = set()
            for role in ['calibration', 'evaluation']:
                frame = pd.read_csv(ROOT/p['cache_root']/task/(role+'.csv'),float_precision='round_trip').sort_values(['canonical','source_row']).reset_index(drop=True)
                assert frame.source_row.is_unique and frame.canonical.is_unique
                assert set(frame.source_row) == set(manifest['roles'][role])
                assert not seen & set(frame.canonical); seen.update(frame.canonical)
                assert np.isfinite(frame[['y','prediction','sali_mean','augmented_risk']].to_numpy()).all()
                assert sum(x['fingerprint_rows'] for x in state['batches'])+len(frame) <= p['max_fingerprint_rows']
                fps, _, _, ok = env['featurize'](frame.smiles.tolist()); assert ok.all()
                edges, identical = structural_pairs(fps); independent = {(i,j) for i,j,_ in matching(edges)}
                records = []
                for i,j,s in edges:
                    assert time.monotonic()-start < p['wall_seconds']
                    x,z = frame.iloc[i], frame.iloc[j]; ei,ej = x.prediction-x.y,z.prediction-z.y
                    dy = x.y-z.y; dp = x.prediction-z.prediction
                    contrast,common,mse = (ei-ej)**2/4,(ei+ej)**2/4,(ei**2+ej**2)/2
                    assert math.isclose(contrast+common,mse,abs_tol=1e-12)
                    records.append(dict(source_i=int(x.source_row),source_j=int(z.source_row),similarity=s,
                        y_i=x.y,y_j=z.y,pred_i=x.prediction,pred_j=z.prediction,abs_true_delta=abs(dy),
                        abs_delta_error=abs(dp-dy),correct_sign=bool(dp*dy>0),contrast2=contrast,common2=common,pair_mse=mse,
                        mean_sali=(x.sali_mean+z.sali_mean)/2,mean_augmented_risk=(x.augmented_risk+z.augmented_risk)/2,disjoint=(i,j) in independent))
                columns=['source_i','source_j','similarity','y_i','y_j','pred_i','pred_j','abs_true_delta','abs_delta_error','correct_sign','contrast2','common2','pair_mse','mean_sali','mean_augmented_risk','disjoint']
                pairs = pd.DataFrame(records,columns=columns); filename=task+'_'+role+'.csv'; pairs.to_csv(dest/filename,index=False)
                summaries=[]
                for pairing in ['all','disjoint']:
                    selected = pairs if pairing == 'all' else pairs[pairs.disjoint.astype(bool)]
                    for subset in ['all_structural','delta_at_least_1']:
                        group = selected if subset == 'all_structural' else selected[selected.abs_true_delta >= 1.]
                        summaries.append(dict(pairing=pairing,subset=subset,values=summarize(group)))
                state['batches'].append(dict(task=task,role=role,molecules=len(frame),fingerprint_rows=len(fps),pairs=len(pairs),
                    excluded_identical_fingerprint_pairs=identical,file=filename,sha256=sha(dest/filename),summaries=summaries))
                save(dest/'execution.json',state); print('DIAGNOSED',task,role,len(pairs),'pairs',flush=True)
        assert time.monotonic()-start < p['wall_seconds'] and len(state['batches']) == 6
        assert all(sha(ROOT/k) == h for k,h in p['inputs_sha256'].items())
        state.update(status='completed',wall_seconds=time.monotonic()-start); save(dest/'execution.json',state)
    except BaseException as e:
        state.update(status='failed',error=repr(e),wall_seconds=time.monotonic()-start); save(dest/'execution.json',state); raise


if __name__ == '__main__': main()
