"""M52: three fixed nine-dimensional risk arms; never load a point predictor."""
import argparse
import ast
import importlib.metadata
import json
import math
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from run_phase_a import ROOT, GENERIC, EXTRA, save, sha, cq_function, metrics

DOC = ROOT/'docs/research/query_pair_pilot_20261007'
BASE = ROOT/'artifacts/reliability_phase_a_20261005_v3'
NEW = ['eta_mean', 'joint', 'separate', 'radial']


def read(path):
    return pd.read_csv(path, float_precision='round_trip')


def geometry(sim, pair, y):
    """Only reference labels enter. Keep original neighbor order, including ties."""
    i, j = np.triu_indices(len(y), 1)
    a, b, c = 1-sim[i], 1-sim[j], 1-pair[i, j]
    assert len(i) and np.isfinite([a, b, c]).all()
    assert np.isfinite(y).all() and np.all((pair >= 0) & (pair <= 1))
    assert np.all((sim >= 0) & (sim <= 1))
    assert np.all(np.abs(a-b) <= c+1e-12) and np.all(c <= a+b+1e-12)
    eta = np.clip(np.divide(c-np.abs(a-b), a+b, out=np.zeros_like(c), where=a+b > 0), 0, 1)
    sali = np.abs(y[i]-y[j])/np.maximum(c, 1e-3)
    joint = float(np.mean(eta*sali))
    separate = float(eta.mean()*sali.mean())
    covariance = float(np.mean((eta-eta.mean())*(sali-sali.mean())))
    assert math.isclose(joint, separate+covariance, abs_tol=1e-10)
    assert -1e-12 <= joint <= sali.mean()+1e-10
    return [float(eta.mean()), joint, separate, float(np.mean(sim[i]*sim[j]*sali))], [float(np.std(y)), float(sali.mean())]


def selfcheck():
    # Compare vectorized production math with the existing finite-bitset example.
    sim = np.array([.5, .5, 0.])
    pair = np.eye(3)
    values, raw = geometry(sim, pair, np.array([0., 0., 2.]))
    assert np.allclose(values, [5/9, 4/9, 20/27, 0.], atol=1e-12)
    assert math.isclose(raw[1], 4/3)
    assert geometry(np.array([.5, .5]), np.ones((2, 2)), np.array([0., 2.]))[0][1] == 0
    try:
        geometry(np.array([.9, .9]), np.eye(2), np.array([0., 2.]))
    except AssertionError:
        pass
    else:
        raise AssertionError('Illegal triangle accepted')
    f = pd.DataFrame(dict(canonical=['b', 'a', 'a'], source_row=[0, 2, 1], prediction=[3., 2., 1.], y=[0.]*3))
    curve, count = independent_curve(f, np.zeros(3), [.5, 1.])
    assert count == [2, 3] and math.isclose(curve[0], math.sqrt(2.5))
    print('SELF CHECK PASSED: geometry, blind spot, invalid triangle, ties and ceil')


def independent_curve(frame, scores, grid):
    # Independent scalar replay of cached scores, without the shared metrics helper.
    rows = list(zip(map(float, scores), frame.canonical, map(int, frame.source_row),
                    map(float, frame.prediction), map(float, frame.y)))
    rows.sort(key=lambda r: r[:3])
    counts = [math.ceil(rho*len(rows)) for rho in grid]
    return [math.sqrt(math.fsum((r[3]-r[4])**2 for r in rows[:n])/n) for n in counts], counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['selfcheck', 'prepare', 'run', 'audit'])
    args = ap.parse_args()
    if args.mode == 'selfcheck':
        selfcheck(); return
    p = json.loads((DOC/'protocol.json').read_text(encoding='utf-8'))
    dest = ROOT/p['output']
    assert dest.parent == ROOT/'artifacts'
    inputs = {ROOT/k: v for k, v in p['inputs_sha256'].items()}
    inputs[DOC/'protocol.json'] = sha(DOC/'protocol.json')
    assert all(sha(f) == h for f, h in inputs.items()), 'Frozen input changed'
    assert all(importlib.metadata.version(k) == v for k, v in p['software'].items())
    fixed = json.loads((ROOT/'experiments/reliability_base/fixed_protocol.json').read_text(encoding='utf-8'))
    start = time.monotonic()
    if args.mode == 'prepare':
        assert not dest.exists(), 'Refuse overwriting or resuming'
        dest.mkdir()
        state = dict(status='preparing', jobs=[], risk_fits_started=0,
                     code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
        save(dest/'preparation.json', state)
        try:
            author = ROOT/p['author_features']
            tree = ast.parse(author.read_text(encoding='utf-8'))
            fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'featurize')
            env = dict(Chem=Chem, np=np, DataStructs=DataStructs,
                       _gen=rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048))
            exec(compile(ast.Module(body=[fn], type_ignores=[]), str(author), 'exec'), env)
            jobs = json.loads((BASE/'prepare.json').read_text(encoding='utf-8'))['jobs']
            for task in p['tasks']:
                manifest = json.loads((ROOT/'artifacts/reliability_protocol_20261005_v2'/(task+'.json')).read_text(encoding='utf-8'))
                roles = {role: read(BASE/task/(role+'.csv')) for role in ['oof', 'calibration', 'evaluation']}
                for role, frame in roles.items():
                    assert frame.source_row.is_unique and frame.canonical.is_unique
                    expected = manifest['roles']['fit' if role == 'oof' else role]
                    assert set(frame.source_row) == set(expected)
                    assert np.isfinite(frame[GENERIC+EXTRA+['y']].to_numpy()).all()
                assert sum(len(set(f.canonical)) for f in roles.values()) == len(set().union(*(set(f.canonical) for f in roles.values())))
                by_id = pd.concat(list(roles.values())).set_index('source_row')
                frames = []; out = dest/task; out.mkdir()
                for name in ['fold0', 'fold1', 'fold2', 'full']:
                    tick = time.monotonic()
                    job = next(j for j in jobs if j['task'] == task and j['name'] == name)
                    expected_query = manifest['oof_folds'][name[-1]] if name != 'full' else manifest['roles']['calibration']+manifest['roles']['evaluation']
                    expected_fit = sorted(set(manifest['roles']['fit'])-set(expected_query)) if name != 'full' else manifest['roles']['fit']
                    assert job['fit'] == expected_fit and job['query'] == expected_query
                    fixture = read(BASE/task/name/'features/data/fixture.csv')
                    ref = fixture[fixture.split.eq('train')].reset_index(drop=True)
                    query = fixture[fixture.split.eq('test')].reset_index(drop=True)
                    joined = read(BASE/task/name/'joined.csv')
                    assert len(fixture) == len(ref)+len(query)
                    assert ref.smiles.tolist() == by_id.loc[job['fit'], 'smiles'].tolist()
                    assert np.allclose(ref.y, by_id.loc[job['fit'], 'y'], rtol=0, atol=1e-12)
                    assert query.smiles.tolist() == joined.smiles.tolist() == by_id.loc[job['query'], 'smiles'].tolist()
                    assert joined.source_row.tolist() == job['query']
                    assert not set(by_id.loc[job['fit'], 'canonical']) & set(joined.canonical)
                    fps, _, _, ok = env['featurize'](ref.smiles.tolist())
                    qfps, _, sizes, qok = env['featurize'](query.smiles.tolist())
                    assert ok.all() and qok.all() and np.array_equal(sizes, joined.mol_size)
                    values = []; replay = []
                    for qfp in qfps:
                        assert time.monotonic()-start < p['budgets']['prepare_wall_seconds']
                        sims = np.array(DataStructs.BulkTanimotoSimilarity(qfp, fps))
                        nn = np.argsort(-sims)[:10]
                        selected = [fps[j] for j in nn]
                        pair = np.array([DataStructs.BulkTanimotoSimilarity(fp, selected) for fp in selected])
                        v, raw = geometry(sims[nn], pair, ref.y.to_numpy()[nn])
                        values.append(v); replay.append(raw+[float(sims[nn].max()), float(sims[nn].mean())])
                    delta = float(np.max(np.abs(np.asarray(replay)-joined[EXTRA+['nn_sim', 'local_dens']].to_numpy())))
                    assert delta <= 1e-10, (task, name, delta)
                    joined[NEW] = np.asarray(values)
                    joined.to_csv(out/(name+'.csv'), index=False); frames.append(joined)
                    state['jobs'].append(dict(task=task, job=name, reference_rows=len(ref), query_rows=len(query),
                                              raw_replay_max_abs=delta, seconds=time.monotonic()-tick))
                    save(dest/'preparation.json', state)
                    print('PREPARED', task, name, len(query), 'rows', flush=True)
                combined = pd.concat(frames[:3]).set_index('source_row')
                full = frames[3].set_index('source_row')
                for role, frame in roles.items():
                    derived = combined if role == 'oof' else full
                    frame[NEW] = derived.loc[frame.source_row, NEW].to_numpy()
                    assert all(frame[k].nunique() > 1 for k in NEW)
                    frame.to_csv(out/(role+'.csv'), index=False)
            assert all(sha(f) == h for f, h in inputs.items())
            state.update(status='prepared', seconds=time.monotonic()-start,
                         input_sha256={str(f.relative_to(ROOT)): h for f, h in inputs.items()},
                         prepared_files={str(f.relative_to(dest)): sha(f) for f in dest.rglob('*.csv')})
            save(dest/'preparation.json', state)
        except BaseException as e:
            state.update(status='failed', error=repr(e), seconds=time.monotonic()-start)
            save(dest/'preparation.json', state); raise
        return
    prep = json.loads((dest/'preparation.json').read_text(encoding='utf-8'))
    assert prep['status'] == 'prepared' and prep['input_sha256'] == {str(f.relative_to(ROOT)): h for f,h in inputs.items()}
    assert all(sha(dest/k) == h for k,h in prep['prepared_files'].items())
    if args.mode == 'audit':
        state = json.loads((dest/'execution.json').read_text(encoding='utf-8'))
        assert state['status'] == 'completed' and state['risk_fits_started'] == state['arms_completed'] == 9
        max_delta = 0.; aggregates = []
        for row in state['results']:
            folder = dest/row['task']/row['arm']
            assert sha(folder/'risk_rf.joblib') == row['model_sha256']
            curves = {}
            for role in ['calibration', 'evaluation']:
                scores = read(folder/(role+'_scores.csv'))
                frame = read(dest/row['task']/(role+'.csv'))
                assert scores.source_row.tolist() == frame.source_row.tolist()
                assert sha(folder/(role+'_scores.csv')) == row['score_sha256'][role]
                curve, counts = independent_curve(frame, scores.risk, fixed['coverage_grid'])
                delta = float(np.max(np.abs(np.array(curve)-row[role]['curve_rmse'])))
                assert delta <= 1e-12; max_delta = max(max_delta, delta)
                curves[role] = dict(curve_rmse=curve, curve_mean_rmse=float(np.mean(curve)), accepted_counts=counts)
            aggregates.append(dict(task=row['task'], arm=row['arm'], columns=row['columns'], **curves))
        for task in p['tasks']:
            for role in ['calibration', 'evaluation']:
                full = [r[role]['curve_rmse'][-1] for r in aggregates if r['task'] == task]
                assert len(full) == 3 and max(full)-min(full) <= 1e-12
        result = dict(passed=True, results=aggregates, preparation=prep['jobs'],
                      execution={k:v for k,v in state.items() if k != 'results'},
                      independent_curve_max_abs=max_delta, source_sha256=prep['input_sha256'],
                      preparation_seconds=prep['seconds'], official_test_rows_read=0)
        save(DOC/'results.json', result)
        print('AUDIT PASSED: 9 arms, both roles, independent tie/ceil/RMSE replay', flush=True)
        return
    assert not (dest/'execution.json').exists(), 'No automatic restart'
    state = dict(status='running', risk_fits_started=0, arms_completed=0, model_loads=0, predict_calls=0, results=[])
    save(dest/'execution.json', state)
    try:
        cq = cq_function(BASE/'author/src/conformal.py')
        for task in p['tasks']:
            frames = {role:read(dest/task/(role+'.csv')) for role in ['oof', 'calibration', 'evaluation']}
            target = np.abs(frames['oof'].prediction-frames['oof'].y)
            for arm in ['joint', 'separate', 'radial']:
                assert time.monotonic()-start < p['budgets']['risk_wall_seconds']
                columns = GENERIC+EXTRA+['eta_mean', arm]
                folder = dest/task/arm; folder.mkdir()
                state['risk_fits_started'] += 1; save(dest/'execution.json', state)
                rf = RandomForestRegressor(**p['risk_RF']).fit(frames['oof'][columns], target)
                scores = {}
                for role in ['calibration', 'evaluation']:
                    state['predict_calls'] += 1; scores[role] = rf.predict(frames[role][columns])
                joblib.dump(rf, folder/'risk_rf.joblib')
                state['model_loads'] += 1; loaded = joblib.load(folder/'risk_rf.joblib')
                replay = 0.
                for role in scores:
                    state['predict_calls'] += 1
                    replay = max(replay, float(np.max(np.abs(scores[role]-loaded.predict(frames[role][columns])))))
                    pd.DataFrame(dict(source_row=frames[role].source_row, risk=scores[role])).to_csv(folder/(role+'_scores.csv'), index=False)
                assert replay <= 1e-10
                m = {role:metrics(frames[role], scores[role], frames['calibration'], scores['calibration'], cq, fixed) for role in scores}
                state['results'].append(dict(task=task, arm=arm, columns=columns, **m,
                    model_sha256=sha(folder/'risk_rf.joblib'), reload_max_abs=replay,
                    score_sha256={role:sha(folder/(role+'_scores.csv')) for role in scores}))
                state['arms_completed'] += 1; save(dest/'execution.json', state)
                print('FITTED', task, arm, state['arms_completed'], '/9', flush=True)
        assert state['risk_fits_started'] == 9 and state['model_loads'] == 9 and state['predict_calls'] == 36
        assert all(sha(f) == h for f,h in inputs.items())
        state.update(status='completed', seconds=time.monotonic()-start); save(dest/'execution.json', state)
    except BaseException as e:
        state.update(status='failed', error=repr(e), seconds=time.monotonic()-start)
        save(dest/'execution.json', state); raise


if __name__ == '__main__':
    main()
