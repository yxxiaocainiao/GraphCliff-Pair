"""M65: lawful input preparation and bounded five-arm wiring. CLI never trains."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

from prepare_conditional_pilot import ROOT, PROTOCOL, jobs_for
from conditional_risk_weights import FEATURES, joint_and_conditional_weights, normalized_weights
from run_phase_a import run_command, save, sha

SOURCE = Path('D:/GraphCliff-main/benchmark_data/CHEMBL234_Ki.csv')
SOURCE_SHA = '447d38f659b86c7177f022b16f028db6828ab09c80051e74dad420168970f657'
AUTHOR = ROOT/'artifacts/reliability_phase_a_20261005_v2/author'


def read_selected(source, ids, columns):
    """Skip forbidden records before parsing; retain original CSV row identities."""
    ids = sorted(ids)
    if not ids or len(ids) != len(set(ids)) or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError('Expected distinct nonnegative source row IDs')
    selected = set(ids)
    frame = pd.read_csv(source, usecols=columns,
                        skiprows=lambda line: line > 0 and line-1 not in selected)
    if len(frame) != len(ids):
        raise ValueError('Missing source rows')
    frame.index = pd.Index(ids, name='source_row')
    return frame[columns]


def prepare(output):
    output = output.resolve()
    if output == ROOT/'artifacts' or not output.is_relative_to(ROOT/'artifacts'):
        raise ValueError('Output must be a new artifacts child')
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status='preparing', fits=0, predictions=0, evaluation_labels_read=0)
    save(output/'state.json', state)
    try:
        p = json.loads(PROTOCOL.read_text(encoding='utf-8'))
        for name, expected in p['input_sha256'].items():
            if sha(ROOT/name) != expected:
                raise ValueError('Pinned input changed: '+name)
        if sha(SOURCE) != SOURCE_SHA:
            raise ValueError('Source CSV hash changed')
        rows = json.loads((ROOT/p['private_roles_manifest']).read_text(encoding='utf-8'))
        jobs = jobs_for(rows, p, output)
        allowed = [r['source_row'] for r in rows if r['role'] != 'calibration']
        label_ids = [r['source_row'] for r in rows if r['role'] in ('fit', 'monitor')]
        frame = read_selected(SOURCE, allowed, ['smiles', 'split'])
        labels = read_selected(SOURCE, label_ids, ['y'])
        if not frame.split.eq('train').all() or not np.isfinite(labels.y.to_numpy(dtype=float)).all():
            raise ValueError('Invalid development rows or legal training labels')
        mols = frame.smiles.map(Chem.MolFromSmiles)
        if mols.isna().any():
            raise ValueError('Invalid allowed SMILES')
        frame['canonical'] = mols.map(Chem.MolToSmiles)
        for job in jobs:
            sets = [{frame.loc[i, 'canonical'] for i in job[role]} for role in ('fit', 'monitor', 'query')]
            if any(sets[a] & sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
                raise ValueError('Canonical identity crosses job roles')
            folder = output/p['task']/job['name']
            (folder/'features/data').mkdir(parents=True)
            train = pd.concat([frame.loc[job['fit'], ['smiles']].join(labels).assign(partition='train'),
                               frame.loc[job['monitor'], ['smiles']].join(labels).assign(partition='val')])
            train.to_csv(folder/'train_val.csv', index=False)
            query = frame.loc[job['query'], ['smiles']]
            query.to_csv(folder/'query.csv', index=False)
            reference = frame.loc[job['fit'], ['smiles']].join(labels).assign(cliff_mol=0, split='train')
            dummy = query.assign(y=0., cliff_mol=0, split='test')
            pd.concat([reference, dummy]).to_csv(folder/'features/data/fixture.csv', index=False)
        frame[['smiles', 'canonical']].to_csv(output/'identities.csv')
        labels.loc[jobs[-1]['fit']].to_csv(output/'source_labels.csv')
        provenance = json.loads((ROOT/'docs/research/roughness_method_review_20261005/provenance.json').read_text(encoding='utf-8'))
        expected_author = {s['path']: s['sha256'] for s in provenance['sources']
                           if s['repo'] == 'krishnatheaverage/qsar-landscape-roughness'}
        for name in ('LICENSE', 'src/config.py', 'src/build_features.py'):
            if sha(AUTHOR/name) != expected_author[name]:
                raise ValueError('Author source changed: '+name)
            target = output/'author'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(AUTHOR/name, target)
        save(output/'jobs.json', jobs)
        save(output/'protocol.json', p)
        files = {str(f.relative_to(output)): sha(f) for f in output.rglob('*')
                 if f.is_file() and f.name != 'state.json'}
        save(output/'inputs.json', dict(files=files, source_sha256=SOURCE_SHA,
             protocol_sha256=sha(PROTOCOL), runner_sha256=sha(Path(__file__)),
             structure_rows=len(frame), legal_label_rows=len(labels),
             true_evaluation_labels=0, fits=0, predictions=0))
        state.update(status='prepared_inputs')
        save(output/'state.json', state)
        return check(output)
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        save(output/'state.json', state)
        raise


def check(output):
    m = json.loads((output/'inputs.json').read_text(encoding='utf-8'))
    if sha(Path(__file__)) != m['runner_sha256'] or sha(PROTOCOL) != m['protocol_sha256']:
        raise ValueError('Prepared code/protocol changed')
    for name, expected in m['files'].items():
        if sha(output/name) != expected:
            raise ValueError('Prepared file changed: '+name)
    p = json.loads((output/'protocol.json').read_text(encoding='utf-8'))
    jobs = json.loads((output/'jobs.json').read_text(encoding='utf-8'))
    identities = pd.read_csv(output/'identities.csv', index_col='source_row')
    source_y = pd.read_csv(output/'source_labels.csv', index_col='source_row')
    if list(source_y.columns) != ['y'] or source_y.index.tolist() != jobs[-1]['fit']:
        raise ValueError('Risk label identities changed')
    for j in jobs:
        folder = output/p['task']/j['name']
        train = pd.read_csv(folder/'train_val.csv')
        query = pd.read_csv(folder/'query.csv')
        fixture = pd.read_csv(folder/'features/data/fixture.csv')
        assert list(train.columns) == ['smiles', 'y', 'partition']
        assert train.partition.tolist() == ['train']*len(j['fit'])+['val']*len(j['monitor'])
        assert train.smiles.tolist() == identities.loc[j['fit']+j['monitor'], 'smiles'].tolist()
        assert np.isfinite(train.y).all()
        assert list(query.columns) == ['smiles']
        assert query.smiles.tolist() == identities.loc[j['query'], 'smiles'].tolist()
        assert fixture.smiles.tolist() == identities.loc[j['fit']+j['query'], 'smiles'].tolist()
        assert fixture.split.tolist() == ['train']*len(j['fit'])+['test']*len(j['query'])
        assert fixture.iloc[len(j['fit']):].y.eq(0).all() and fixture.cliff_mol.eq(0).all()
        assert np.array_equal(fixture.iloc[:len(j['fit'])].y, train.iloc[:len(j['fit'])].y)
    return dict(status='checked_inputs', jobs=4, csv_inputs=12, structure_rows=m['structure_rows'],
                legal_label_rows=m['legal_label_rows'], evaluation_labels_read=0, fits=0, predictions=0)


def five_arms(source, target, absolute_error, source_bits, target_bits, protocol):
    """Future execution only: identical RF7 inputs/targets, five sample-weight rules."""
    if list(source.columns) != FEATURES or list(target.columns) != FEATURES:
        raise ValueError('Expected seven evidence columns in frozen order')
    if not source.index.is_unique or not target.index.is_unique or not source.index.intersection(target.index).empty:
        raise ValueError('Invalid source/target identities')
    if not absolute_error.index.equals(source.index):
        raise ValueError('OOF supervision must align by source identity')
    if (not np.isfinite(source.to_numpy(dtype=float)).all() or not np.isfinite(target.to_numpy(dtype=float)).all()
            or not np.isfinite(absolute_error).all() or (absolute_error < 0).any()):
        raise ValueError('Invalid evidence or absolute OOF errors')
    for bits, frame in ((source_bits, source), (target_bits, target)):
        if not bits.index.equals(frame.index) or bits.shape[1] != 2048 or not np.isin(bits.to_numpy(), [0, 1]).all():
            raise ValueError('Expected identity-aligned Morgan2048 bits')
    scaler = StandardScaler()
    s = pd.DataFrame(scaler.fit_transform(source), columns=FEATURES, index=source.index)
    t = pd.DataFrame(scaler.transform(target), columns=FEATURES, index=target.index)
    shared = joint_and_conditional_weights(s, t)
    pca = PCA(n_components=7, svd_solver='full', whiten=False)
    ps = pca.fit_transform(source_bits)
    if not np.isfinite(pca.singular_values_).all() or pca.singular_values_[-1] <= np.finfo(float).eps * max(source_bits.shape) * pca.singular_values_[0]:
        raise ValueError('Source structure rank below seven')
    pt = pca.transform(target_bits)
    structure_scaler = StandardScaler()
    ps, pt = structure_scaler.fit_transform(ps), structure_scaler.transform(pt)
    def ratio(a, b):
        ks = KernelDensity(bandwidth='scott', kernel='gaussian', metric='euclidean', atol=0, rtol=0).fit(a)
        kt = KernelDensity(bandwidth='scott', kernel='gaussian', metric='euclidean', atol=0, rtol=0).fit(b)
        return normalized_weights(kt.score_samples(a)-ks.score_samples(a))
    weights = {'RF7': np.ones(len(source)), 'IW-GEN5': ratio(s.iloc[:, :5], t.iloc[:, :5]),
               'IW-STRUCT7': ratio(ps, pt), 'IW-RISK7': shared['joint'], 'IW-COND7': shared['conditional']}
    scores = {}
    for arm in protocol['arms']:
        model = RandomForestRegressor(**protocol['risk_rf'])
        model.fit(source, absolute_error, sample_weight=weights[arm])
        scores[arm] = model.predict(target)
    scores = pd.DataFrame(scores, index=target.index)
    if not np.isfinite(scores).all().all():
        raise ValueError('Nonfinite risk scores')
    return scores, pd.DataFrame(weights, index=source.index)


def postprocess(output):
    """Child process, bounded by the parent's subprocess timeout; never run in preparation."""
    state = json.loads((output/'execution.json').read_text(encoding='utf-8'))
    if state.get('status') != 'running' or not state.get('explicit_training_authorization'):
        raise ValueError('No authorized execution in progress')
    p = json.loads((output/'protocol.json').read_text(encoding='utf-8'))
    jobs = json.loads((output/'jobs.json').read_text(encoding='utf-8'))
    parts = []
    for j in jobs:
        folder = output/p['task']/j['name']
        pred = pd.read_csv(folder/'predictions.csv', usecols=['smiles', 'y'])
        feature = pd.read_csv(folder/'features/cache/fixture.csv', usecols=['smiles']+FEATURES[1:])
        expected = pd.read_csv(folder/'query.csv').smiles.tolist()
        if pred.smiles.tolist() != expected or feature.smiles.tolist() != expected:
            raise ValueError('Prediction/feature query ordering changed')
        feature = feature[FEATURES[1:]].copy()
        feature.insert(0, 'prediction', pred.y.to_numpy())
        feature.index = pd.Index(j['query'], name='source_row')
        parts.append(feature)
    source, target = pd.concat(parts[:3]).sort_index(), parts[-1]
    labels = pd.read_csv(output/'source_labels.csv', index_col='source_row').y
    identities = pd.read_csv(output/'identities.csv', index_col='source_row')
    # Execute only the existing pure featurize definition, avoiding author's import side effects.
    import ast
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    tree = ast.parse((output/'author/src/build_features.py').read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'featurize')
    env = dict(np=np, Chem=Chem, DataStructs=DataStructs,
               _gen=rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048))
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<author featurize>', 'exec'), env)
    bits = []
    for frame in (source, target):
        _, arrays, _, ok = env['featurize'](identities.loc[frame.index, 'smiles'])
        if not ok.all():
            raise ValueError('Invalid structure during projection')
        bits.append(pd.DataFrame(arrays, index=frame.index))
    scores, weights = five_arms(source, target, (source.prediction-labels.loc[source.index]).abs(), *bits, p)
    scores.to_csv(output/'scores.csv')
    weights.to_csv(output/'weights.csv')
    # All scores and exact prepared configuration locked before reading any target truth.
    save(output/'score_lock.json', dict(scores_sha256=sha(output/'scores.csv'),
         inputs_sha256=sha(output/'inputs.json'), runner_sha256=sha(Path(__file__)),
         protocol_sha256=sha(output/'protocol.json'), arms=list(scores.columns)))
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError('Source changed before evaluation')
    truth = read_selected(SOURCE, target.index.tolist(), ['y']).y
    if not np.isfinite(truth).all():
        raise ValueError('Invalid evaluation labels')
    results = {}
    for arm in p['arms']:
        order = sorted(range(len(target)), key=lambda i: (scores[arm].iloc[i],
                       identities.loc[target.index[i], 'canonical'], int(target.index[i])))
        error = (target.prediction-truth).to_numpy()
        curve = [float(np.sqrt(np.mean(error[order[:math.ceil(f*len(target))]]**2))) for f in p['coverage_grid']]
        results[arm] = dict(curve_rmse=curve, mean_of_six_rmse=float(np.mean(curve)))
    assert np.allclose([r['curve_rmse'][-1] for r in results.values()], results['RF7']['curve_rmse'][-1], rtol=0, atol=1e-12)
    save(output/'results.json', results)


def execute(output, *, explicit_training_authorization=False):
    """Not exposed by CLI. A future explicitly authorized call must opt in."""
    if not explicit_training_authorization:
        raise ValueError('Real training has not been authorized')
    output = output.resolve()
    check(output)
    if (output/'execution.json').exists():
        raise ValueError('Refuse retry or overwrite of an execution')
    p = json.loads((output/'protocol.json').read_text(encoding='utf-8'))
    jobs = json.loads((output/'jobs.json').read_text(encoding='utf-8'))
    started = time.monotonic()
    state = dict(status='running', explicit_training_authorization=True,
                 chemprop_fits_started=0, point_predict_batches_started=0,
                 auxiliary_rf_batches_started=0, postprocess_started=False)
    def command(cmd, folder, cap):
        remaining = p['budget']['whole_run_seconds']-(time.monotonic()-started)
        if remaining <= 0:
            raise TimeoutError('Whole-run budget exhausted')
        save(output/'execution.json', state)
        run_command(cmd, folder, min(cap, remaining))
    try:
        for j in jobs:
            folder = output/p['task']/j['name']
            state['chemprop_fits_started'] += 1
            command(j['train_command'], folder/'train_log', p['budget']['fit_seconds_each'])
            if not (folder/'training/model_0/best.pt').is_file():
                raise ValueError('Missing point checkpoint')
            state['point_predict_batches_started'] += 1
            command(j['predict_command'], folder/'predict_log', p['budget']['predict_seconds_each'])
            feature = folder/'features'
            env = dict(os.environ, MOLECULEACE_DATA=str(feature/'data'), QSAR_DATA=str(feature/'data'),
                       QSAR_CACHE=str(feature/'cache'), QSAR_RESULTS=str(feature/'results'), QSAR_FIGURES=str(feature/'figures'))
            state['auxiliary_rf_batches_started'] += 1
            remaining = p['budget']['whole_run_seconds']-(time.monotonic()-started)
            if remaining <= 0:
                raise TimeoutError('Whole-run budget exhausted')
            save(output/'execution.json', state)
            run_command([sys.executable, str(output/'author/src/build_features.py'), 'fixture'],
                        folder/'feature_log', min(p['budget']['feature_seconds_each'], remaining), env)
        state['postprocess_started'] = True
        # A fresh child bounds PCA/KDE/RF and evaluation together, including native-library work.
        command([sys.executable, '-c',
                 'import sys; sys.path.insert(0, sys.argv[2]); from pathlib import Path; from run_conditional_pilot import postprocess; postprocess(Path(sys.argv[1]))',
                 str(output), str(Path(__file__).parent)], output/'postprocess_log', p['budget']['postprocess_seconds'])
        state['status'] = 'completed'
    except BaseException as error:
        state.update(status='failed_or_interrupted', error=repr(error))
        raise
    finally:
        state['elapsed_seconds'] = time.monotonic()-started
        save(output/'execution.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'check'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output) if args.mode == 'prepare' else check(args.output)))
