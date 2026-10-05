"""Validation-only reuse audit; no fitting or test prediction reads."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

DATASET = 'CHEMBL234_Ki'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def protocol(path):
    spec = importlib.util.spec_from_file_location('reuse_protocol', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.split_train_valid


def prepare(args):
    out, old, project = args.output, args.baselines, args.project
    out.mkdir(parents=True, exist_ok=True)
    source = args.data / f'{DATASET}.csv'
    meta = pd.read_csv(source, usecols=['smiles', 'split'])
    dev_rows = np.flatnonzero(meta['split'].eq('train'))
    dev_set = set(dev_rows)
    dev = pd.read_csv(source, usecols=['smiles', 'y', 'split', 'cliff_mol'],
                      skiprows=lambda i: i > 0 and i-1 not in dev_set)
    dev.index = dev_rows
    # Restore positional rows without loading any official-test labels.
    frame = meta.copy()
    frame['cliff_mol'] = np.nan
    frame.loc[dev_rows, 'cliff_mol'] = dev.cliff_mol
    splitter = protocol(project / 'graphcliff_pair/vendor/protocol.py')
    train, valid = splitter(frame, .1, 42)
    old_splitter = protocol(old / 'utils/splits.py')
    ot, ov = old_splitter(frame, .1, 42)
    assert np.array_equal(train, ot) and np.array_equal(valid, ov)
    processed = old / f'Data/processed/{DATASET}/rows.csv'
    partitions = pd.read_csv(processed, usecols=['source_row', 'partition'])
    selected_positions = set(np.flatnonzero(partitions.partition.isin(['train', 'val'])))
    old_dev = pd.read_csv(processed, usecols=['source_row', 'smiles', 'y', 'cliff_mol', 'partition'],
                          skiprows=lambda i: i > 0 and i-1 not in selected_positions).set_index('source_row')
    assert set(old_dev.index) == set(dev_rows)
    old_dev = old_dev.loc[dev_rows]
    assert np.array_equal(old_dev.smiles, dev.smiles)
    assert np.array_equal(old_dev.cliff_mol, dev.cliff_mol)
    assert np.allclose(old_dev.y, dev.y, atol=1e-12, rtol=0)
    assert np.array_equal(old_dev.index[old_dev.partition.eq('train')], train)
    assert np.array_equal(old_dev.index[old_dev.partition.eq('val')], valid)
    for model in ['gcn', 'gat', 'mlp']:
        for seed in [42, 43, 44]:
            cfg_path = old / f'results/{model}/{DATASET}/seed_{seed}/config.json'
            cfg = json.loads(cfg_path.read_text())
            assert np.array_equal(cfg['train_rows'], train)
            assert np.array_equal(cfg['validation_rows'], valid)
    svm_cfg = old / f'results/svm/{DATASET}/config.json'
    assert np.array_equal(json.loads(svm_cfg.read_text())['train_rows'], train)
    hashes = {str(p): sha(p) for p in [source, processed, svm_cfg,
               old/'utils/splits.py', project/'graphcliff_pair/vendor/protocol.py']}
    for partition, rows in [('train', train), ('val', valid)]:
        cp_path = old / f'results/chemprop/{DATASET}/training/{partition}.csv'
        cp_frame = pd.read_csv(cp_path)
        assert np.array_equal(cp_frame.smiles, dev.loc[rows].smiles)
        assert np.allclose(cp_frame.y, dev.loc[rows].y, atol=1e-12, rtol=0)
        hashes[str(cp_path)] = sha(cp_path)
    validation = dev.loc[valid, ['smiles', 'y', 'cliff_mol']].copy()
    validation.insert(0, 'source_row', valid)
    if (out/'validation_rows.csv').exists():
        pd.testing.assert_frame_equal(pd.read_csv(out/'validation_rows.csv'),
                                      validation.reset_index(drop=True), check_exact=False,
                                      atol=1e-12, rtol=0)
        assert np.array_equal(pd.read_csv(out/'chemprop_validation_input.csv').smiles, validation.smiles)
    else:
        validation.to_csv(out/'validation_rows.csv', index=False)
        validation[['smiles']].to_csv(out/'chemprop_validation_input.csv', index=False)
    dump(out/'identity_audit.json', dict(dataset=DATASET, passed=True,
          train_rows=len(train), validation_rows=len(valid), validation_cliff=int(validation.cliff_mol.sum()),
          split_seed=42, valid_fraction=.1, official_test_labels_read=False,
          old_test_predictions_read=False, source_sha256=hashes,
          checks=['both split functions', 'processed train/val row identity',
                  'SMILES/y/cliff equality', 'SVM fit row identity', 'Chemprop saved train/val identity']))
    print('Identity audit passed:', len(train), 'train;', len(valid), 'val')


def svm_predict(args):
    old, out = args.baselines, args.output
    os.environ['MOLECULEACE_DATA_DIR'] = str(old/'Data')
    sys.path.insert(0, str(old/'third_party/MoleculeACE'))
    from MoleculeACE.benchmark.featurization import Featurizer
    import importlib.metadata
    rows = pd.read_csv(out/'validation_rows.csv')
    weight = old / f'checkpoints/svm/{DATASET}/model.pkl'
    with weight.open('rb') as handle:
        model = pickle.load(handle)
    x = Featurizer.ecfp(rows.smiles.tolist(), radius=2, nbits=1024)
    rows['y_pred'] = model.predict(x)
    assert np.isfinite(rows.y_pred).all()
    rows.to_csv(out/'svm_validation_predictions.csv', index=False)
    dump(out/'svm_reuse.json', dict(weight=str(weight), sha256=sha(weight), trained=False,
         versions={p: importlib.metadata.version(p) for p in ['rdkit','scikit-learn','numpy','pandas']},
         featurizer_sha256=sha(old/'third_party/MoleculeACE/MoleculeACE/benchmark/featurization.py')))
    print('SVM validation prediction:', len(rows))


def analyze(args):
    old, project, out = args.baselines, args.project, args.output
    truth = pd.read_csv(out/'validation_rows.csv').set_index('source_row')
    files = [('svm', None, out/'svm_validation_predictions.csv')]
    cp = pd.read_csv(out/'chemprop_validation_raw.csv')
    assert np.array_equal(cp.smiles, truth.smiles)
    assert 'y' in cp.columns
    cp_rows = truth.reset_index().copy()
    cp_rows['y_pred'] = cp.y.to_numpy()
    cp_rows.to_csv(out/'chemprop_validation_predictions.csv', index=False)
    files.append(('chemprop', 42, out/'chemprop_validation_predictions.csv'))
    cp_manifest = old / f'results/chemprop/{DATASET}/manifest.json'
    cp_binding = json.loads(cp_manifest.read_text())
    assert sha(old/cp_binding['checkpoint']) == cp_binding['checkpoint_sha256']
    binding_hashes = {str(cp_manifest): sha(cp_manifest),
                      str(old/cp_binding['checkpoint']): cp_binding['checkpoint_sha256']}
    svm_hash_file = old/'results/artifact_hashes.csv'
    svm_weights = old/f'checkpoints/svm/{DATASET}/model.pkl'
    svm_hashes = pd.read_csv(svm_hash_file)
    expected = svm_hashes.loc[svm_hashes.path.eq(f'checkpoints/svm/{DATASET}/model.pkl'), 'sha256']
    assert len(expected) == 1 and sha(svm_weights) == expected.iloc[0]
    binding_hashes[str(svm_weights)] = sha(svm_weights)
    binding_hashes[str(svm_hash_file)] = sha(svm_hash_file)
    source_mismatches = {}
    for model in ['gcn', 'gat', 'mlp']:
        for seed in [42,43,44]:
            files.append((model, seed, old/f'results/{model}/{DATASET}/seed_{seed}/validation_predictions.csv'))
            manifest_path = old/f'results/{model}/{DATASET}/seed_{seed}/manifest.json'
            manifest = json.loads(manifest_path.read_text())
            for rel, expected in manifest['identity']['files'].items():
                path = old/rel
                actual = sha(path)
                if actual != expected:
                    source_mismatches[str(path)] = dict(expected=expected, actual=actual)
                binding_hashes[str(path)] = actual
            for rel, expected in manifest['artifacts'].items():
                path = old/rel
                assert sha(path) == expected, f'Historical binding mismatch: {path}'
                binding_hashes[str(path)] = expected
            binding_hashes[str(manifest_path)] = sha(manifest_path)
    for seed in [42,43,44]:
        files.append(('graphcliff',seed,project/f'artifacts/aca_pilot_20261005/{DATASET}/seed{seed}/mse/validation_predictions.csv'))
    records, predictions, hashes = [], {}, {}
    for model, seed, path in files:
        df = pd.read_csv(path)
        if model == 'graphcliff':
            df = df.rename(columns={'query':'source_row','prediction':'y_pred'})
        assert df.source_row.is_unique and set(df.source_row) == set(truth.index)
        df = df.set_index('source_row').loc[truth.index]
        assert np.allclose(df.y, truth.y, atol=3e-7, rtol=0)
        assert np.array_equal(df.cliff_mol, truth.cliff_mol)
        if 'smiles' in df.columns:
            assert np.array_equal(df.smiles, truth.smiles)
        pred = df.y_pred.to_numpy(dtype=float)
        assert np.isfinite(pred).all()
        key = f'{model}_{seed}' if seed is not None else model
        predictions[key] = pred
        hashes[str(path)] = sha(path)
        error = pred - truth.y.to_numpy()
        rec = dict(model=model, seed=seed, rows=len(df))
        for group, mask in [('overall',np.ones(len(df),dtype=bool)),('cliff',truth.cliff_mol.eq(1).to_numpy()),
                            ('noncliff',truth.cliff_mol.eq(0).to_numpy())]:
            rec[group+'_rmse'] = float(np.sqrt(np.mean(error[mask]**2)))
        records.append(rec)
    metrics = pd.DataFrame(records)
    metrics.to_csv(out/'validation_metrics.csv', index=False)
    summary = metrics.groupby('model').agg(runs=('overall_rmse','size'),
        overall_mean=('overall_rmse','mean'), overall_sd=('overall_rmse','std'),
        cliff_mean=('cliff_rmse','mean'),cliff_sd=('cliff_rmse','std'),
        noncliff_mean=('noncliff_rmse','mean')).reset_index()
    summary.to_csv(out/'model_summary.csv', index=False)
    wide = truth.copy()
    for key,pred in predictions.items():
        wide[key] = pred
    wide.to_csv(out/'aligned_validation_predictions.csv')
    # Every comparison uses the common training seed 42; no fitted ensemble or oracle.
    target = truth.y.to_numpy()
    gc_error = predictions['graphcliff_42'] - target
    cliff = truth.cliff_mol.eq(1).to_numpy()
    comparisons = []
    for model in ['svm','chemprop','gcn','gat','mlp']:
        key = model if model == 'svm' else model+'_42'
        other_error = predictions[key] - target
        for group, mask in [('overall',np.ones(len(truth),dtype=bool)),('cliff',cliff)]:
            idx = np.flatnonzero(mask)
            k = max(1, int(np.ceil(len(idx)*.2)))
            gc_top = set(idx[np.argsort(np.abs(gc_error[idx]))[-k:]])
            other_top = set(idx[np.argsort(np.abs(other_error[idx]))[-k:]])
            delta = gc_error[idx]**2-other_error[idx]**2
            trimmed = np.delete(delta, np.argmax(np.abs(delta)))
            comparisons.append(dict(model=model, subset=group,n=len(idx),
               graphcliff_minus_baseline_mse=float(delta.mean()),
               removed_largest_absolute_difference_mse=float(trimmed.mean()),
               baseline_lower_absolute_error_count=int(np.sum(np.abs(other_error[idx])<np.abs(gc_error[idx]))),
               top20percent_count=k, top_error_intersection=len(gc_top & other_top),
               top_error_jaccard=len(gc_top & other_top)/len(gc_top | other_top),
               signed_error_correlation=float(np.corrcoef(gc_error[idx],other_error[idx])[0,1])))
    pd.DataFrame(comparisons).to_csv(out/'error_overlap.csv',index=False)
    gc_rows = pd.read_csv(project/f'artifacts/aca_pilot_20261005/{DATASET}/seed42/mse/validation_predictions.csv').set_index('query').loc[truth.index]
    cases = truth.copy()
    cases['graphcliff_prediction'] = predictions['graphcliff_42']
    cases['svm_prediction'] = predictions['svm']
    cases['graphcliff_absolute_error'] = np.abs(gc_error)
    cases['svm_absolute_error'] = np.abs(predictions['svm']-target)
    cases['graphcliff_minus_svm_squared_error'] = gc_error**2-(predictions['svm']-target)**2
    cases['nearest_train_similarity'] = gc_rows.similarity.to_numpy()
    cases['nearest_train_row'] = gc_rows.reference.to_numpy()
    cases['shared_error_score'] = np.minimum(cases.graphcliff_absolute_error,cases.svm_absolute_error)
    cases[cliff].sort_values('graphcliff_minus_svm_squared_error',ascending=False).to_csv(out/'cliff_case_ranking.csv')
    cases[cliff].sort_values('shared_error_score',ascending=False).to_csv(out/'shared_hard_case_ranking.csv')
    dump(out/'prediction_audit.json',dict(passed=True, prediction_sets=len(files),
         official_test_predictions_read=False, fitting_performed=False, source_sha256=hashes,
         label_absolute_tolerance=3e-7, note='GraphCliff saved labels are float32; metrics recomputed against source float64 labels.'))
    dump(out/'historical_binding_audit.json',dict(passed=True,
         historical_neural_runs=9, chemprop_checkpoint_matches_historical_manifest=True,
         artifact_hashes_match=True, current_training_sources_match=not source_mismatches,
         source_mismatches=source_mismatches, source_sha256=binding_hashes,
         scope='Saved artifacts can be compared; changed historical training source prevents claiming exact current-code reproduction.'))
    print(summary.to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare','svm','analyze'])
    parser.add_argument('--baselines',type=Path,required=True)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    {'prepare':prepare,'svm':svm_predict,'analyze':analyze}[args.mode](args)
