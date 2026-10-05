"""Run pinned author RF/roughness code on development-only fixtures, not a benchmark."""
import argparse
import ast
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from rdkit import Chem, rdBase

ROOT = Path(__file__).resolve().parents[2]
DEPLOYABLE = ['nbr_disp', 'sali_max', 'sali_mean', 'holder', 'nn_sim', 'local_dens', 'mol_size', 'rf_var']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refuse to overwrite existing output')
    provenance = json.loads((ROOT / 'docs/research/roughness_method_review_20261005/provenance.json').read_text(encoding='utf-8'))
    expected = {r['path']: r['sha256'] for r in provenance['sources'] if r['repo'].endswith('/qsar-landscape-roughness')}
    reused = ['LICENSE', 'src/config.py', 'src/build_features.py', 'src/conformal.py']
    for file in reused:
        assert digest(args.source_root / file) == expected[file], file
    manifest = json.loads(args.manifest.read_text())
    csv_hash, manifest_hash = digest(args.csv), digest(args.manifest)
    assert csv_hash == manifest['input_sha256']
    train_rows = sorted(manifest['train_rows'])[:128]
    valid_rows = sorted(manifest['valid_rows'])[:32]
    assert len(train_rows) == 128 and len(valid_rows) == 32
    assert set(train_rows).isdisjoint(valid_rows)
    selected = set(train_rows + valid_rows)
    records = {}
    with args.csv.open(encoding='utf-8-sig', newline='') as handle:
        for number, row in enumerate(csv.DictReader(handle)):
            if number not in selected:
                continue
            assert row['split'] == 'train', 'Official test cannot enter smoke'
            mol = Chem.MolFromSmiles(row['smiles'])
            assert mol is not None
            y = float(row['y'])
            assert np.isfinite(y)
            records[number] = {'smiles': row['smiles'], 'y': y, 'cliff_mol': int(row['cliff_mol']),
                               'canonical': Chem.MolToSmiles(mol)}
    assert set(records) == selected
    assert not ({records[i]['canonical'] for i in train_rows} & {records[i]['canonical'] for i in valid_rows})
    args.output.mkdir(parents=True)
    author = args.output / 'author'
    for file in reused:
        target = author / file
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.source_root / file, target)
        assert digest(target) == expected[file]
    outputs, durations = {}, {}
    started = time.monotonic()
    shuffled = np.random.RandomState(0).permutation([records[i]['y'] for i in train_rows])
    for variant in ['original', 'query_label_shift', 'train_label_permutation']:
        folder = args.output / variant
        data = folder / 'data'
        data.mkdir(parents=True)
        rows = []
        for k, i in enumerate(train_rows):
            rows.append(dict(smiles=records[i]['smiles'], y=shuffled[k] if variant == 'train_label_permutation' else records[i]['y'],
                             cliff_mol=records[i]['cliff_mol'], split='train'))
        for i in valid_rows:
            # Upstream expects "test". These are manifest validation rows, never official test.
            rows.append(dict(smiles=records[i]['smiles'], y=records[i]['y'] + (1000 if variant == 'query_label_shift' else 0),
                             cliff_mol=records[i]['cliff_mol'], split='test'))
        dataset = 'development_interface_fixture'
        pd.DataFrame(rows).to_csv(data / (dataset + '.csv'), index=False)
        env = os.environ.copy()
        env.update({'MOLECULEACE_DATA': str(data.resolve()), 'QSAR_CACHE': str((folder / 'cache').resolve()),
                    'QSAR_DATA': str(data.resolve()), 'QSAR_RESULTS': str((folder / 'results').resolve()),
                    'QSAR_FIGURES': str((folder / 'figures').resolve()), 'PYTHONIOENCODING': 'utf-8'})
        tick = time.monotonic()
        result = subprocess.run([sys.executable, str((author / 'src/build_features.py').resolve()), dataset],
                                env=env, capture_output=True, text=True, encoding='utf-8', timeout=120)
        (folder / 'stdout.txt').write_text(result.stdout, encoding='utf-8')
        (folder / 'stderr.txt').write_text(result.stderr, encoding='utf-8')
        assert result.returncode == 0, result.stderr
        durations[variant] = time.monotonic() - tick
        frame = pd.read_csv(folder / 'cache' / (dataset + '.csv'))
        assert len(frame) == 32 and frame['smiles'].tolist() == [records[i]['smiles'] for i in valid_rows]
        assert np.isfinite(frame[DEPLOYABLE].to_numpy(float)).all()
        outputs[variant] = frame
        print(variant, '32 rows verified', flush=True)
    a, b, c = (outputs[v] for v in ['original', 'query_label_shift', 'train_label_permutation'])
    np.testing.assert_allclose(a[DEPLOYABLE], b[DEPLOYABLE], rtol=0, atol=1e-10)
    assert not np.allclose(a['dirichlet'], b['dirichlet'])
    assert not np.allclose(a['lipschitz'], b['lipschitz'])
    assert not np.allclose(a['sali_mean'], c['sali_mean'])
    # Reuse only the author's two inspected pure functions; skip file-reading top-level code.
    tree = ast.parse((author / 'src/conformal.py').read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'cq', 'qbins'}]
    assert len(functions) == 2
    namespace = {'np': np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'pinned_author_conformal_functions', 'exec'), namespace)
    resid, rough = a['rf_err'].to_numpy(), a['sali_mean'].to_numpy()
    cal, ev = np.arange(16), np.arange(16, 32)
    bins = namespace['qbins'](rough[cal], rough[cal], 2)
    ev_bins = namespace['qbins'](rough[cal], rough[ev], 2)
    widths = np.array([namespace['cq'](resid[cal][bins == k], 0.1) for k in ev_bins])
    assert np.isfinite(widths).all() and (widths >= 0).all()
    assert digest(args.csv) == csv_hash and digest(args.manifest) == manifest_hash
    result = {'status': 'interface_checks_passed_not_performance_evidence', 'dataset': manifest['dataset'],
              'csv_sha256': csv_hash, 'manifest_sha256': manifest_hash, 'train_source_rows': train_rows,
              'validation_source_rows': valid_rows, 'official_test_rows_used': 0,
              'source_sha256': {k: expected[k] for k in reused},
              'versions': {'python': sys.version.split()[0], 'sklearn': sklearn.__version__, 'rdkit': rdBase.rdkitVersion},
              'checks': {'author_files_unchanged': True, 'query_label_shift_invariance': True,
                         'query_dependent_columns_change': True, 'training_label_dependence': True,
                         'canonical_train_valid_overlap': 0, 'finite_deployable_features': True,
                         'pure_author_calibration_interface': True, 'input_hashes_unchanged': True},
              'counts': {'author_RF_smoke_fits': 3, 'formal_training': 0, 'test_evaluation': 0,
                         'calibration_validation_rows': 16, 'evaluation_validation_rows': 16},
              'runtime_seconds': durations, 'total_seconds': time.monotonic() - started,
              'limitations': ['RF is an engineering reference, not the main algorithm.', 'No GNN/UNIQUE runtime verification.',
                             '2-bin, 16/16 calibration is smoke only; no coverage or improvement claim.',
                             'Label permutations test dependency only, not an empirical null performance test.']}
    (args.output / 'smoke.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('All interface checks passed; official test unused.', flush=True)


if __name__ == '__main__':
    main()
