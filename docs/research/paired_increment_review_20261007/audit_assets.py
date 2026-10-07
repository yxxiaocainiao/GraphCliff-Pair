"""Read identities and hash bytes only; never deserialize or run a model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
TASKS = ['CHEMBL234_Ki', 'CHEMBL244_Ki', 'CHEMBL4792_Ki']
FEATURES = ['prediction', 'nn_sim', 'local_dens', 'mol_size', 'rf_var',
            'nbr_disp', 'sali_mean']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(path, expected):
    actual = digest(path)
    assert actual == expected, str(path)
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': actual,
            'bytes': path.stat().st_size, 'matches_binding': True}


def identities(path):
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        assert set(FEATURES).issubset(reader.fieldnames)
        rows = [(int(r['source_row']), r['canonical'], r['smiles']) for r in reader]
    assert len(rows) == len({r[0] for r in rows})
    assert len(rows) == len({r[1] for r in rows})
    return rows


def selfcheck():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'identity.csv'
        fields = ['source_row', 'canonical', 'smiles'] + FEATURES
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerow([1, 'C', 'C'] + ['DO_NOT_PARSE_NUMERIC'] * len(FEATURES))
        assert identities(path) == [(1, 'C', 'C')]
        before = digest(path)
        with path.open('a', newline='', encoding='utf-8') as stream:
            csv.writer(stream).writerow([1, 'C', 'C'] + ['DO_NOT_PARSE_NUMERIC'] * len(FEATURES))
        assert digest(path) != before
        try:
            identities(path)
        except AssertionError:
            pass
        else:
            raise AssertionError('Duplicate identity was not rejected')
    print('PASS: numeric values untouched, duplicate identity rejected, changed bytes detected')


def run(output):
    assert not output.exists(), 'Preserve prior audit; refuse overwrite'
    start = time.monotonic()
    base = ROOT / 'artifacts/reliability_phase_a_20261005_v3'
    permutation = ROOT / 'artifacts/roughness_permutation_20261006_v2'
    load = lambda path: json.loads(path.read_text(encoding='utf-8'))
    binding = {k.replace('\\', '/'): v for k, v in load(base / 'artifact_binding.json').items()}
    prepared = load(base / 'prepare.json')
    preparation = {k.replace('\\', '/'): v for k, v in prepared['files'].items()}
    historical = load(ROOT / 'experiments/reliability_base/permutation_results.json')
    fixed = load(ROOT / 'experiments/reliability_base/fixed_protocol.json')
    checks, summaries, risks = [], [], []
    for task in TASKS:
        manifest_path = ROOT / 'artifacts/reliability_protocol_20261005_v2' / (task + '.json')
        summary = next(s for s in fixed['tasks'] if s['dataset'] == task)
        checks.append(checked(manifest_path, summary['partition_manifest_sha256']))
        manifest = load(manifest_path)
        roles = {}
        for role in ['oof', 'calibration', 'evaluation']:
            path = base / task / (role + '.csv')
            checks.append(checked(path, binding[path.relative_to(base).as_posix()]))
            rows = identities(path)
            expected = manifest['roles']['fit' if role == 'oof' else role]
            assert {r[0] for r in rows} == set(expected)
            roles[role] = rows
        sets = [set(r[1] for r in rows) for rows in roles.values()]
        assert not any(sets[i] & sets[j] for i in range(3) for j in range(i))
        identity = {r[0]: r for rows in roles.values() for r in rows}
        jobs = []
        for name in ['fold0', 'fold1', 'fold2', 'full']:
            job = next(j for j in prepared['jobs'] if j['task'] == task and j['name'] == name)
            query = set(manifest['roles']['calibration'] + manifest['roles']['evaluation']) if name == 'full' else set(manifest['oof_folds'][name[-1]])
            reference = set(manifest['roles']['fit']) if name == 'full' else set(manifest['roles']['fit']) - query
            assert set(job['fit']) == reference and set(job['query']) == query
            joined = base / task / name / 'joined.csv'
            fixture = base / task / name / 'features/data/fixture.csv'
            checks.append(checked(joined, binding[joined.relative_to(base).as_posix()]))
            checks.append(checked(fixture, preparation[fixture.relative_to(base).as_posix()]))
            assert set(identities(joined)) == {identity[i] for i in query}
            with fixture.open(encoding='utf-8', newline='') as stream:
                # The local 'test' token denotes development queries, not official test.
                records = [(r['smiles'], r['split']) for r in csv.DictReader(stream)]
            assert records == [(identity[i][2], 'train') for i in job['fit']] + [(identity[i][2], 'test') for i in job['query']]
            full_queries = set(manifest['roles']['calibration'] + manifest['roles']['evaluation'])
            assert not ({identity[i][1] for i in reference} & {identity[i][1] for i in full_queries})
            jobs.append({'name': name, 'reference_rows': len(reference), 'original_query_rows': len(query),
                         'reference_excludes_all_full_queries': True})
        model = permutation / task / 'true_replay/risk_rf.joblib'
        checks.append(checked(model, historical['output_sha256'][model.relative_to(permutation).as_posix()]))
        risks.append({'task': task, 'kind': 'enhanced7', 'path': model.relative_to(ROOT).as_posix(),
                      'sha256': digest(model), 'deserialized': False, 'configuration_source': 'permutation_protocol.json'})
        summaries.append({'task': task, 'roles': {k: len(v) for k, v in roles.items()}, 'jobs': jobs})
    serialized = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / 'artifacts').rglob('*')
                        if p.is_file() and p.suffix.lower() in {'.joblib', '.pkl', '.pickle'})
    named_generic = [p for p in serialized if any(token in Path(p).name.lower() for token in ['generic', 'ordinary', 'baseline', 'g0'])]
    assert not named_generic, 'Review possible ordinary models before claiming missing'
    report = {'scope': 'current-project risk serialization paths, pinned model hashes and development identities only',
              'source_model_inventory_count': len(serialized), 'ordinary5_identified': 0,
              'absence_limit': 'No identifiable ordinary5 checkpoint in audited artifacts or phase-A saving code; not a whole-disk absence proof',
              'enhanced7': risks, 'identity_summaries': summaries, 'byte_checks': checks,
              'serialized_path_inventory_sha256': hashlib.sha256('\n'.join(serialized).encode()).hexdigest(),
              'budget_if_separately_authorized': {'ordinary5_fits': 3, 'enhanced7_fits': 0, 'chemprop_fits': 0},
              'actual': {'model_loads': 0, 'fits': 0, 'model_predict_calls': 0, 'real_feature_recomputations': 0,
                         'official_test_records': 0, 'label_or_prediction_values_analyzed': 0},
              'seconds': time.monotonic() - start, 'runner_sha256': digest(Path(__file__))}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'PASS: {len(checks)} pinned byte checks, 9 roles/12 jobs, 3 enhanced weights; 0 identified ordinary5; 0 model loads/fits')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.self_check:
        selfcheck()
    else:
        assert args.output is not None
        run(args.output)
