"""Frozen score-only exchange diagnostic. Stdlib; no ML imports or model access."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n', encoding='utf-8')


def sign(x):
    return 0 if abs(x) <= 1e-12 else (1 if x > 0 else -1)


def exchange(rows, fraction):
    n = len(rows)
    assert n > 0 and 0 < fraction <= 1
    m = math.ceil(fraction * n)
    # Same ordering contract as run_phase_a.metrics; bound source hash and replay below.
    sets = [set(sorted(range(n), key=lambda i: (rows[i][arm],
                rows[i]['canonical'], rows[i]['source_row']))[:m])
            for arm in ['generic_risk', 'augmented_risk']]
    g, a = sets
    common, added, removed, neither = g & a, a - g, g - a, set(range(n)) - (a | g)
    groups = [common, added, removed, neither]
    assert set.union(*groups) == set(range(n))
    assert sum(map(len, groups)) == n and len(added) == len(removed)
    e2 = [(r['prediction'] - r['y']) ** 2 for r in rows]
    total = lambda ids: math.fsum(e2[i] for i in sorted(ids))
    gm, am = total(g) / m, total(a) / m
    contributions = [(i, e2[i]) for i in sorted(removed)] + [(i, -e2[i]) for i in sorted(added)]
    net = math.fsum(c for _, c in contributions)
    gain = net / m
    assert math.isclose(gm - am, gain, abs_tol=1e-12, rel_tol=1e-12)
    gross = math.fsum(abs(c) for _, c in contributions)
    zeroed = [(net - c) / m for _, c in contributions]
    record = dict(fraction=fraction, rows=n, accepted=m, common=len(common),
                  added=len(added), removed=len(removed), neither=len(neither),
                  generic_rmse=math.sqrt(gm), augmented_rmse=math.sqrt(am),
                  common_sse=total(common), added_sse=total(added), removed_sse=total(removed),
                  gain_mse=gain, gain_rmse=math.sqrt(gm) - math.sqrt(am),
                  max_abs_share_gross=max((abs(c) for _, c in contributions), default=0) / gross if gross else None,
                  single_contribution_sign_reversals=sum(sign(gain) * sign(x) < 0 for x in zeroed),
                  single_contribution_zeroed_gain_range=[min(zeroed), max(zeroed)] if zeroed else None)
    if fraction == 1:
        assert not added and not removed and gm == am and gain == 0
    private = [dict(source_row=rows[i]['source_row'], canonical=rows[i]['canonical'],
                    side='removed' if i in removed else 'added', signed_sse=c)
               for i, c in contributions]
    return record, private


def selfcheck():
    rows = [dict(source_row=i, canonical=c, y=y, prediction=0., generic_risk=g,
                 augmented_risk=a) for i, c, y, g, a in
            [(2, 'b', 4., 0., 2.), (1, 'a', 1., 0., 0.),
             (3, 'c', 2., 2., 0.), (4, 'd', 3., 3., 3.)]]
    r, p = exchange(rows, .5)
    assert (r['common'], r['added'], r['removed'], r['neither']) == (1, 1, 1, 1)
    assert r['gain_mse'] == 6 and {v['source_row'] for v in p} == {2, 3}
    assert r['single_contribution_sign_reversals'] == 1
    assert r['max_abs_share_gross'] == .8
    assert exchange(rows, .6)[0]['accepted'] == 3
    assert exchange(rows, 1.)[0]['gain_mse'] == 0
    same = [dict(r, augmented_risk=r['generic_risk']) for r in rows]
    assert not exchange(same, .5)[1]
    ties = [dict(r, generic_risk=0., augmented_risk=0.) for r in rows]
    assert not exchange(ties, .5)[1]
    for value in [0., -1., 1.1]:
        try:
            exchange(rows, value)
        except AssertionError:
            pass
        else:
            raise AssertionError('Invalid coverage accepted')
    assert sign(1e-14) == 0 and sign(-1) == -1
    print('PASS: partition, equal swap counts, SSE identity, ceil, ties, identical/full sets, invalid coverage')


def run(output):
    assert output.resolve().parent == (ROOT / 'artifacts').resolve()
    assert not output.exists(), 'Refuse overwrite; preserve previous attempt'
    output.mkdir()
    tick = time.monotonic()
    state = dict(status='running',attempts=1, fits=0, model_loads=0,
                 model_predict_calls=0, real_feature_recomputation=0, official_test_rows=0,
                 parameter_searches=0, code_commit=subprocess.check_output(
                     ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    save(output / 'execution.json', state)
    try:
        plan = json.loads((HERE / 'protocol.json').read_text(encoding='utf-8'))
        inputs = dict(plan['inputs'])
        for p in [HERE / 'protocol.json', Path(__file__)]:
            inputs[p.relative_to(ROOT).as_posix()] = sha(p)
        for p, h in inputs.items():
            assert sha(ROOT / p) == h, p
        old = json.loads((ROOT / 'experiments/reliability_base/phase_a_results.json').read_text(encoding='utf-8'))
        public, details, comparisons, counts = [], [], [], []
        for task in plan['tasks']:
            manifest = json.loads((ROOT / plan['partition_directory'] / (task + '.json')).read_text(encoding='utf-8'))
            role_sets, role_records = {}, {}
            all_roles = [set(ids) for ids in manifest['roles'].values()]
            assert len(set.union(*all_roles)) == sum(map(len, all_roles))
            for role in plan['roles']:
                path = ROOT / plan['input_directory'] / task / (role + '.csv')
                with path.open(encoding='utf-8', newline='') as stream:
                    reader = csv.DictReader(stream)
                    required = ['source_row', 'canonical', 'y', 'prediction'] + plan['arms']
                    assert set(required) <= set(reader.fieldnames)
                    rows = [dict(source_row=int(r['source_row']), canonical=r['canonical'],
                                 **{k: float(r[k]) for k in required[2:]}) for r in reader]
                assert len(rows) == len({r['source_row'] for r in rows}) == len({r['canonical'] for r in rows})
                assert {r['source_row'] for r in rows} == set(manifest['roles'][role])
                assert all(r['canonical'] and all(math.isfinite(r[k]) for k in required[2:]) for r in rows)
                role_sets[role] = {r['canonical'] for r in rows}
                role_records[role] = []
                counts.append(dict(task=task, role=role, rows=len(rows)))
                for fraction in plan['coverage_grid']:
                    r, p = exchange(rows, fraction)
                    r.update(task=task, role=role)
                    public.append(r); role_records[role].append(r)
                    details.append(dict(task=task, role=role, fraction=fraction, exchanges=p))
                if role == 'evaluation':
                    saved = next(t for t in old['results'] if t['dataset'] == task)['arms']
                    for arm in ['generic', 'augmented']:
                        assert all(math.isclose(r[arm + '_rmse'], expected, abs_tol=1e-12, rel_tol=0)
                                   for r, expected in zip(role_records[role], saved[arm]['curve_rmse']))
                assert time.monotonic() - tick < plan['execution']['wall_seconds_limit']
            assert not set.intersection(*role_sets.values())
            for c, e in zip(role_records['calibration'][:-1], role_records['evaluation'][:-1]):
                comparisons.append(dict(task=task, fraction=c['fraction'], calibration_sign=sign(c['gain_mse']),
                                        evaluation_sign=sign(e['gain_mse']), same_sign=sign(c['gain_mse']) == sign(e['gain_mse'])))
        assert len(public) == 36 and len(comparisons) == 15
        for p, h in inputs.items():
            assert sha(ROOT / p) == h, p
        save(output / 'exchanges_private.json', details)
        result = dict(cells=public, role_sign_comparisons=comparisons, counts=counts,
                      inputs_sha256=inputs, private_sha256=sha(output / 'exchanges_private.json'),
                      code_commit=state['code_commit'], scope=plan['interpretation'],
                      validation=dict(cells=36, historical_curve_values=36, partition_identity=True,
                                      numeric_tables=6, role_rows=sum(r['rows'] for r in counts), hashes=len(inputs)))
        save(output / 'results.json', result)
        state.update(status='complete', seconds=time.monotonic() - tick)
        print(json.dumps(dict(status='complete', cells=36, counts=counts, seconds=state['seconds'])))
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), seconds=time.monotonic() - tick)
        raise
    finally:
        save(output / 'execution.json', state)


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
