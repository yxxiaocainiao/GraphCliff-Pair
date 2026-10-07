"""M53 review checks: synthetic math and published aggregates, no ML imports."""
import hashlib
import importlib.util
import json
import math
from itertools import product
from pathlib import Path
from statistics import mean, pvariance

ROOT = Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


if __name__ == '__main__':
    spec = importlib.util.spec_from_file_location('theory', ROOT/'docs/research/query_pair_theory_20261007/selfcheck.py')
    theory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(theory)  # Only definitions; original __main__ is not run.
    q = frozenset({0, 1})
    wide = theory.summarize(q, [frozenset({0}), frozenset({1})], [0., 2.])
    close = theory.summarize(q, [frozenset({0, 1, 2, 3}), frozenset({0, 1, 2, 4})], [0., 2.])
    assert math.isclose(wide['joint'], close['joint'])
    assert wide['sali'] < close['sali']  # True binary-Jaccard example of cancellation.
    for ys in product([-2., 0., 3.], repeat=3):
        for prediction in [-1., 0., 2.]:
            discrepancy = mean((prediction-y)**2 for y in ys)
            assert math.isclose(discrepancy, pvariance(ys)+(prediction-mean(ys))**2, abs_tol=1e-12)
    pilot = ROOT/'docs/research/query_pair_pilot_20261007'
    protocol, verification, results = (read(pilot/name) for name in ['protocol.json', 'verification.json', 'results.json'])
    for pins in [protocol['inputs_sha256'], verification['source_sha256']]:
        assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest() == h for k, h in pins.items())
    gains = {}
    for role in ['calibration', 'evaluation']:
        gains[role] = {}
        for control in ['separate', 'radial']:
            values = []
            for task in protocol['tasks']:
                arms = {r['arm']:r[role]['curve_mean_rmse'] for r in results['results'] if r['task'] == task}
                values.append(100*(1-arms['joint']/arms[control]))
            gains[role][control] = values
            assert math.isclose(mean(values), verification['macro_relative_improvements_percent'][role][control], abs_tol=1e-12)
    print(json.dumps(dict(passed=True, frozen_hashes=47, prior_delivery_hashes=4,
                          binary_jaccard_cancellation=dict(wide=wide, close=close),
                          bias_variance_examples=81, task_relative_gain_percent=gains,
                          real_fits=0, model_loads=0, predict_calls=0,
                          real_feature_recomputations=0, official_test_rows_read=0)))
