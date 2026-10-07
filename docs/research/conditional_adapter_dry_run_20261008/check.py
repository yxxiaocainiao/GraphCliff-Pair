"""Only synthetic KDEs and legacy sorting; no real data, predictors or RF fits."""
import ast
import argparse
import importlib.util
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('conditional_risk_weights', ROOT / 'experiments/reliability_base/conditional_risk_weights.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def density_check(source, target):
    original_fit = KernelDensity.fit
    with patch.object(KernelDensity, 'fit', autospec=True, side_effect=original_fit) as spy:
        shifted = adapter.joint_and_conditional_weights(source, target)
        assert spy.call_count == 4
        assert np.isclose(shifted['conditional'].mean(), 1) and np.ptp(shifted['conditional']) > 1e-5
        assert np.allclose(shifted['log_joint'], shifted['log_generic'] + shifted['log_conditional'])
        assert np.isclose(shifted['source_bandwidth'], 12 ** (-1/11))
        assert np.isclose(shifted['target_bandwidth'], 8 ** (-1/11))
        # R constant: conditional ratio is a G-independent bandwidth constant.
        s0, t0 = source.copy(), target.copy()
        s0.iloc[:, 5:] = 0
        t0.iloc[:, 5:] = 0
        t0.iloc[:, :5] += 0.6
        noop = adapter.joint_and_conditional_weights(s0, t0)
        assert spy.call_count == 8
        assert np.allclose(noop['conditional'], 1, atol=1e-12)
        assert np.allclose(noop['log_conditional'], 2 * math.log(noop['source_bandwidth'] / noop['target_bandwidth']))


def main(validation_only=False):
    source = pd.DataFrame(np.sin(np.arange(84).reshape(12, 7) / 7), columns=adapter.FEATURES, index=range(12))
    target = source.iloc[:8].copy()
    target.index = range(100, 108)
    target.iloc[:, 5:] += 0.5
    if not validation_only:
        density_check(source, target)
    # No silent clipping; a common offset cannot overflow normalization.
    assert np.allclose(adapter.normalized_weights([10000, 10001]), adapter.normalized_weights([0, 1]))
    invalid = [source.assign(y=0), source[adapter.FEATURES[::-1]], source.iloc[:1],
               source.set_axis([0] * len(source)), source.assign(nn_sim=np.nan), source.astype(str),
               source.astype(complex), source.astype(bool)]
    with patch.object(KernelDensity, 'fit', side_effect=AssertionError('Invalid input reached fit')):
        for bad in invalid:
            try:
                adapter.joint_and_conditional_weights(bad, target)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid evidence accepted')
        try:
            adapter.joint_and_conditional_weights(source, target.set_axis(range(8)))
        except ValueError:
            pass
        else:
            raise AssertionError('Overlapping identities accepted')
    for bad in ([np.nan, 0], [0], [0, -10000]):
        try:
            adapter.normalized_weights(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid log weights accepted')
    # Reuse precisely the legacy evaluation prefix, before calibration code.
    tree = ast.parse((ROOT / 'experiments/reliability_base/run_phase_a.py').read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'metrics')
    cut = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'scale' for t in n.targets))
    fn.body = fn.body[:cut] + [ast.Return(value=ast.Tuple(elts=[ast.Name(id='curve', ctx=ast.Load()), ast.Name(id='order', ctx=ast.Load())], ctx=ast.Load()))]
    env = {'np': np, 'math': math}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), '<legacy metrics prefix>', 'exec'), env)
    frame = pd.DataFrame({'prediction': [0., 1., 2., 3., 4., 5.], 'y': [0.] * 6,
                          'canonical': ['a', 'a', 'c', 'd', 'e', 'f'], 'source_row': [11, 10, 12, 13, 14, 15]})
    grid = {'coverage_grid': [.5, .6, .7, .8, .9, 1.]}
    scores_a, scores_b = [0, 0, 1, 2, 3, 4], [2, 2, 0, 1, 3, 4]
    ca, oa = env['metrics'](frame, np.array(scores_a), None, None, None, grid)
    cb, ob = env['metrics'](frame, np.array(scores_b), None, None, None, grid)
    assert oa[:2] == [1, 0] and ca[-1] == cb[-1]
    for c in grid['coverage_grid']:
        m = math.ceil(c * len(frame))
        a, b = set(oa[:m]), set(ob[:m])
        assert len(b - a) == len(a - b)
        e2 = np.square(frame.prediction.to_numpy())
        delta = (sum(e2[i] for i in b) - sum(e2[i] for i in a)) / m
        assert np.isclose(delta, (sum(e2[i] for i in b-a) - sum(e2[i] for i in a-b)) / m)
    print(json.dumps({'status': 'passed', 'synthetic_kde_fits': 0 if validation_only else 8,
        'rejected_evidence_cases_before_fit': 9, 'rejected_log_weight_cases': 3,
        'legacy_sort_and_exchange_coverages': 6, 'calibration_objects': None,
        'chemprop_aux_rf_risk_rf_fits': 0, 'real_data_reads': 0,
        'skada_pipeline_executed': False}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-only', action='store_true', help='Resume input/sort checks without repeating synthetic fits')
    main(parser.parse_args().validation_only)
