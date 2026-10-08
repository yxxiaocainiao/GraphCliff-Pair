"""No estimator fit/predict: poison-label isolation and mocked five-arm/executor wiring."""
import ast
from contextlib import chdir
import json
import math
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments/reliability_base'))
import run_conditional_pilot as pilot


def main():
    protocol = json.loads(pilot.PROTOCOL.read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        poison = folder/'poison.csv'
        poison.write_text('smiles,split,y\nCC,train,1\nCO,train,2\nCN,train,POISON_EVAL\nBAD,train,POISON_CAL\nBAD,test,POISON_TEST\n', encoding='utf-8')
        assert pilot.read_selected(poison, [0, 1], ['y']).y.tolist() == [1, 2]
        assert pilot.read_selected(poison, [0, 1, 2], ['smiles', 'split']).smiles.tolist() == ['CC', 'CO', 'CN']
        try:
            pilot.read_selected(poison, [0, 0], ['y'])
        except ValueError:
            pass
        else:
            raise AssertionError('Duplicate IDs accepted')

        source = pd.DataFrame(np.arange(84).reshape(12, 7), columns=pilot.FEATURES, index=range(12))
        target = source.iloc[:8].copy().set_axis(range(100, 108))
        error = pd.Series(np.arange(12.), index=source.index)
        bits_s = pd.DataFrame(np.zeros((12, 2048)), index=source.index)
        bits_t = pd.DataFrame(np.zeros((8, 2048)), index=target.index)
        counts = dict(scalers=0, pca=0, kde=0, rf=0, predict=0, shared=0)
        class Scaler:
            def fit_transform(self, x):
                assert len(x) == 12
                counts['scalers'] += 1
                return np.asarray(x)
            def transform(self, x):
                assert len(x) == 8
                return np.asarray(x)
        class Projection(Scaler):
            def __init__(self, **kwargs):
                assert kwargs == dict(n_components=7, svd_solver='full', whiten=False)
                self.singular_values_ = np.ones(7)
            def fit_transform(self, x):
                assert len(x) == 12
                counts['pca'] += 1
                return np.ones((12, 7))
            def transform(self, x):
                return np.ones((len(x), 7))
        class Density:
            def __init__(self, **kwargs):
                assert kwargs == dict(bandwidth='scott', kernel='gaussian', metric='euclidean', atol=0, rtol=0)
            def fit(self, x):
                assert len(x) in (12, 8) and x.shape[1] in (5, 7)
                counts['kde'] += 1
                return self
            def score_samples(self, x):
                return np.zeros(len(x))
        class RF:
            def __init__(self, **kwargs):
                assert kwargs == protocol['risk_rf']
            def fit(self, x, y, sample_weight):
                pd.testing.assert_frame_equal(x, source)
                pd.testing.assert_series_equal(y, error)
                assert np.isfinite(sample_weight).all() and np.isclose(np.mean(sample_weight), 1)
                counts['rf'] += 1
                return self
            def predict(self, x):
                pd.testing.assert_frame_equal(x, target)
                counts['predict'] += 1
                return np.ones(len(x))
        def shared(s, t):
            counts['shared'] += 1
            assert list(s.columns) == pilot.FEATURES and list(t.columns) == pilot.FEATURES
            return dict(joint=np.arange(1., 13)/6.5, conditional=np.arange(12., 0, -1)/6.5)
        with patch.multiple(pilot, StandardScaler=Scaler, PCA=Projection, KernelDensity=Density,
                            RandomForestRegressor=RF, joint_and_conditional_weights=shared):
            scores, weights = pilot.five_arms(source, target, error, bits_s, bits_t, protocol)
            assert list(scores.columns) == protocol['arms'] and weights.shape == (12, 5)
            assert np.array_equal(weights['IW-RISK7'], np.arange(1., 13)/6.5)
            assert np.array_equal(weights['IW-COND7'], np.arange(12., 0, -1)/6.5)
            assert counts == dict(scalers=2, pca=1, kde=4, rf=5, predict=5, shared=1)
            before = counts.copy()
            for bad in (error.iloc[::-1], error*float('nan'), -error):
                try:
                    pilot.five_arms(source, target, bad, bits_s, bits_t, protocol)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Invalid supervision accepted')
            assert counts == before
            class RankZero(Projection):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self.singular_values_[-1] = 0
            with patch.object(pilot, 'PCA', RankZero):
                try:
                    pilot.five_arms(source, target, error, bits_s, bits_t, protocol)
                except ValueError as e:
                    assert 'rank below' in str(e)
                else:
                    raise AssertionError('Low rank accepted')
                assert counts['rf'] == 5

        # No actual subprocess/model: verify bounded sequencing and refusal to retry.
        jobs = [dict(name=f'fold{i}', train_command=['train'], predict_command=['predict']) for i in range(4)]
        pilot.save(folder/'jobs.json', jobs)
        pilot.save(folder/'protocol.json', protocol)
        calls = []
        def command(cmd, cwd, timeout, env=None):
            assert 0 < timeout <= 900
            calls.append(cmd)
            if len(cmd) > 2 and cmd[1] == '-c':
                assert Path(cmd[3]).is_absolute() and Path(cmd[4]).is_absolute()
            if cmd == ['train']:
                checkpoint = cwd.parent/'training/model_0/best.pt'
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_text('mock marker, not a model', encoding='utf-8')
            if env is not None:
                assert env['MOLECULEACE_DATA'] == env['QSAR_DATA']
        with patch.object(pilot, 'check'), patch.object(pilot, 'run_command', side_effect=command):
            try:
                pilot.execute(folder)
            except ValueError:
                pass
            else:
                raise AssertionError('Missing authorization accepted')
            assert not calls
            with chdir(folder.parent):
                pilot.execute(Path(folder.name), explicit_training_authorization=True)
            assert len(calls) == 13
            state = json.loads((folder/'execution.json').read_text(encoding='utf-8'))
            assert state['status'] == 'completed' and state['chemprop_fits_started'] == 4
            try:
                pilot.execute(folder, explicit_training_authorization=True)
            except ValueError:
                pass
            else:
                raise AssertionError('Retry accepted')
        failed = folder/'failure'
        failed.mkdir()
        pilot.save(failed/'jobs.json', jobs)
        pilot.save(failed/'protocol.json', protocol)
        with patch.object(pilot, 'check'), patch.object(pilot, 'run_command', side_effect=TimeoutError('mock timeout')) as spy:
            try:
                pilot.execute(failed, explicit_training_authorization=True)
            except TimeoutError:
                pass
            else:
                raise AssertionError('Timeout swallowed')
            assert spy.call_count == 1
            state = json.loads((failed/'execution.json').read_text(encoding='utf-8'))
            assert state['status'] == 'failed_or_interrupted' and state['chemprop_fits_started'] == 1
        exhausted = folder/'exhausted'
        exhausted.mkdir()
        pilot.save(exhausted/'jobs.json', jobs)
        pilot.save(exhausted/'protocol.json', protocol)
        with patch.object(pilot, 'check'), patch.object(pilot, 'run_command') as spy, \
             patch.object(pilot.time, 'monotonic', side_effect=[0., 5401., 5402.]):
            try:
                pilot.execute(exhausted, explicit_training_authorization=True)
            except TimeoutError:
                pass
            else:
                raise AssertionError('Whole-run cap ignored')
            spy.assert_not_called()
        # Exercise the real score-lock/evaluation join and legacy sort on synthetic inputs.
        gate = folder/'gate'
        gate.mkdir()
        jobs = [dict(name=f'fold{i}', query=list(range(i*4, i*4+4))) for i in range(3)]
        jobs += [dict(name='full', query=list(target.index))]
        pilot.save(gate/'jobs.json', jobs)
        pilot.save(gate/'protocol.json', protocol)
        pilot.save(gate/'inputs.json', {'synthetic': True})
        pilot.save(gate/'execution.json', dict(status='running', explicit_training_authorization=True))
        identity = pd.DataFrame(dict(smiles=['CC']*20, canonical=['CC']*20), index=list(source.index)+list(target.index))
        identity.index.name = 'source_row'
        identity.to_csv(gate/'identities.csv')
        error.rename('y').to_csv(gate/'source_labels.csv', index_label='source_row')
        author = gate/'author/src/build_features.py'
        author.parent.mkdir(parents=True)
        author.write_bytes((pilot.AUTHOR/'src/build_features.py').read_bytes())
        for job in jobs:
            path = gate/protocol['task']/job['name']
            (path/'features/cache').mkdir(parents=True)
            pd.DataFrame(dict(smiles=['CC']*len(job['query']))).to_csv(path/'query.csv', index=False)
            pd.DataFrame(dict(smiles=['CC']*len(job['query']), y=np.arange(len(job['query'])))).to_csv(path/'predictions.csv', index=False)
            f = pd.DataFrame(1., index=range(len(job['query'])), columns=pilot.FEATURES[1:])
            f.insert(0, 'smiles', 'CC')
            f.to_csv(path/'features/cache/fixture.csv', index=False)
        score = pd.DataFrame({arm: np.array([0, 0, 2, 1, 3, 3, 4, 5.]) for arm in protocol['arms']}, index=target.index)
        truth = pd.DataFrame({'y': np.ones(8)}, index=target.index)
        real_sha = pilot.sha
        def locked_truth(path, ids, columns):
            assert (gate/'score_lock.json').exists()
            lock = json.loads((gate/'score_lock.json').read_text(encoding='utf-8'))
            assert lock['scores_sha256'] == real_sha(gate/'scores.csv')
            assert ids == list(target.index) and columns == ['y']
            return truth
        with patch.object(pilot, 'five_arms', return_value=(score, weights)), \
             patch.object(pilot, 'sha', side_effect=lambda path: pilot.SOURCE_SHA if path == pilot.SOURCE else real_sha(path)), \
             patch.object(pilot, 'read_selected', side_effect=locked_truth) as label_read:
            pilot.postprocess(gate)
            assert label_read.call_count == 1
        tree = ast.parse((ROOT/'experiments/reliability_base/run_phase_a.py').read_text(encoding='utf-8'))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'metrics')
        cut = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'scale' for t in n.targets))
        fn.body = fn.body[:cut]+[ast.Return(value=ast.Name(id='curve', ctx=ast.Load()))]
        env = dict(np=np, math=math)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), '<legacy sort>', 'exec'), env)
        frame = pd.DataFrame(dict(prediction=np.arange(8.), y=1., canonical='CC', source_row=target.index))
        expected = env['metrics'](frame, score.RF7.to_numpy(), None, None, None, protocol)
        results = json.loads((gate/'results.json').read_text(encoding='utf-8'))
        assert np.allclose(results['RF7']['curve_rmse'], expected)
    print(json.dumps(dict(status='passed', actual_estimator_fits=0, actual_predictions=0,
                         mock_risk_arms=5, mock_subprocess_stages=13, timeout_stops=1,
                         poison_label_isolation=True, score_lock_before_truth=True,
                         legacy_curve_match=True, source_rank_stop=True,
                         whole_run_cap_stop=True, real_data_reads=0)))


if __name__ == '__main__':
    main()
