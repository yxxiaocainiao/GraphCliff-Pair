"""Audit completed pilot and export aggregate curves/exchanges; no fit or feature recomputation."""
import json
import math
import re
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments/reliability_base'))
import run_conditional_pilot as pilot


def main():
    public = Path(__file__).parent
    output = ROOT/'artifacts/conditional_pilot_inputs_20261008_v2'
    state = json.loads((output/'execution.json').read_text(encoding='utf-8'))
    if state['status'] != 'completed':
        raise ValueError('Pilot is incomplete; do not compute an effectiveness conclusion')
    pilot.check(output)
    protocol = json.loads((output/'protocol.json').read_text(encoding='utf-8'))
    launch = json.loads((output/'launch_record.json').read_text(encoding='utf-8'))
    lock = json.loads((output/'score_lock.json').read_text(encoding='utf-8'))
    for key, path in [('scores_sha256', output/'scores.csv'), ('inputs_sha256', output/'inputs.json'),
                      ('runner_sha256', Path(pilot.__file__)), ('protocol_sha256', output/'protocol.json')]:
        assert pilot.sha(path) == lock[key], key
    assert launch['runner_sha256'] == lock['runner_sha256']
    assert launch['inputs_sha256'] == lock['inputs_sha256']
    assert pilot.sha(pilot.SOURCE) == pilot.SOURCE_SHA
    jobs = json.loads((output/'jobs.json').read_text(encoding='utf-8'))
    identities = pd.read_csv(output/'identities.csv', index_col='source_row')
    weights = pd.read_csv(output/'weights.csv', index_col='source_row')
    scores = pd.read_csv(output/'scores.csv', index_col='source_row')
    assert list(scores.columns) == list(weights.columns) == protocol['arms'] == lock['arms']
    assert weights.index.tolist() == jobs[-1]['fit'] and scores.index.tolist() == jobs[-1]['query']
    assert np.isfinite(weights).all().all() and (weights >= 0).all().all()
    assert np.allclose(weights.mean(), 1) and np.isfinite(scores).all().all()
    assert weights.gt(0).sum().ge(2).all()
    timing = {}
    point_audit = {}
    from chemprop.cli.main import construct_parser
    parser = construct_parser()
    for job in jobs:
        folder = output/protocol['task']/job['name']
        pred = pd.read_csv(folder/'predictions.csv', usecols=['smiles', 'y'])
        feature = pd.read_csv(folder/'features/cache/fixture.csv', usecols=['smiles']+pilot.FEATURES[1:])
        expected = identities.loc[job['query'], 'smiles'].tolist()
        assert pred.smiles.tolist() == feature.smiles.tolist() == expected
        assert np.isfinite(pred.y).all() and np.isfinite(feature[pilot.FEATURES[1:]]).all().all()
        config = folder/'training/config.toml'
        # Chemprop writes configargparse syntax despite the .toml extension.
        actual = parser.parse_args(['train', '--config-path', str(config)])
        expected_args = parser.parse_args(job['train_command'][4:])
        for key in ['epochs', 'patience', 'batch_size', 'data_seed', 'pytorch_seed',
                    'num_workers', 'accelerator', 'devices', 'splits_column',
                    'loss_function', 'task_type', 'ensemble_size']:
            assert getattr(actual, key) == getattr(expected_args, key), key
        for name, role in [('train', 'fit'), ('val', 'monitor')]:
            saved = pd.read_csv(folder/'training'/(name+'.csv'), usecols=['smiles'])
            assert saved.smiles.tolist() == identities.loc[job[role], 'smiles'].tolist()
        epochs = re.findall(r'Epoch\s+(\d+)', (folder/'train_log/stdout.txt').read_text(encoding='utf-8', errors='replace'))
        point_audit[job['name']] = dict(config_sha256=pilot.sha(config),
             last_displayed_epoch_zero_based=int(epochs[-1]) if epochs else None,
             saved_fit_rows=len(job['fit']), saved_monitor_rows=len(job['monitor']))
        timing[job['name']] = {}
        for step, cap in [('train', 900), ('predict', 120), ('feature', 300)]:
            metadata = json.loads((folder/(step+'_log')/'exit.json').read_text(encoding='utf-8'))
            assert metadata['returncode'] == 0 and metadata['seconds'] < cap
            timing[job['name']][step] = metadata['seconds']
    post = json.loads((output/'postprocess_log/exit.json').read_text(encoding='utf-8'))
    assert post['returncode'] == 0 and post['seconds'] < 600
    assert state['elapsed_seconds'] < 5400 and sum(v['train'] for v in timing.values()) < 3600
    assert state['chemprop_fits_started'] == state['point_predict_batches_started'] == state['auxiliary_rf_batches_started'] == 4
    # Scoring has already finished: only re-read the exact evaluation labels allowed by the score lock.
    truth = pilot.read_selected(pilot.SOURCE, scores.index.tolist(), ['y']).y
    error2 = np.square(pred.y.to_numpy()-truth.to_numpy())
    orders = {arm: sorted(range(len(scores)), key=lambda i: (scores[arm].iloc[i],
              identities.loc[scores.index[i], 'canonical'], int(scores.index[i]))) for arm in protocol['arms']}
    original = json.loads((output/'results.json').read_text(encoding='utf-8'))
    curves, exchanges = [], []
    for arm in protocol['arms']:
        for k, fraction in enumerate(protocol['coverage_grid']):
            count = math.ceil(fraction*len(scores))
            assert count == protocol['expected_accept_counts'][k]
            rmse = float(np.sqrt(np.mean(error2[orders[arm][:count]])))
            assert np.isclose(rmse, original[arm]['curve_rmse'][k], atol=1e-12, rtol=0)
            curves.append(dict(arm=arm, coverage=fraction, accepted=count, rmse=rmse))
    assert np.ptp([r['rmse'] for r in curves if r['coverage'] == 1]) < 1e-12
    candidate = 'IW-COND7'
    for arm in protocol['arms']:
        if arm == candidate:
            continue
        for k, fraction in enumerate(protocol['coverage_grid']):
            count = protocol['expected_accept_counts'][k]
            a, b = set(orders[arm][:count]), set(orders[candidate][:count])
            common, added, removed = a & b, b-a, a-b
            assert common.isdisjoint(added | removed) and added.isdisjoint(removed)
            assert common | added == b and common | removed == a and len(added) == len(removed)
            add_sse, remove_sse = sum(error2[i] for i in added), sum(error2[i] for i in removed)
            delta = float((add_sse-remove_sse)/count)
            direct = float((sum(error2[i] for i in b)-sum(error2[i] for i in a))/count)
            assert np.isclose(delta, direct, atol=1e-12, rtol=0)
            assert np.isclose(delta, original[candidate]['curve_rmse'][k]**2-original[arm]['curve_rmse'][k]**2, atol=1e-12, rtol=0)
            signed = [float(error2[i]) for i in added]+[-float(error2[i]) for i in removed]
            largest = max(signed, key=abs) if signed else 0.
            mass = sum(abs(v) for v in signed)
            exchanges.append(dict(control=arm, coverage=fraction, accepted=count, common=len(common),
                 added=len(added), removed=len(removed), added_sse=float(add_sse), removed_sse=float(remove_sse),
                 candidate_minus_control_mse=delta, largest_signed_sse=largest,
                 largest_absolute_share=(abs(largest)/mass if mass else 0.),
                 mse_delta_without_largest_contribution=delta-largest/count))
    diagnostics = {arm: dict(ess=float(weights[arm].sum()**2/np.square(weights[arm]).sum()),
                      max_share=float(weights[arm].max()/weights[arm].sum()),
                      positive_rows=int(weights[arm].gt(0).sum()),
                      quantiles={str(q):float(weights[arm].quantile(q)) for q in [0,.5,.9,.99,1]})
                   for arm in protocol['arms']}
    means = {arm:original[arm]['mean_of_six_rmse'] for arm in protocol['arms']}
    for arm in means:
        assert np.isclose(means[arm], np.mean([r['rmse'] for r in curves if r['arm']==arm]), atol=1e-12, rtol=0)
    comparisons = {arm:dict(candidate_minus_control=means[candidate]-means[arm],
                   relative_reduction_percent=100*(means[arm]-means[candidate])/means[arm])
                   for arm in means if arm != candidate}
    passed = all(means[candidate] < means[arm] for arm in comparisons)
    pd.DataFrame(curves).to_csv(public/'curves.csv', index=False)
    pd.DataFrame(exchanges).to_csv(public/'exchanges.csv', index=False)
    result = dict(milestone='M66',status='completed_and_checked',task=protocol['task'],seed=42,
       means=means,comparisons=comparisons,necessary_continuation_condition_met=passed,
       no_automatic_expansion=True,weight_diagnostics=diagnostics,timings=timing,
       point_training_audit=point_audit,
       postprocess_seconds=post['seconds'],execution=state,
       completed_path_counts=dict(chemprop=4,auxiliary_rf=4,risk_rf=5,kde=8,scaler=2,pca=1),
       counts_evidence='four logged point/feature jobs plus completed frozen postprocess code path; no per-estimator instrumentation',
       official_test_record_parsing=False,calibration_record_parsing=False,
       target_labels_first_read_after_score_lock=True,curves_checked=30,exchange_rows_checked=24,
       diagnostics_fits=0,diagnostics_predictions=0,
       audit_history=['initial aggregate audit passed',
          'supplementary read-only config audit: tomllib failed because Chemprop config.toml uses configargparse syntax',
          'corrected audit uses official Chemprop parser only, never calls CLI handler; no training retry'],
       checks=dict(source_order=True,finite_features=True,score_config_hashes=True,
                   acceptance_counts=True,partitions_and_exchange_identity=True,full_acceptance_equal=True),
       evidence_hashes={name:pilot.sha(output/name) for name in ['launch_record.json','inputs.json','score_lock.json','scores.csv','weights.csv','results.json','execution.json']})
    pilot.save(public/'verification.json', result)
    print(json.dumps(dict(means=means,comparisons=comparisons,necessary_condition=passed)))


if __name__ == '__main__':
    main()
