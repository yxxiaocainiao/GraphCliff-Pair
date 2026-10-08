"""One author auxiliary-RF batch; paired input sensitivity only, never target truth."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd

from prepare_conditional_pilot import ROOT
from run_phase_a import run_command, save, sha

BASE = ROOT/'artifacts/conditional_pilot_inputs_20261008_v2'
OUTPUT = ROOT/'artifacts/reference_context_check_20261008'
PUBLIC = ROOT/'docs/research/reference_context_check_20261008'
COLUMNS = ['nn_sim', 'local_dens', 'mol_size', 'rf_var', 'nbr_disp', 'sali_mean']
TOLERANCE = 1e-12


def inputs():
    prepared = json.loads((BASE/'inputs.json').read_text(encoding='utf-8'))
    for name, expected in prepared['files'].items():
        assert sha(BASE/name) == expected, name
    prior = json.loads((ROOT/'docs/research/conditional_pilot_run_20261008/verification.json').read_text(encoding='utf-8'))
    for name, expected in prior['evidence_hashes'].items():
        assert sha(BASE/name) == expected, name
    review = json.loads((ROOT/'docs/research/reference_context_review_20261008/verification.json').read_text(encoding='utf-8'))
    assert sha(BASE/'CHEMBL234_Ki/full/features/cache/fixture.csv') == review['cache_hashes_observed_now']['full']
    assert importlib.metadata.version('scikit-learn') == '1.9.0'
    assert importlib.metadata.version('rdkit') == '2026.3.4'
    jobs = json.loads((BASE/'jobs.json').read_text(encoding='utf-8'))
    assert jobs[0]['name'] == 'fold0' and jobs[-1]['name'] == 'full'
    assert len(jobs[0]['fit']) == 935 and len(jobs[-1]['query']) == 585
    assert set(jobs[0]['fit']) < set(jobs[-1]['fit'])
    assert not set(jobs[0]['fit']) & set(jobs[-1]['query'])
    return jobs


def audit():
    jobs = inputs()
    state = json.loads((OUTPUT/'state.json').read_text(encoding='utf-8'))
    assert state['status'] in ('features_completed', 'completed')
    manifest = json.loads((OUTPUT/'inputs.json').read_text(encoding='utf-8'))
    assert sha(Path(__file__)) == manifest['runner_sha256']
    for name, expected in manifest['files'].items():
        assert sha(OUTPUT/name) == expected, name
    assert sha(OUTPUT/'fixed_predictions.csv') == sha(BASE/'CHEMBL234_Ki/full/predictions.csv')
    ids = pd.read_csv(BASE/'identities.csv', index_col='source_row', usecols=['source_row', 'smiles'])
    expected = ids.loc[jobs[-1]['query'], 'smiles'].tolist()
    original = pd.read_csv(BASE/'CHEMBL234_Ki/full/features/cache/fixture.csv', usecols=['smiles']+COLUMNS)
    changed = pd.read_csv(OUTPUT/'features/cache/fixture.csv', usecols=['smiles']+COLUMNS)
    prediction = pd.read_csv(OUTPUT/'fixed_predictions.csv', usecols=['smiles', 'y'])
    assert original.smiles.tolist() == changed.smiles.tolist() == prediction.smiles.tolist() == expected
    a, b = original[COLUMNS].copy(), changed[COLUMNS].copy()
    assert np.isfinite(a).all().all() and np.isfinite(b).all().all() and np.isfinite(prediction.y).all()
    assert np.array_equal(a.mol_size, b.mol_size)
    assert (a.nn_sim+TOLERANCE >= b.nn_sim).all()
    assert (a.local_dens+TOLERANCE >= b.local_dens).all()
    a.insert(0, 'prediction', prediction.y.to_numpy())
    b.insert(0, 'prediction', prediction.y.to_numpy())
    statistics = []
    for column in a:
        difference = b[column]-a[column]
        absolute = difference.abs()
        correlation = a[column].corr(b[column], method='spearman') if min(a[column].nunique(), b[column].nunique()) > 1 else None
        statistics.append(dict(feature=column, full_mean=float(a[column].mean()), fold0_mean=float(b[column].mean()),
             mean_difference=float(difference.mean()), median_absolute_difference=float(absolute.median()),
             p90_absolute_difference=float(absolute.quantile(.9)), max_absolute_difference=float(absolute.max()),
             spearman=float(correlation) if correlation is not None else None,
             numerically_changed_rows=int(absolute.gt(TOLERANCE).sum())))
    exit_data = json.loads((OUTPUT/'feature_log/exit.json').read_text(encoding='utf-8'))
    assert exit_data['returncode'] == 0 and exit_data['seconds'] < 300
    return dict(milestone='M68',status='passed',task='CHEMBL234_Ki',query_rows=585,
         full_reference_rows=1403,subset_reference_rows=935,statistics=statistics,
         auxiliary_rf_fits=1,auxiliary_feature_batches=1,auxiliary_tree_prediction_query_rows=585,
         chemprop_fits_or_predictions=0,risk_rf_fits_or_predictions=0,model_checkpoint_loads=0,
         kde_scaler_pca_fits=0,target_truth_reads=0,calibration_record_reads=0,official_test_record_reads=0,
         prediction_byte_identical=True,mol_size_identical=True,similarity_monotonicity_passed=True,
         numerical_tolerance=TOLERANCE,feature_seconds=exit_data['seconds'],
         evidence_hashes=dict(input_manifest=sha(OUTPUT/'inputs.json'),
              fixed_predictions=sha(OUTPUT/'fixed_predictions.csv'),new_cache=sha(OUTPUT/'features/cache/fixture.csv'),
              full_cache=sha(BASE/'CHEMBL234_Ki/full/features/cache/fixture.csv'),runner=sha(Path(__file__))))


def run():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='preparing',auxiliary_rf_batches_started=0,automatic_retries=0)
    save(OUTPUT/'state.json', state)
    try:
        jobs = inputs()
        (OUTPUT/'features/data').mkdir(parents=True)
        reference = pd.read_csv(BASE/'CHEMBL234_Ki/fold0/train_val.csv', usecols=['smiles','y','partition'], nrows=935)
        query = pd.read_csv(BASE/'CHEMBL234_Ki/full/query.csv', usecols=['smiles'])
        ids = pd.read_csv(BASE/'identities.csv',index_col='source_row',usecols=['source_row','smiles'])
        assert reference.partition.eq('train').all() and np.isfinite(reference.y).all()
        assert reference.smiles.tolist() == ids.loc[jobs[0]['fit'],'smiles'].tolist()
        assert query.smiles.tolist() == ids.loc[jobs[-1]['query'],'smiles'].tolist()
        fixture = pd.concat([reference[['smiles','y']].assign(cliff_mol=0,split='train'),
                             query.assign(y=0.,cliff_mol=0,split='test')],ignore_index=True)
        assert len(fixture) == 1520 and fixture.iloc[935:].y.eq(0).all()
        fixture.to_csv(OUTPUT/'features/data/fixture.csv',index=False)
        shutil.copyfile(BASE/'CHEMBL234_Ki/full/predictions.csv',OUTPUT/'fixed_predictions.csv')
        save(OUTPUT/'inputs.json',dict(runner_sha256=sha(Path(__file__)),
             author_sha256=sha(BASE/'author/src/build_features.py'),
             files={str(p.relative_to(OUTPUT)):sha(p) for p in [OUTPUT/'features/data/fixture.csv',OUTPUT/'fixed_predictions.csv']},
             reference_rows=935,query_rows=585,query_y='dummy_zero',
             authorization='User continue after M67 single auxiliary-RF specification',
             feature_seconds_cap=300,whole_run_seconds_cap=600))
        environment = dict(os.environ, MOLECULEACE_DATA=str(OUTPUT/'features/data'),
            QSAR_DATA=str(OUTPUT/'features/data'),QSAR_CACHE=str(OUTPUT/'features/cache'),
            QSAR_RESULTS=str(OUTPUT/'features/results'),QSAR_FIGURES=str(OUTPUT/'features/figures'))
        remaining = 600-(time.monotonic()-started)
        if remaining <= 0:
            raise TimeoutError('Whole-run budget exhausted before feature batch')
        state.update(status='running',auxiliary_rf_batches_started=1)
        save(OUTPUT/'state.json',state)
        run_command([sys.executable,str(BASE/'author/src/build_features.py'),'fixture'],
                    OUTPUT/'feature_log',min(300,remaining),environment)
        state['status']='features_completed'
        save(OUTPUT/'state.json',state)
        result = audit()
        if time.monotonic()-started >= 600:
            raise TimeoutError('Whole-run budget exceeded')
        state['status']='completed'
        state['elapsed_seconds']=time.monotonic()-started
        save(OUTPUT/'state.json',state)
        result['elapsed_seconds']=state['elapsed_seconds']
        PUBLIC.mkdir(parents=True,exist_ok=True)
        save(PUBLIC/'verification.json',result)
        print(json.dumps(result))
    except BaseException as error:
        state.update(status='failed_or_interrupted',error=repr(error),elapsed_seconds=time.monotonic()-started)
        save(OUTPUT/'state.json',state)
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['run','check'])
    args=parser.parse_args()
    if args.mode=='run':
        run()
    else:
        print(json.dumps(audit()))
