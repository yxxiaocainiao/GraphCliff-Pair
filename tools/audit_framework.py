"""Read-only historical validation inventory; publish aggregates, never test rows."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import subprocess
import numpy as np

ROOT = Path('D:/GraphCliff-Pair')
OLD = Path('D:/GraphCliff-main')
BASE = Path('D:/WORK_SPACE/WORK_SPACE/my_work/graphcliff_baselines')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def f32(value):
    return struct.unpack('f', struct.pack('f', float(value)))[0]


def rmse(records, dtype):
    output = {}
    for name, flag in [('overall_rmse', None), ('cliff_rmse', 1), ('noncliff_rmse', 0)]:
        selected = [r for r in records if flag is None or int(float(r['cliff_mol'])) == flag]
        errors = []
        for r in selected:
            prediction = float(r.get('prediction', r.get('y_pred')))
            label = float(r.get('label_float32', r['y'])) if dtype == 'float32' else float(r['y'])
            if dtype == 'float32':
                label = f32(label)
            assert math.isfinite(prediction) and math.isfinite(label)
            errors.append((prediction-label)**2)
        output[name] = math.sqrt(statistics.mean(errors)) if errors else None
    return output


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    hashes, records, incomplete, errors = {}, [], [], []
    def track(path):
        hashes[str(path)] = sha(path)
    for owner, folder in [('original', OLD/'eval_out'), ('pair', ROOT/'artifacts'), ('baselines', BASE/'results')]:
        for path in sorted(folder.rglob('validation_predictions.csv')):
            parent = path.parent
            metadata = next((p for p in [parent/'summary.json', parent/'result.json', parent/'manifest.json', parent/'metrics.json'] if p.exists()), None)
            if metadata is None:
                continue
            try:
                info = load(metadata)
                if not isinstance(info, dict):
                    continue
                predictions = rows(path)
                if not predictions or not {'y', 'cliff_mol'}.issubset(predictions[0]):
                    continue
                for file in [path, metadata, parent/'history.csv', parent/'config.json', parent/'manifest.json']:
                    if file.exists():
                        track(file)
                checkpoint = next((p for p in [parent/'best.pt', parent/'best_model.pt', parent/'best.pth', parent/'checkpoint.pt'] if p.exists()), None)
                if owner == 'baselines':
                    candidate = BASE/'checkpoints'/parent.relative_to(BASE/'results')/'best.pt'
                    if candidate.exists():
                        checkpoint = candidate
                if checkpoint:
                    track(checkpoint)
                dtype = 'float32' if owner in ['original', 'pair'] else 'float64'
                computed = rmse(predictions, dtype)
                available = info.get('validation', info.get('val', info))
                differences = {k: computed[k]-float(available[k]) for k in computed
                               if isinstance(available, dict) and available.get(k) is not None and computed[k] is not None}
                label = str(parent.relative_to(folder))
                history = rows(parent/'history.csv') if (parent/'history.csv').exists() else []
                config_path = next((p/'config.json' for p in [parent]+list(parent.parents)
                                    if p != folder.parent and (p/'config.json').exists()), None)
                # Only search ancestors inside the owned evidence folder.
                if config_path and not config_path.is_relative_to(folder):
                    config_path = None
                config = load(config_path) if config_path else {}
                if config_path:
                    track(config_path)
                ancestors=[p for p in [parent]+list(parent.parents) if p.is_relative_to(folder)]
                manifest_path=next((p/'manifest.json' for p in ancestors if (p/'manifest.json').exists()),None)
                manifest=load(manifest_path) if manifest_path else {}
                if manifest_path:
                    track(manifest_path)
                dataset = info.get('dataset') or next((part for part in parent.parts if part.startswith('CHEMBL')), None)
                mode = 'pair_delta' if owner == 'pair' and info.get('arm') not in ['direct', 'mse', 'aca'] and not any(t in label for t in ['aca_', 'branch_swap']) else 'single_molecule'
                if owner == 'pair' and 'branch_swap' in label:
                    mode = 'pair_delta'
                records.append(dict(owner=owner, path=str(parent), dataset=dataset, seed=info.get('seed', config.get('seed')),
                    arm=info.get('arm', info.get('model', parent.name)), mode=mode,
                    smoke=('smoke' in label or info.get('epochs_run') in [2, 3]),
                    status='saved_validation'+('_checkpoint' if checkpoint else '_no_checkpoint'),
                    validation_rows=len(predictions), cliff_rows=sum(int(float(r['cliff_mol'])) for r in predictions),
                    epochs_run=info.get('epochs_run', len(history)), parameters=info.get('parameters'),
                    elapsed_seconds=info.get('elapsed_seconds', info.get('train_seconds', info.get('training_seconds'))),
                    metrics=computed, stored_metric_differences=differences, label_precision=dtype,
                    checkpoint_path=str(checkpoint) if checkpoint else None,
                    checkpoint_matches_recorded_hash=(sha(checkpoint)==info['checkpoint_sha256']) if checkpoint and info.get('checkpoint_sha256') else None,
                    config_path=str(config_path) if config_path else None, config_sha256=sha(config_path) if config_path else None,
                    source_commit=info.get('code_commit',manifest.get('code_commit')),
                    manifest_path=str(manifest_path) if manifest_path else None,
                    training_source_hashes_record=manifest.get('source_sha256',config.get('source_hashes',{})),
                    test_evaluated_record=info.get('test_evaluated', 'not_recorded'),
                    scope='saved artifact arithmetic; not checkpoint replay or strict current-source reproduction'))
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(dict(path=str(path), error=str(exc)))
        for path in sorted(folder.rglob('history.csv')):
            if not (path.parent/'validation_predictions.csv').exists():
                track(path)
                incomplete.append(dict(owner=owner, path=str(path.parent), history_rows=len(rows(path)),
                    status='history_without_validation_predictions_unconfirmed_completion'))
    # Dedicated authoritative three-seed direct FPPool comparison.
    selected=[]
    for seed in [42,43,44]:
        parent = OLD/'eval_out/fppool_pilot/20260919_211456' if seed==42 else OLD/f'eval_out/fppool_multiseed/20260920_seeds43_44/seed{seed}'
        for arm in ['baseline','fppool']:
            row=next(r for r in records if r['path']==str(parent/arm))
            assert all(abs(v)<1e-12 for v in row['stored_metric_differences'].values())
            selected.append(dict(seed=seed, arm=arm, metrics=row['metrics'], elapsed_seconds=row['elapsed_seconds']))
    means={arm:{key:statistics.mean(r['metrics'][key] for r in selected if r['arm']==arm)
                for key in ['overall_rmse','cliff_rmse']} for arm in ['baseline','fppool']}
    improved=sum(next(r for r in selected if r['arm']=='fppool' and r['seed']==s)['metrics']['cliff_rmse'] <
                 next(r for r in selected if r['arm']=='baseline' and r['seed']==s)['metrics']['cliff_rmse'] for s in [42,43,44])
    assert improved==1 and means['fppool']['cliff_rmse']>means['baseline']['cliff_rmse']
    # Verify saved explanatory sensitivity summaries without generating new explanations.
    explanation=OLD/'eval_out/explanation_diagnostics/20260930_validation_v4'
    details=rows(explanation/'molecule_diagnostics.csv')
    checks=[]
    for table in rows(explanation/'faithfulness_by_seed.csv'):
        subset=[r for r in details if r['dataset']==table['dataset'] and r['model']==table['model'] and r['seed']==table['seed']
                and (table['subset']=='all' or int(r['cliff_mol'])==int(table['subset']=='cliff'))]
        if not subset:
            continue
        assert len(subset)==int(table['n'])
        values=[float(r['excess_change']) for r in subset]
        for row in subset:
            random=json.loads(row['random_changes'])
            assert len(random)==20
            # Historical explainer used NumPy float32 reduction, not Python float64 mean.
            assert abs(float(np.asarray(random,dtype=np.float32).mean())-float(row['random_mean']))<1e-12
            assert abs(float(row['top_change'])-float(row['random_mean'])-float(row['excess_change']))<1e-12
        reconstructed={'mean_excess':statistics.mean(values), 'median_excess':statistics.median(values),
                       'top_above_random_fraction':sum(v>0 for v in values)/len(values),
                       'zero_attribution_fraction':sum(r['attribution_all_zero']=='True' for r in subset)/len(subset)}
        assert all(abs(reconstructed[k]-float(table[k]))<1e-12 for k in reconstructed)
        checks.append(dict(dataset=table['dataset'], model=table['model'], seed=int(table['seed']), subset=table['subset'], n=len(subset), **reconstructed))
    for name in ['molecule_diagnostics.csv','faithfulness_by_seed.csv','stability_summary.csv','verification.json','报告.md','displayed_cases.csv']:
        track(explanation/name)
    # Snapshot current repo versions/status; originals never mutated.
    repo_states={}
    for name, repo in [('original',OLD),('pair',ROOT),('baselines',BASE)]:
        repo_states[name]={}
        for key, command in [('head',['rev-parse','HEAD']),('status',['status','--short'])]:
            probe=subprocess.run(['git','-C',str(repo),*command],capture_output=True,text=True,encoding='utf-8',errors='replace')
            repo_states[name][key]=probe.stdout.strip() if probe.returncode==0 else None
            if probe.returncode:
                repo_states[name][key+'_error']=probe.stderr.strip()
    duplicates={}
    for record in records:
        checkpoint=record['checkpoint_path']
        if checkpoint:
            duplicates.setdefault(hashes[checkpoint],[]).append(record['path'])
    result=dict(scope='All discoverable saved validation runs under original/eval_out, pair/artifacts, baselines/results; not all historical experiments anywhere.',
        records=records, incomplete=incomplete, parse_errors=errors, repo_states=repo_states,
        identical_checkpoint_groups=[paths for paths in duplicates.values() if len(paths)>1],
        fppool_direct=dict(runs=selected, means=means, improved_cliff_seeds=improved, go=False),
        explanation=dict(existing=True, models=12, unique_task_molecules=len({(r['dataset'],r['source_row']) for r in details}),
                         molecule_explanations=len(details), independently_recomputed_summaries=checks,
                         stability_saved=rows(explanation/'stability_summary.csv'),
                         limitation='Hidden-representation Gradient x Input; not graph-input masking or biochemical mechanism; stability table not rederived here.'),
        input_sha256=hashes)
    (output/'inventory.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    verification=dict(saved_runs=len(records), incomplete_histories=len(incomplete), parse_errors=len(errors),
                      protected_inputs=len(hashes), inputs_unchanged=all(sha(Path(p))==v for p,v in hashes.items()),
                      fppool_direct_checks=18, explanation_summary_checks=len(checks)*4,
                      no_training=True, no_test_prediction_or_label_evaluation=True)
    assert verification['inputs_unchanged'] and not errors
    (output/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(json.dumps(verification,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    main(parser.parse_args().output)
