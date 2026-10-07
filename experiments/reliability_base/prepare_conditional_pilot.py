"""Prepare metadata and CLI commands only. There is deliberately no execute mode."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'docs/research/conditional_pilot_prepare_20261008/protocol.json'


def jobs_for(rows, protocol, output):
    if any(set(r) != {'source_row', 'component', 'role', 'oof_fold'} for r in rows):
        raise ValueError('Role manifest must contain identities only')
    if len({r['source_row'] for r in rows}) != len(rows) or dict(Counter(r['role'] for r in rows)) != protocol['roles']:
        raise ValueError('Duplicate identities or incorrect role counts')
    assignments = defaultdict(set)
    for row in rows:
        assignments[row['component']].add((row['role'], row['oof_fold']))
        if (row['role'] == 'fit' and row['oof_fold'] not in (0, 1, 2)) or (row['role'] != 'fit' and row['oof_fold'] is not None):
            raise ValueError('Invalid OOF assignment')
    if any(len(v) != 1 for v in assignments.values()):
        raise ValueError('Component crosses roles or OOF folds')
    roles = {name: sorted(r['source_row'] for r in rows if r['role'] == name) for name in protocol['roles']}
    jobs = []
    for k in range(3):
        query = sorted(r['source_row'] for r in rows if r['role'] == 'fit' and r['oof_fold'] == k)
        fit = sorted(set(roles['fit']) - set(query))
        if len(query) != protocol['oof_query_rows'][k] or len(fit) != protocol['oof_reference_rows'][k]:
            raise ValueError('OOF counts changed')
        jobs.append(dict(name=f'fold{k}', fit=fit, query=query, monitor=roles['monitor']))
    jobs.append(dict(name='full', fit=roles['fit'], query=roles['evaluation'], monitor=roles['monitor']))
    cli = [sys.executable, '-u', '-c', 'from chemprop.cli.main import main; main()']
    for job in jobs:
        if set(job['fit']) & set(job['query']) or set(job['monitor']) & (set(job['fit']) | set(job['query'])):
            raise ValueError('Optimization, monitor and query overlap')
        if set(roles['calibration']) & (set(job['fit']) | set(job['query']) | set(job['monitor'])):
            raise ValueError('Calibration must remain unused')
        folder = output / protocol['task'] / job['name']
        model = protocol['point_model']
        job['train_command'] = cli + ['train', '-i', str(folder/'train_val.csv'), '-o', str(folder/'training'),
            '--smiles-columns', 'smiles', '--target-columns', 'y', '--splits-column', 'partition',
            '--task-type', 'regression', '--loss-function', 'mse', '--epochs', str(model['epochs']),
            '--patience', str(model['patience']), '--batch-size', str(model['batch_size']),
            '--num-workers', '0', '--accelerator', 'gpu', '--devices', '1', '--data-seed', '42',
            '--pytorch-seed', '42', '--save-data-splits']
        job['predict_command'] = cli + ['predict', '-i', str(folder/'query.csv'), '-o', str(folder/'predictions.csv'),
            '--model-paths', str(folder/'training/model_0/best.pt'), '--smiles-columns', 'smiles',
            '--num-workers', '0', '--accelerator', 'gpu', '--devices', '1']
    return jobs


def selfcheck():
    p = dict(roles=dict(fit=3, monitor=1, calibration=1, evaluation=2),
             oof_query_rows=[1, 1, 1], oof_reference_rows=[2, 2, 2],
             task='CHEMBL234_Ki', point_model=dict(epochs=50, patience=15, batch_size=64))
    rows = [dict(source_row=i, component=i, role='fit', oof_fold=i) for i in range(3)]
    rows += [dict(source_row=i, component=i, role=r, oof_fold=None)
             for i, r in enumerate(['monitor', 'calibration', 'evaluation', 'evaluation'], 3)]
    jobs = jobs_for(rows, p, ROOT/'artifacts/unwritten_synthetic_prepare')
    assert len(jobs) == 4 and jobs[-1]['query'] == [5, 6]
    assert sum(len(j['query']) for j in jobs) == 5
    for changes in ({'component': 3}, {'source_row': 1}, {'oof_fold': 8}, {'y': 0}):
        bad = [dict(r) for r in rows]
        bad[0].update(changes)
        try:
            jobs_for(bad, p, ROOT/'artifacts/unwritten_synthetic_prepare')
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid role metadata accepted')
    print(json.dumps(dict(status='passed', synthetic_jobs=4, invalid_manifests_rejected=4, fits=0)))


def prepare(output):
    output = output.resolve()
    if output == ROOT/'artifacts' or not output.is_relative_to(ROOT/'artifacts'):
        raise ValueError('Output must be a new child of this workspace artifacts directory')
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status='preparing', training_authorized=False, fits=0, predictions=0)
    def save(name, value):
        (output/name).write_bytes((json.dumps(value, indent=2, default=str)+'\n').encode('utf-8'))
    save('state.json', state)
    try:
        protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
        if protocol['real_training_authorized']:
            raise ValueError('This preparation contract must not authorize training')
        for name, expected in protocol['input_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
                raise ValueError('Pinned source hash changed: '+name)
        rows = json.loads((ROOT/protocol['private_roles_manifest']).read_text(encoding='utf-8'))
        jobs = jobs_for(rows, protocol, output)
        # Parse only: never call the CLI handler or create inputs/checkpoints.
        from chemprop.cli.main import construct_parser
        parser = construct_parser()
        parsed = []
        for job in jobs:
            parsed.append(dict(job=job['name'], train=vars(parser.parse_args(job['train_command'][4:])),
                               predict=vars(parser.parse_args(job['predict_command'][4:]))))
        save('jobs.json', jobs)
        save('parsed_cli.json', parsed)
        save('arm_bindings.json', protocol['arm_bindings'])
        summary = dict(task=protocol['task'], seed=protocol['seed'], jobs=len(jobs),
                       fit_rows=[len(j['fit']) for j in jobs], query_rows=[len(j['query']) for j in jobs],
                       monitor_rows=[len(j['monitor']) for j in jobs],
                       arms=protocol['arms'], budget=protocol['budget'],
                       cli_commands_parsed=8, real_input_csv_files_created=0,
                       real_features_or_labels_read=0, training_authorized=False, runnable_training=False,
                       weights_callable=protocol['arm_bindings']['IW-COND7'],
                       protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest())
        save('summary.json', summary)
        state.update(status='prepared_metadata_only', cli_commands_parsed=8)
        save('state.json', state)
        print(json.dumps(dict(status=state['status'], task=protocol['task'], jobs=4,
                              arms=5, cli_commands_parsed=8, fits=0, predictions=0)))
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        save('state.json', state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--selfcheck', action='store_true')
    group.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        prepare(args.output)
