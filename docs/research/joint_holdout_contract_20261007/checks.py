"""M58: set-membership audit only; never read labels, predictions or weights."""
import hashlib
import json
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/reliability_phase_a_20261005_v3'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit(folds, monitor, external):
    parts=[set(v) for v in folds.values()]; fit=set.union(*parts)
    assert len(parts)==3 and all(parts) and sum(map(len,parts))==len(fit)
    assert not (fit & monitor or fit & external or monitor & external)
    models=[fit-part for part in parts]
    witnesses=[]
    for k,l in [(k,l) for k in range(3) for l in range(3) if k!=l]:
        q,r=next(iter(parts[k])),next(iter(parts[l]))
        assert q not in models[k] and r in models[k]  # Original query model/reference contract.
        assert r not in models[l] and q in models[l]  # Naive other-fold OOF can depend on query.
        joint=fit-parts[k]-parts[l]
        assert not (parts[k] | parts[l]) & (joint | monitor)
        assert joint and all(joint!=m for m in models)
        witnesses.append((k,l))
    # External queries are excluded from every existing fold model and its monitor.
    for l in range(3): assert not (parts[l] | external) & (models[l] | monitor)
    # Zero-fit same-fold witness option changes the original fit-reference contract.
    for k in range(3): assert not parts[k] & models[k]
    return dict(fit_rows=len(fit),monitor_rows=len(monitor),external_query_rows=len(external),
        fold_rows=[len(p) for p in parts],existing_fold_fit_rows=[len(m) for m in models],
        naive_unsafe_ordered_fold_relations=len(witnesses),
        joint_exclusion_models=[dict(excluded_folds=[k,l],fit_rows=len(fit-parts[k]-parts[l]),
                                    reference_predictions_if_all_excluded_rows=len(parts[k])+len(parts[l]))
                                for k,l in combinations(range(3),2)])


def main():
    toy=audit({'0':[0,1],'1':[2,3],'2':[4,5]},{6},{7,8})
    assert toy['naive_unsafe_ordered_fold_relations']==6
    assert len(toy['joint_exclusion_models'])==3
    assert all(m['fit_rows']==2 for m in toy['joint_exclusion_models'])
    pinned=read(ROOT/'docs/research/query_reference_prediction_20261007/protocol.json')
    # Pin input bytes without parsing any molecular labels/predictions or checkpoint.
    for name,expected in pinned['inputs_sha256'].items():
        path=ROOT/name
        if path.suffix=='.pt': continue  # Historical binding was verified at M57; no weight reads here.
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
    manifest=read(ROOT/'artifacts/reliability_protocol_20261005_v2/CHEMBL234_Ki.json')
    folds=manifest['oof_folds']; roles=manifest['roles']
    fit=set(roles['fit']); monitor=set(roles['monitor'])
    external=set(roles['calibration'])|set(roles['evaluation'])
    result=audit(folds,monitor,external)
    assert fit==set().union(*map(set,folds.values()))
    jobs=[j for j in read(BASE/'prepare.json')['jobs'] if j['task']=='CHEMBL234_Ki']
    assert len(jobs)==4
    job_results=[]; sources=[Path(__file__),ROOT/'experiments/reliability_base/run_phase_a.py',
                           BASE/'prepare.json',ROOT/'artifacts/reliability_protocol_20261005_v2/CHEMBL234_Ki.json',
                           ROOT/'docs/research/query_reference_prediction_20261007/protocol.json']
    for job in jobs:
        name=job['name']; folder=BASE/'CHEMBL234_Ki'/name
        expected_fit=fit if name=='full' else fit-set(folds[name[-1]])
        expected_query=external if name=='full' else set(folds[name[-1]])
        assert set(job['fit'])==expected_fit and set(job['query'])==expected_query
        assert set(job['monitor'])==monitor and not expected_query & (expected_fit|monitor)
        cmd=read(folder/'train_command.json')
        assert cmd[4]=='train' and cmd[cmd.index('--splits-column')+1]=='partition'
        assert cmd[cmd.index('-i')+1]==str(folder/'train_val.csv')
        assert cmd[cmd.index('--epochs')+1]=='50' and cmd[cmd.index('--patience')+1]=='15'
        timing=read(folder/'fit_log/exit.json');assert timing['returncode']==0
        assert (folder/'training/model_0/best.pt').is_file()
        job_results.append(dict(name=name,fit_rows=len(expected_fit),query_rows=len(expected_query),
                                monitor_rows=len(monitor),historical_fit_seconds=timing['seconds']))
        sources += [folder/'train_command.json',folder/'fit_log/exit.json']
    # M57 private identity file contains roles/IDs/SMILES, no labels or predictions.
    identities=read(ROOT/'artifacts/query_reference_prediction_20261007/identities.json')
    ref_ids={row['source_row'] for row in identities if row['kind']=='reference'}
    assert len(ref_ids)==15 and ref_ids<=fit
    result.update(selected_case_references_per_fold={k:len(ref_ids&set(v)) for k,v in folds.items()},
                  existing_jobs=job_results,
                  complete_joint_reference_prediction_rows=sum(m['reference_predictions_if_all_excluded_rows'] for m in result['joint_exclusion_models']),
                  new_fits_executed=0,model_loads=0,predict_calls=0,official_test_rows_read=0,
                  parsed_label_or_prediction_values=0,pure_set_selfcheck='passed',real_identity_audit='passed')
    sources.append(ROOT/'artifacts/query_reference_prediction_20261007/identities.json')
    result['sources_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='sources_sha256'}))


if __name__=='__main__': main()
