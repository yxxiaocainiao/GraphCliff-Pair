"""Aggregate a read-only audit_framework inventory; never train or evaluate test.

Run audit_framework.py first, then this file with INVENTORY OUTPUT_DIRECTORY.
Positive improvement_pct means lower RMSE. Copied recovery runs are excluded.
"""
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def main(inventory, output):
    data = read(inventory)
    records = [r for r in data['records'] if not r['smoke'] and 'recovered' not in r['path']]
    comparisons = []
    used = {}

    def select(stage, arm, dataset=None):
        return [r for r in records if stage in r['path'] and r['arm'] == arm
                and (dataset is None or r['dataset'] == dataset)]

    def compare(name, left, right):
        keys = lambda rs: {(r['dataset'], r['seed']): r for r in rs}
        a, b = keys(left), keys(right)
        assert len(a) == len(left) and len(b) == len(right), name
        for dataset in sorted({k[0] for k in a.keys() & b.keys()}):
            pairs = [(a[k], b[k]) for k in sorted(a.keys() & b.keys()) if k[0] == dataset]
            result = dict(comparison=name, dataset=dataset, seeds=[x['seed'] for x, y in pairs], metrics={})
            for metric in ['overall_rmse', 'cliff_rmse']:
                av = [x['metrics'][metric] for x, y in pairs]
                bv = [y['metrics'][metric] for x, y in pairs]
                am, bm = statistics.mean(av), statistics.mean(bv)
                result['metrics'][metric] = dict(baseline_mean=am, candidate_mean=bm,
                    improvement_pct=100*(am-bm)/am, improved_seeds=sum(y<x for x,y in zip(av,bv)),
                    per_seed=[dict(seed=x['seed'], baseline=x['metrics'][metric],candidate=y['metrics'][metric]) for x,y in pairs])
            result['joint_improved_seeds'] = sum(all(y['metrics'][m]<x['metrics'][m] for m in ['overall_rmse','cliff_rmse']) for x,y in pairs)
            comparisons.append(result)
            for x,y in pairs:
                for r in [x,y]:
                    used[r['path']] = r

    direct = [r for r in records if 'fppool_multiseed' in r['path'] or 'fppool_pilot' in r['path']]
    compare('direct_fppool_vs_baseline',[r for r in direct if r['arm']=='baseline'],[r for r in direct if r['arm']=='fppool'])
    compare('residual3979_vs_baseline',select('fppool_residual','baseline'),select('fppool_residual','residual_fppool'))
    residual = lambda arm: [r for r in select('cross_task_residual',arm) if 'bridge_' not in r['path']]
    compare('residual_fppool_vs_baseline',residual('baseline'),residual('residual_fppool'))
    compare('residual_fppool_vs_gmt',select('gmt_residual','residual_gmt'),residual('residual_fppool'))
    compare('true_membership_vs_permuted',select('membership_ablation','residual_fppool'),residual('residual_fppool'))
    for arm in ['direct','pair_mlp','global']:
        compare('cross_vs_'+arm,select('interaction_seed',arm),select('interaction_seed','cross'))
    for arm in ['full','short']:
        compare('branch_cross_vs_'+arm,select('branch_swap_screen',arm),select('branch_swap_screen','cross'))
    for candidate,baseline in [('global_fp','global'),('cross_fp','cross'),('global_dynamic','global'),('cross_dynamic','cross'),('cross_fp_dynamic','direct')]:
        compare(candidate+'_vs_'+baseline,select('interaction_seed',baseline),select('ablation_seed',candidate))
    for baseline in ['mse','uniform','permuted']:
        compare('balanced_vs_'+baseline,select('component_pair_pilot',baseline),select('component_pair_pilot','balanced'))
    compare('aca_vs_mse',select('aca_pilot_','mse'),select('aca_pilot_','aca'))
    reliability_path = ROOT/'artifacts/reliability_phase_a_20261005_v3/results.json'
    reliability = read(reliability_path)
    for r in reliability['results']:
        a,b = (r['arms'][arm]['curve_mean_rmse'] for arm in ['generic','augmented'])
        comparisons.append(dict(comparison='roughness_vs_generic',dataset=r['dataset'],seeds=[r['seed']],metrics={
            'mean_selective_rmse':dict(baseline_mean=a,candidate_mean=b,improvement_pct=100*(a-b)/a,improved_seeds=int(b<a))}))
    input_hashes = dict(data['input_sha256'])
    input_hashes[str(Path(inventory).resolve())] = sha(inventory)
    input_hashes[str(reliability_path)] = sha(reliability_path)
    pair_root = Path('D:/GraphCliff-main/eval_out/component_pair_pilot/20261001_frozen/three_seed_report')
    pair_rows = list(csv.DictReader((pair_root/'legal_pair_predictions.csv').open(encoding='utf-8-sig')))
    pair_checks = 0
    for summary in csv.DictReader((pair_root/'pair_metrics.csv').open(encoding='utf-8-sig')):
        group=[r for r in pair_rows if all(r[k]==summary[k] for k in ['dataset','seed','arm'])
               and (summary['subset'] != 'cliff_pairs' or r['cliff_pair'].lower() in ['true','1','1.0'])]
        assert len(group)==int(summary['n_pairs']), (summary, len(group))
        mae=statistics.mean(abs(float(r['delta_prediction'])-float(r['delta_true'])) for r in group)
        assert abs(mae-float(summary['signed_delta_mae']))<1e-10, summary
        pair_checks += 1
    for name in ['legal_pair_predictions.csv','pair_metrics.csv']:
        p=pair_root/name; input_hashes[str(p)] = sha(p)
    assert all(sha(path)==value for path,value in input_hashes.items()), 'An input changed'
    dest=Path(output); dest.mkdir(parents=True,exist_ok=True)
    def public_path(path):
        for root,label in [(str(ROOT),'pair'),('D:\\GraphCliff-main','original'),('D:\\WORK_SPACE\\WORK_SPACE\\my_work','my_work')]:
            if path.startswith(root): return label+'/'+path[len(root):].lstrip('\\/').replace('\\','/')
        return Path(path).name
    provenance=dict(date='2026-10-06',source_commit='8df3c39f5e7d6f87081784cda5c8089e6a66ab28',
        interpretation='Post-hoc reassessment; original gates and decisions unchanged.',
        fresh_audit=read(Path(inventory).parent/'verification.json'),pair_mae_rows_checked=pair_checks,
        protected_inputs_unchanged=True,new_training=0,new_test_evaluation=0,
        inputs={public_path(p):h for p,h in input_hashes.items()},
        selected_runs=[dict(source=public_path(r['path']),dataset=r['dataset'],seed=r['seed'],arm=r['arm'],
            config_sha256=r['config_sha256'],source_commit=r['source_commit'],parameters=r['parameters'],elapsed_seconds=r['elapsed_seconds']) for r in used.values()],
        analysis_script_sha256=sha(__file__),scope='Saved validation arithmetic, not checkpoint replay; reliability uses existing phase-A aggregates.')
    for name,value in [('comparisons.json',comparisons),('provenance.json',provenance)]:
        (dest/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(comparisons=len(comparisons),selected_runs=len(used),pair_checks=pair_checks,inputs_unchanged=True)))

if __name__ == '__main__':
    main(*sys.argv[1:])
