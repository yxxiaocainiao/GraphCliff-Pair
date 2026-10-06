"""Zero-fit, post-hoc control on pinned phase-A caches; no checkpoint/test use."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from run_phase_a import cq_function, metrics, save, sha


def combination(reference, queries):
    columns = ['rf_var', 'sali_mean']
    means = reference[columns].mean().to_numpy()
    std = reference[columns].std(ddof=0).to_numpy() + 1e-9
    score = ((queries[columns].to_numpy()-means)/std).sum(axis=1)
    assert np.isfinite(score).all()
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.input, args.output = args.input.resolve(), args.output.resolve()
    assert not args.output.exists(), 'Refuse to overwrite an earlier review'
    protocol_path = ROOT/'experiments/reliability_base/fixed_protocol.json'
    fixed_path = ROOT/'experiments/reliability_base/incremental_evidence_protocol.md'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    bindings = json.loads((args.input/'artifact_binding.json').read_text(encoding='utf-8'))
    author = ROOT/'artifacts/roughness_sources_20261005/qsar-landscape-roughness/src/conformal.py'
    inputs = {p: sha(p) for p in [protocol_path, fixed_path, author, Path(__file__),
              ROOT/'experiments/reliability_base/run_phase_a.py', args.input/'results.json', args.input/'artifact_binding.json']}
    cq = cq_function(author)
    old = json.loads((args.input/'results.json').read_text(encoding='utf-8'))
    results = []
    for task in protocol['tasks']:
        dataset = task['dataset']
        frames = {}
        for role in ['oof', 'calibration', 'evaluation']:
            path = args.input/dataset/(role+'.csv')
            assert sha(path) == bindings[str(path.relative_to(args.input))], path
            inputs[path] = sha(path)
            frames[role] = pd.read_csv(path)
            assert frames[role].source_row.is_unique
        oof, cal, ev = (frames[r] for r in ['oof','calibration','evaluation'])
        manifest = ROOT/'artifacts/reliability_protocol_20261005_v2'/(dataset+'.json')
        assert sha(manifest) == task['partition_manifest_sha256']
        inputs[manifest] = sha(manifest)
        roles = json.loads(manifest.read_text(encoding='utf-8'))['roles']
        for name, frame in frames.items():
            assert set(frame.source_row)==set(roles['fit' if name=='oof' else name])
            assert np.isfinite(frame[['prediction','y','rf_var','sali_mean']]).all().all()
        assert not set(oof.canonical)&(set(cal.canonical)|set(ev.canonical))
        assert not set(cal.canonical)&set(ev.canonical)
        saved = next(r for r in old['results'] if r['dataset']==dataset)
        arms = {}
        for arm in ['distance','sali','generic','augmented','simple_combination']:
            sc, se = (combination(oof,frame) if arm=='simple_combination' else frame[arm+'_risk'].to_numpy() for frame in [cal,ev])
            # Existing metrics also calculates intervals; only ranking curves are retained for signed z scores.
            value = metrics(ev,se,cal,sc,cq,protocol)
            arms[arm] = {k:value[k] for k in ['curve_rmse','curve_mean_rmse']}
            if arm!='simple_combination':
                assert np.allclose(value['curve_rmse'],saved['arms'][arm]['curve_rmse'],atol=1e-12,rtol=0)
        baseline = arms['simple_combination']['curve_mean_rmse']
        gain = 100*(baseline-arms['augmented']['curve_mean_rmse'])/baseline
        results.append(dict(dataset=dataset,seed=42,arms=arms,augmented_vs_combination_pct=gain))
    gains = [r['augmented_vs_combination_pct'] for r in results]
    passed = sum(g>0 for g in gains)>=2 and np.mean(gains)>0 and min(gains)>=-5
    assert all(sha(p)==value for p,value in inputs.items()), 'Input changed during review'
    args.output.mkdir(parents=True)
    save(args.output/'results.json', dict(results=results,mean_relative_improvement_pct=float(np.mean(gains)),
         gate_G0_pass=bool(passed),new_fits=0,new_checkpoint_predictions=0,official_test_rows=0,
         source_commit='14b1d87',plan='Registered before new control computation; existing evaluation previously observed.',
         decision='Prepare bounded replication; method-contribution gate unresolved' if passed else 'Archive current learned-augmentation candidate; do not train seed43/44',
         input_sha256={str(p.relative_to(ROOT)).replace('\\','/'):value for p,value in inputs.items()}))
    print(json.dumps(dict(gains=gains,passed=bool(passed),new_fits=0)))


if __name__ == '__main__':
    main()
