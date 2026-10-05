"""Independent standard-library reconstruction of the saved stratified comparisons."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def read(path):
    with path.open(encoding='utf-8',newline='') as f: return list(csv.DictReader(f))


def mean(values): return math.fsum(values)/len(values)


def quantile(values,p):
    values=sorted(values); pos=(len(values)-1)*p
    lo=math.floor(pos); hi=math.ceil(pos)
    return values[lo]+(values[hi]-values[lo])*(pos-lo)


def main(out):
    rules=json.loads((out/'frozen_rules.json').read_text())
    records=json.loads((out/'comparison.json').read_text())
    audit=json.loads((out/'artifact_audit.json').read_text())
    assert len(records)==18
    checks=0; max_difference=0.; unique_files=set(); prediction_checks=0
    for dataset in rules['datasets']:
        folder=out/dataset
        frame=read(folder/'aligned_squared_errors.csv')
        support=json.loads((folder/'support.json').read_text())
        for model in rules['models']:
            for seed in rules['seeds']:
                suffix=f'results/{model}/{dataset}/seed_{seed}/validation_predictions.csv'
                matches=[p for p in audit['source_sha256'] if p.replace('\\','/').endswith(suffix)]
                assert len(matches)==1
                predictions={int(r['source_row']):r for r in read(Path(matches[0]))}
                assert set(predictions)=={int(r['source_row']) for r in frame}
                for row in frame:
                    original=predictions[int(row['source_row'])]
                    assert original['smiles']==row['smiles']
                    assert original['cliff_mol']==row['cliff_mol']
                    assert abs(float(original['y'])-float(row['y']))<1e-12
                    error=(float(original['y_pred'])-float(row['y']))**2
                    assert math.isfinite(error)
                    key=f'{model}_{seed}_squared_error'
                    assert abs(error-float(row[key]))<1e-12
                    row[key]=error
                    prediction_checks+=1
        for row in frame:
            value=float(row['y'])
            expected_bin=sum(value>=edge for edge in support['activity_edges'])
            assert expected_bin==int(row['activity_bin'])
            assert int(row['coverage_bin'])==int(float(row['neighbor_similarity'])>=support['train_loo_similarity_median'])
        for record in [r for r in records if r['dataset']==dataset]:
            model=record['model']
            cols=['activity_bin'] if record['design']=='activity' else ['activity_bin','coverage_bin']
            cells={}
            for row in frame:
                key=tuple(int(row[c]) for c in cols)
                cells.setdefault(key,{0:[],1:[]})[int(row['cliff_mol'])].append(row)
            retained={key:values for key,values in cells.items() if min(len(values[0]),len(values[1]))>=3}
            support_design=next(d for d in support['designs'] if d['design']==record['design'])
            total_counts={c:sum(len(v[c]) for v in retained.values()) for c in [0,1]}
            assert total_counts[0]==support_design['retained_noncliff']
            assert total_counts[1]==support_design['retained_cliff']
            assert total_counts[0]==record['retained_noncliff']
            assert total_counts[1]==record['retained_cliff']
            eligible=all(total_counts[c]>=20 and total_counts[c]/sum(int(r['cliff_mol'])==c for r in frame)>=.5 for c in [0,1])
            assert eligible==record['eligible']
            denom=sum(min(len(v[0]),len(v[1])) for v in retained.values())
            gaps=[]
            for seed in [42,43,44]:
                gap=0.
                for values in retained.values():
                    weight=min(len(values[0]),len(values[1]))/denom
                    gap+=weight*(mean([float(r[f'{model}_{seed}_squared_error']) for r in values[1]])-
                                 mean([float(r[f'{model}_{seed}_squared_error']) for r in values[0]]))
                gaps.append(gap)
            for actual,expected in zip(gaps,record['seed_gaps']):
                difference=abs(actual-expected); assert difference<1e-12
                max_difference=max(max_difference,difference); checks+=1
            assert abs(mean(gaps)-record['matched_mse_gap'])<1e-12
            bootstrap=[float(r[0]) for r in csv.reader((folder/f'{model}_{record["design"]}_bootstrap.csv').open())]
            assert len(bootstrap)==1000
            interval=[quantile(bootstrap,.025),quantile(bootstrap,.975)]
            assert max(abs(a-b) for a,b in zip(interval,record['bootstrap_interval']))<1e-12
            assert record['positive_and_stable']==(eligible and all(g>0 for g in gaps) and interval[0]>0)
        identity=json.loads((folder/'identity_audit.json').read_text())
        for path,digest in identity['source_sha256'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
            unique_files.add(path)
    assert audit['historical_prediction_sets']==27 and not audit['training_started']
    for path,digest in audit['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
        unique_files.add(path)
    qualified={model:[] for model in rules['models']}
    for model in rules['models']:
        for dataset in rules['datasets']:
            pair=[r for r in records if r['model']==model and r['dataset']==dataset]
            if all(r['positive_and_stable'] for r in pair): qualified[model].append(dataset)
    go=sum(len(v)>=2 for v in qualified.values())>=2
    decision=json.loads((out/'decision.json').read_text())
    assert go==decision['go'] and qualified==decision['qualifying_tasks_by_model']
    result=dict(passed=True,comparisons=18,seed_gap_checks=checks,bootstrap_intervals_checked=18,
          original_prediction_row_checks=prediction_checks,
          maximum_seed_gap_difference=max_difference,unchanged_input_files=len(unique_files),
          independently_verified_decision=go,
          scope='Arithmetic/support/percentile/decision and hash checks; no claim of inferential validity or model replay.')
    (out/'independent_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    main(parser.parse_args().output)
