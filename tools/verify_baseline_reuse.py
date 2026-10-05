"""Independent standard-library verification of the saved validation comparison."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def read(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def mean(values):
    return math.fsum(values) / len(values)


def verify(folder):
    wide = read(folder/'aligned_validation_predictions.csv')
    truth = [float(r['y']) for r in wide]
    assert len(wide) == 292
    assert len({r['source_row'] for r in wide}) == len(wide)
    assert sum(int(r['cliff_mol']) for r in wide) == 128
    metrics = read(folder/'validation_metrics.csv')
    assert len(metrics) == 14
    maximum = 0.
    for record in metrics:
        seed = record['seed']
        key = record['model'] + ('_'+str(int(float(seed))) if seed else '')
        error = [float(r[key])-y for r,y in zip(wide,truth)]
        for group in ['overall','cliff','noncliff']:
            idx = [i for i,r in enumerate(wide) if group=='overall' or
                   int(r['cliff_mol']) == int(group=='cliff')]
            actual = math.sqrt(mean([error[i]**2 for i in idx]))
            difference = abs(actual-float(record[group+'_rmse']))
            assert difference < 1e-12
            maximum = max(maximum,difference)
    for record in read(folder/'model_summary.csv'):
        group = [r for r in metrics if r['model'] == record['model']]
        assert len(group) == int(record['runs'])
        for metric in ['overall','cliff','noncliff']:
            values = [float(r[metric+'_rmse']) for r in group]
            assert abs(mean(values)-float(record[metric+'_mean'])) < 1e-12
            if metric != 'noncliff' and len(values)>1:
                sd = math.sqrt(math.fsum((v-mean(values))**2 for v in values)/(len(values)-1))
                assert abs(sd-float(record[metric+'_sd'])) < 1e-12
            elif metric != 'noncliff':
                assert not record[metric+'_sd']
    gc = [float(r['graphcliff_42'])-y for r,y in zip(wide,truth)]
    for record in read(folder/'error_overlap.csv'):
        idx = [i for i,r in enumerate(wide) if record['subset']=='overall' or int(r['cliff_mol'])==1]
        key = record['model'] + ('' if record['model']=='svm' else '_42')
        other = [float(r[key])-y for r,y in zip(wide,truth)]
        delta = [gc[i]**2-other[i]**2 for i in idx]
        trimmed = delta.copy()
        trimmed.pop(max(range(len(delta)),key=lambda j: abs(delta[j])))
        assert abs(mean(delta)-float(record['graphcliff_minus_baseline_mse'])) < 1e-12
        assert abs(mean(trimmed)-float(record['removed_largest_absolute_difference_mse'])) < 1e-12
        assert sum(abs(other[i]) < abs(gc[i]) for i in idx) == int(record['baseline_lower_absolute_error_count'])
        k = math.ceil(len(idx)*.2)
        a = set(sorted(idx,key=lambda i: abs(gc[i]))[-k:])
        b = set(sorted(idx,key=lambda i: abs(other[i]))[-k:])
        assert k == int(record['top20percent_count'])
        assert len(a & b) == int(record['top_error_intersection'])
        assert abs(len(a & b)/len(a | b)-float(record['top_error_jaccard'])) < 1e-12
        x,y = [gc[i] for i in idx], [other[i] for i in idx]
        mx,my = mean(x),mean(y)
        corr = math.fsum((a-mx)*(b-my) for a,b in zip(x,y)) / math.sqrt(
            math.fsum((a-mx)**2 for a in x)*math.fsum((b-my)**2 for b in y))
        assert abs(corr-float(record['signed_error_correlation'])) < 1e-12
    checked = set()
    for name in ['identity_audit.json','prediction_audit.json','historical_binding_audit.json']:
        audit = json.loads((folder/name).read_text(encoding='utf-8'))
        assert audit['passed']
        for path,expected in audit['source_sha256'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
            checked.add(path)
    result = dict(passed=True, prediction_sets=14, validation_rows=292,
                  cliff_rows=128, rmse_checks=42, overlap_records=10,
                  maximum_rmse_difference=maximum, unchanged_input_files=len(checked),
                  scope='Independent recomputation, not checkpoint forward replay or full historical training reproduction.')
    (folder/'independent_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    verify(parser.parse_args().output)
