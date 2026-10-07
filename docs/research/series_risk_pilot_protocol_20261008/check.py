"""Metadata and synthetic contract check only; never opens labels or runs models."""
from collections import Counter, defaultdict
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path


def main():
    folder = Path(__file__).resolve().parent
    root = folder.parents[2]
    protocol = json.loads((folder / 'protocol.json').read_text(encoding='utf-8'))
    for name, expected in protocol['input_sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    rows = json.loads((root / protocol['private_roles_manifest']).read_text(encoding='utf-8'))
    assert all(set(r) == {'source_row', 'component', 'role', 'oof_fold'} for r in rows)
    assert len({r['source_row'] for r in rows}) == len(rows)
    assert dict(Counter(r['role'] for r in rows)) == protocol['roles']
    components = defaultdict(set)
    for row in rows:
        components[row['component']].add((row['role'], row['oof_fold']))
    assert all(len(assignments) == 1 for assignments in components.values())
    assert all(r['oof_fold'] is None for r in rows if r['role'] != 'fit')
    fit = {r['source_row'] for r in rows if r['role'] == 'fit'}
    for k, count in enumerate(protocol['oof_query_rows']):
        query = {r['source_row'] for r in rows if r['role'] == 'fit' and r['oof_fold'] == k}
        reference = fit - query
        assert len(query) == count and len(reference) == protocol['oof_reference_rows'][k]
        assert not query & reference and query | reference == fit
    budget = protocol['budget']
    assert sum(protocol['oof_query_rows']) + protocol['roles']['evaluation'] == budget['chemprop_predicted_rows']
    assert budget['risk_predicted_rows'] == 4 * protocol['roles']['evaluation']
    assert budget['chemprop_fits'] == budget['auxiliary_activity_rf_fits'] == budget['risk_rf_fits'] == 4
    assert budget['kde_fits'] == 2 * (len(protocol['arms']) - 1) == 6
    assert protocol['real_training_authorized'] is False
    counts = [(F(str(c)) * protocol['roles']['evaluation']).__ceil__() for c in protocol['coverage_grid']]
    assert counts == protocol['expected_accept_counts']
    # Tie/partition/exchange identity on synthetic examples; no real errors read.
    error2, risk_a, risk_b = [F(x*x) for x in (0, 1, 2, 3, 4, 5)], [0, 0, 1, 2, 3, 4], [2, 2, 0, 1, 3, 4]
    canonical = ['b', 'a', 'c', 'd', 'e', 'f']
    def order(scores):
        return sorted(range(6), key=lambda i: (scores[i], canonical[i], i))
    assert order(risk_a)[:2] == [1, 0]
    for m in range(1, 7):
        a, b = set(order(risk_a)[:m]), set(order(risk_b)[:m])
        common, added, removed = a & b, b - a, a - b
        assert len(added) == len(removed)
        assert common | removed == a and common | added == b
        assert not common & added and not common & removed and not added & removed
        delta = (sum(error2[i] for i in b) - sum(error2[i] for i in a)) / m
        assert delta == (sum(error2[i] for i in added) - sum(error2[i] for i in removed)) / m
        if m == 6:
            assert delta == 0 and a == b
    print(json.dumps({'status': 'passed', 'input_hashes_checked': 6,
        'manifest_rows_checked': len(rows), 'private_identities_exported': 0,
        'synthetic_exchange_sizes': 6, 'real_fits_loads_predictions_feature_computations': 0,
        'real_labels_or_predictions_parsed': 0, 'official_test_reads': 0}, indent=2))


if __name__ == '__main__':
    main()
