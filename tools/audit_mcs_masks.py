"""Bounded structural check, using existing train pairs; no model or activity reads."""
import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import rdFMCS


def masks(mol, query):
    matches = mol.GetSubstructMatches(query, uniquify=True, maxMatches=256)
    return sorted({tuple(sorted(match)) for match in matches}), len(matches) >= 256


def analyze(left, right):
    params = rdFMCS.MCSParameters()
    params.Timeout = 2
    params.StoreAll = True
    params.AtomCompareParameters.MatchValences = True
    params.BondCompareParameters.RingMatchesRingOnly = True
    params.BondCompareParameters.CompleteRingsOnly = True
    result = rdFMCS.FindMCS([left, right], params)
    if result.canceled or not result.numAtoms:
        return {'complete': False, 'canceled': result.canceled}
    queries = list(result.degenerateSmartsQueryMolDict.values())
    assert queries, 'StoreAll did not return queries'
    sides = []
    for mol in [left, right]:
        all_sets, capped = set(), False
        for query in queries:
            sets, hit_cap = masks(mol, query)
            all_sets.update(sets)
            capped |= hit_cap
        assert all_sets
        ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
        signatures = {tuple(sorted(ranks[i] for i in mask)) for mask in all_sets}
        sides.append({'mask_sets': len(all_sets), 'rank_multisets': len(signatures), 'capped': capped})
    return {'complete': True, 'mcs_atoms': result.numAtoms, 'degenerate_queries': len(queries),
            'sides': sides, 'different_environment_certificate': any(s['rank_multisets'] > 1 for s in sides),
            'capped': any(s['capped'] for s in sides)}


def synthetic():
    cases = [('unique', 'CCCO', 'CCCO'), ('symmetry_only', 'CC(C)CO', 'CCCO'),
             ('different_environments', 'CCC(C)CO', 'CCCO')]
    rows = []
    for name, a, b in cases:
        left, right = map(Chem.MolFromSmiles, (a, b))
        result = rdFMCS.FindMCS([left, right], timeout=2, matchValences=True,
                              ringMatchesRingOnly=True, completeRingsOnly=True)
        assert not result.canceled
        query = Chem.MolFromSmarts(result.smartsString)
        scores = [atom.GetDegree() + atom.GetAtomicNum() / 10 for atom in left.GetAtoms()]
        sets, capped = masks(left, query)
        assert not capped
        rng = random.Random(20261005)
        selected, losses, averages = set(), [], []
        for _ in range(24):
            order = list(range(left.GetNumAtoms()))
            rng.shuffle(order)
            permuted = Chem.RenumberAtoms(left, order)
            match = tuple(sorted(order[i] for i in permuted.GetSubstructMatch(query)))
            selected.add(match)
            losses.append(sum(scores[i] ** 2 for i in match))
            mapped_sets = {tuple(sorted(order[i] for i in m)) for m in masks(permuted, query)[0]}
            assert mapped_sets == set(sets)
            averages.append(sum(sum(scores[i] ** 2 for i in m) for m in mapped_sets) / len(mapped_sets))
        assert max(averages) - min(averages) < 1e-10
        rows.append({'case': name, 'synthetic_smiles': [a, b], 'permutations': 24,
                     'mask_statistics': analyze(left, right), 'selected_mask_sets': len(selected),
                     'first_match_common_square_range': max(losses) - min(losses),
                     'all_unique_masks_mean_common_square_range': max(averages) - min(averages)})
    assert rows[0]['first_match_common_square_range'] < 1e-10
    assert rows[1]['first_match_common_square_range'] < 1e-10
    assert rows[2]['first_match_common_square_range'] > 1e-6
    return rows


def inspect_manifest(manifest, csv_root):
    started = time.monotonic()
    raw = manifest.read_bytes()
    meta = json.loads(raw)
    source = csv_root / (meta['dataset'] + '.csv')
    initial_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert initial_hash == meta['input_sha256']
    train, valid = set(meta['train_rows']), set(meta['valid_rows'])
    assert train.isdisjoint(valid)
    pairs = sorted({(int(p['query']), int(p['reference'])) for p in meta['train_pairs']})[:24]
    selected_rows = {i for pair in pairs for i in pair}
    assert selected_rows <= train
    molecules = {}
    with source.open(encoding='utf-8-sig', newline='') as handle:
        for row_number, row in enumerate(csv.DictReader(handle)):
            if row_number in selected_rows:
                assert row['split'] == 'train'
                mol = Chem.MolFromSmiles(row['smiles'])
                assert mol is not None
                molecules[row_number] = mol
    assert set(molecules) == selected_rows
    records = [dict(query=i, reference=j, **analyze(molecules[i], molecules[j])) for i, j in pairs]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == initial_hash
    summary = {'dataset': meta['dataset'], 'manifest_sha256': hashlib.sha256(raw).hexdigest(),
               'csv_sha256': initial_hash, 'pairs': len(records),
               'complete': sum(x['complete'] for x in records),
               'capped': sum(x.get('capped', False) for x in records),
               'different_environment_certificates': sum(x.get('different_environment_certificate', False) for x in records),
               'multiple_mask_sets': sum(any(s['mask_sets'] > 1 for s in x.get('sides', [])) for x in records),
               'seconds': time.monotonic() - started}
    print(json.dumps(summary), flush=True)
    return summary, records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifests', nargs='+', type=Path, required=True)
    parser.add_argument('--csv-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refuse to overwrite existing output')
    toy = synthetic()
    results = [inspect_manifest(p, args.csv_root) for p in args.manifests]
    uncertainty = any((s['pairs'] - s['complete'] + s['capped']) / s['pairs'] > 0.25 for s, _ in results)
    passed = all(s['different_environment_certificates'] >= 2 for s, _ in results) and not uncertainty
    output = {'rdkit_version': rdBase.rdkitVersion, 'synthetic': toy, 'summaries': [s for s, _ in results],
              'raw_records_local_only': [r for _, r in results], 'structural_screen': 'continue_problem_audit' if passed else 'uncertain_or_stop',
              'training': 0, 'test_evaluation': 0, 'activity_fields_used': [],
              'limits': 'No performance or attribution result; rank multiset is a sufficient difference certificate, not exact orbit enumeration.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
